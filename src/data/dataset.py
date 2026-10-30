from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.keywords import extract_topic_keyword_features

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


@dataclass(frozen=True)
class Vocab:
    token2id: dict[str, int]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 2,
        max_size: int | None = None,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        token2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if freq < min_freq:
                continue
            if max_size is not None and len(token2id) >= max_size:
                break
            token2id[token] = len(token2id)

        return cls(token2id=token2id)

    def encode(self, tokens: Sequence[str], max_len: int) -> list[int]:
        ids = [self.token2id.get(token, UNK_ID) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids.extend([PAD_ID] * (max_len - len(ids)))
        return ids

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.token2id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token2id={str(token): int(index) for token, index in data.items()})

    def __len__(self) -> int:
        return len(self.token2id)


class TextCNNDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocab: Vocab,
        label_column: str,
        label2id: dict[str, int],
        max_len: int,
    ) -> None:
        if label_column not in frame.columns:
            raise ValueError(f"Missing label column: {label_column}")
        if "tokens" not in frame.columns:
            raise ValueError("Missing tokens column")

        self.input_ids = [
            vocab.encode(str(tokens).split(), max_len=max_len)
            for tokens in frame["tokens"].tolist()
        ]
        self.labels = [
            label2id[str(label)]
            for label in frame[label_column].tolist()
        ]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return input_ids, labels


class TextCNNMultiTaskDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocab: Vocab,
        sentiment2id: dict[str, int],
        topic2id: dict[str, int],
        max_len: int,
    ) -> None:
        if "tokens" not in frame.columns:
            raise ValueError("Missing tokens column")
        if "sentiment" not in frame.columns or "topic" not in frame.columns:
            raise ValueError("Missing sentiment/topic columns")

        self.input_ids = [
            vocab.encode(str(tokens).split(), max_len=max_len)
            for tokens in frame["tokens"].tolist()
        ]
        self.sentiment_labels = [
            sentiment2id[str(label)]
            for label in frame["sentiment"].tolist()
        ]
        self.topic_labels = [
            topic2id[str(label)]
            for label in frame["topic"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
            "topic_label": torch.tensor(self.topic_labels[index], dtype=torch.long),
        }


def collate_multitask(
    batch: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    sentiment_labels = torch.stack([item["sentiment_label"] for item in batch])
    topic_labels = torch.stack([item["topic_label"] for item in batch])
    return input_ids, sentiment_labels, topic_labels


class EmbeddingMultiTaskDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        frame: pd.DataFrame,
        sentiment2id: dict[str, int],
        topic2id: dict[str, int],
    ) -> None:
        if len(embeddings) != len(frame):
            raise ValueError("Embedding and frame lengths do not match")
        if "sentiment" not in frame.columns or "topic" not in frame.columns:
            raise ValueError("Missing sentiment/topic columns")

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.sentiment_labels = [
            sentiment2id[str(label)]
            for label in frame["sentiment"].tolist()
        ]
        self.topic_labels = [
            topic2id[str(label)]
            for label in frame["topic"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": torch.from_numpy(self.embeddings[index]),
            "sentiment_label": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
            "topic_label": torch.tensor(self.topic_labels[index], dtype=torch.long),
        }


def collate_embeddings(
    batch: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.stack([item["embedding"] for item in batch])
    sentiment_labels = torch.stack([item["sentiment_label"] for item in batch])
    topic_labels = torch.stack([item["topic_label"] for item in batch])
    return embeddings, sentiment_labels, topic_labels


class KeywordEmbeddingMultiTaskDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        frame: pd.DataFrame,
        sentiment2id: dict[str, int],
        topic2id: dict[str, int],
    ) -> None:
        if len(embeddings) != len(frame):
            raise ValueError("Embedding and frame lengths do not match")
        if "tokens" not in frame.columns:
            raise ValueError("Missing tokens column")
        if "sentiment" not in frame.columns or "topic" not in frame.columns:
            raise ValueError("Missing sentiment/topic columns")

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        token_lists = [str(tokens).split() for tokens in frame["tokens"].tolist()]
        self.keyword_features = np.stack(
            [extract_topic_keyword_features(tokens) for tokens in token_lists]
        )
        self.sentiment_labels = [
            sentiment2id[str(label)]
            for label in frame["sentiment"].tolist()
        ]
        self.topic_labels = [
            topic2id[str(label)]
            for label in frame["topic"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": torch.from_numpy(self.embeddings[index]),
            "keyword_features": torch.from_numpy(self.keyword_features[index]),
            "sentiment_label": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
            "topic_label": torch.tensor(self.topic_labels[index], dtype=torch.long),
        }


def collate_keyword_embeddings(
    batch: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.stack([item["embedding"] for item in batch])
    keyword_features = torch.stack([item["keyword_features"] for item in batch])
    sentiment_labels = torch.stack([item["sentiment_label"] for item in batch])
    topic_labels = torch.stack([item["topic_label"] for item in batch])
    return embeddings, keyword_features, sentiment_labels, topic_labels
