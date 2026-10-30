from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "uit_vsfc"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "configs" / "label_map.json"

SPLITS = ("train", "dev", "test")
Split = Literal["train", "dev", "test"]


@dataclass(frozen=True)
class LabelMap:
    sentiment_id2label: dict[int, str]
    topic_id2label: dict[int, str]
    sentiment_label2id: dict[str, int]
    topic_label2id: dict[str, int]

    @property
    def num_sentiment_labels(self) -> int:
        return len(self.sentiment_id2label)

    @property
    def num_topic_labels(self) -> int:
        return len(self.topic_id2label)


def load_label_map(path: str | Path = DEFAULT_LABEL_MAP_PATH) -> LabelMap:
    """Load and validate the UIT-VSFC label mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    sentiment = raw["sentiment"]
    topic = raw["topic"]

    sentiment_id2label = {int(k): str(v) for k, v in sentiment["id2label"].items()}
    topic_id2label = {int(k): str(v) for k, v in topic["id2label"].items()}
    sentiment_label2id = {str(k): int(v) for k, v in sentiment["label2id"].items()}
    topic_label2id = {str(k): int(v) for k, v in topic["label2id"].items()}

    if set(sentiment_id2label) != set(sentiment_label2id.values()):
        raise ValueError("Sentiment id2label/label2id are inconsistent")
    if set(topic_id2label) != set(topic_label2id.values()):
        raise ValueError("Topic id2label/label2id are inconsistent")

    return LabelMap(
        sentiment_id2label=sentiment_id2label,
        topic_id2label=topic_id2label,
        sentiment_label2id=sentiment_label2id,
        topic_label2id=topic_label2id,
    )


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"Data file is empty: {path}")
    return lines


def read_split(
    split: Split,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    label_map: LabelMap | None = None,
) -> pd.DataFrame:
    """Read one UIT-VSFC split and return text + integer/string labels."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}. Expected one of {SPLITS}")

    data_dir = Path(data_dir)
    label_map = label_map or load_label_map()
    split_dir = data_dir / split

    texts = _read_lines(split_dir / "sentences.txt")
    sentiments_raw = _read_lines(split_dir / "sentiments.txt")
    topics_raw = _read_lines(split_dir / "topics.txt")

    lengths = {len(texts), len(sentiments_raw), len(topics_raw)}
    if len(lengths) != 1:
        raise ValueError(
            f"Line count mismatch in {split_dir}: "
            f"sentences={len(texts)}, sentiments={len(sentiments_raw)}, "
            f"topics={len(topics_raw)}"
        )

    texts = [text.strip() for text in texts]
    empty_rows = [i for i, text in enumerate(texts) if not text]
    if empty_rows:
        preview = empty_rows[:5]
        raise ValueError(f"Empty sentences found in {split_dir} at rows {preview}")

    try:
        sentiment_ids = [int(value.strip()) for value in sentiments_raw]
        topic_ids = [int(value.strip()) for value in topics_raw]
    except ValueError as exc:
        raise ValueError(f"Non-integer label found in {split_dir}") from exc

    unknown_sentiments = sorted(set(sentiment_ids) - set(label_map.sentiment_id2label))
    unknown_topics = sorted(set(topic_ids) - set(label_map.topic_id2label))
    if unknown_sentiments:
        raise ValueError(f"Unknown sentiment labels in {split_dir}: {unknown_sentiments}")
    if unknown_topics:
        raise ValueError(f"Unknown topic labels in {split_dir}: {unknown_topics}")

    frame = pd.DataFrame(
        {
            "text": texts,
            "sentiment_id": sentiment_ids,
            "topic_id": topic_ids,
        }
    )
    frame["sentiment"] = frame["sentiment_id"].map(label_map.sentiment_id2label)
    frame["topic"] = frame["topic_id"].map(label_map.topic_id2label)
    frame.insert(0, "split", split)

    return frame


def load_splits(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    label_map_path: str | Path = DEFAULT_LABEL_MAP_PATH,
) -> dict[str, pd.DataFrame]:
    """Load train/dev/test splits with a shared label map."""
    label_map = load_label_map(label_map_path)
    return {
        split: read_split(split, data_dir=data_dir, label_map=label_map)
        for split in SPLITS
    }


def load_all(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    label_map_path: str | Path = DEFAULT_LABEL_MAP_PATH,
) -> pd.DataFrame:
    """Load and concatenate all splits, keeping the split column."""
    splits = load_splits(data_dir=data_dir, label_map_path=label_map_path)
    return pd.concat(splits.values(), ignore_index=True)


if __name__ == "__main__":
    splits = load_splits()
    for split, frame in splits.items():
        print(f"{split}: {len(frame)} rows")
        print(frame[["sentiment", "topic"]].value_counts().head())
    print(f"total: {sum(len(frame) for frame in splits.values())} rows")
