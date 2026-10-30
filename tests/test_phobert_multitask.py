from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.dataset import EmbeddingMultiTaskDataset, collate_embeddings
from src.models.phobert_multitask import PhoBERTMultiTaskHead


def test_embedding_dataset_and_collate() -> None:
    frame = pd.DataFrame(
        {
            "sentiment": ["positive", "negative"],
            "topic": ["lecturer", "training_program"],
        }
    )
    embeddings = np.arange(16, dtype=np.float32).reshape(2, 8)
    dataset = EmbeddingMultiTaskDataset(
        embeddings=embeddings,
        frame=frame,
        sentiment2id={"negative": 0, "positive": 1},
        topic2id={"lecturer": 0, "training_program": 1},
    )

    item = dataset[0]
    assert item["embedding"].shape == (8,)
    assert item["sentiment_label"].item() == 1
    assert item["topic_label"].item() == 0

    batch_embeddings, sentiment_labels, topic_labels = collate_embeddings(
        [dataset[0], dataset[1]]
    )
    assert batch_embeddings.shape == (2, 8)
    assert sentiment_labels.tolist() == [1, 0]
    assert topic_labels.tolist() == [0, 1]


def test_phobert_head_forward_shape() -> None:
    model = PhoBERTMultiTaskHead(
        input_dim=16,
        shared_hidden=8,
        num_sentiment_classes=3,
        num_topic_classes=4,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 16)

    sentiment_logits, topic_logits = model(embeddings)

    assert sentiment_logits.shape == (2, 3)
    assert topic_logits.shape == (2, 4)
    assert model.count_parameters() > 0
