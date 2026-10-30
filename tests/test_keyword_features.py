from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    KeywordEmbeddingMultiTaskDataset,
    collate_keyword_embeddings,
)
from src.features.keywords import (
    extract_topic_keyword_features,
    keyword_feature_names,
    keyword_hits,
)
from src.models.phobert_multitask import KeywordAwarePhoBERTMultiTask


def test_keyword_features_for_lecturer() -> None:
    tokens = ["cần", "cải_thiện", "thêm", ",", "nhất_là", "giảng_viên"]
    features = extract_topic_keyword_features(tokens)
    names = keyword_feature_names()
    values = dict(zip(names, features.tolist(), strict=True))

    assert values["lecturer_present"] == 1.0
    assert values["lecturer_density"] > 0
    assert values["lecturer_last3"] == 1.0
    assert values["facility_present"] == 0.0
    assert len(features) == 12

    hits = keyword_hits(tokens)
    assert hits["lecturer"] == ["giảng_viên"]


def test_keyword_dataset_and_collate() -> None:
    frame = pd.DataFrame(
        {
            "tokens": ["giảng_viên dạy tốt", "phòng_học thiếu máy_chiếu"],
            "sentiment": ["positive", "negative"],
            "topic": ["lecturer", "facility"],
        }
    )
    embeddings = np.arange(16, dtype=np.float32).reshape(2, 8)
    dataset = KeywordEmbeddingMultiTaskDataset(
        embeddings=embeddings,
        frame=frame,
        sentiment2id={"negative": 0, "positive": 1},
        topic2id={"lecturer": 0, "facility": 1},
    )

    batch = collate_keyword_embeddings([dataset[0], dataset[1]])
    embeddings_batch, keyword_batch, sentiment_labels, topic_labels = batch

    assert embeddings_batch.shape == (2, 8)
    assert keyword_batch.shape == (2, 12)
    assert sentiment_labels.tolist() == [1, 0]
    assert topic_labels.tolist() == [0, 1]


def test_keyword_aware_model_forward_shape() -> None:
    model = KeywordAwarePhoBERTMultiTask(
        input_dim=16,
        num_keyword_features=12,
        shared_hidden=8,
        num_sentiment_classes=3,
        num_topic_classes=4,
        dropout=0.0,
    )

    sentiment_logits, topic_logits, keyword_logits = model(
        torch.randn(2, 16),
        torch.randn(2, 12),
    )

    assert sentiment_logits.shape == (2, 3)
    assert topic_logits.shape == (2, 4)
    assert keyword_logits.shape == (2, 4)
