from __future__ import annotations

import pandas as pd
import torch

from src.data.dataset import (
    TextCNNMultiTaskDataset,
    Vocab,
    collate_multitask,
)
from src.models.textcnn import TextCNNMultiTask


def test_multitask_dataset_and_collate() -> None:
    frame = pd.DataFrame(
        {
            "tokens": ["giảng_viên dạy tốt", "bài_tập quá khó"],
            "sentiment": ["positive", "negative"],
            "topic": ["lecturer", "training_program"],
        }
    )
    vocab = Vocab.build([tokens.split() for tokens in frame["tokens"]], min_freq=1)
    dataset = TextCNNMultiTaskDataset(
        frame=frame,
        vocab=vocab,
        sentiment2id={"negative": 0, "positive": 1},
        topic2id={"lecturer": 0, "training_program": 1},
        max_len=4,
    )

    item = dataset[0]
    assert item["input_ids"].shape == (4,)
    assert item["sentiment_label"].item() == 1
    assert item["topic_label"].item() == 0

    input_ids, sentiment_labels, topic_labels = collate_multitask(
        [dataset[0], dataset[1]]
    )
    assert input_ids.shape == (2, 4)
    assert sentiment_labels.tolist() == [1, 0]
    assert topic_labels.tolist() == [0, 1]


def test_multitask_textcnn_forward_shape() -> None:
    model = TextCNNMultiTask(
        vocab_size=100,
        num_sentiment_classes=3,
        num_topic_classes=4,
        embed_dim=16,
        num_filters=8,
        kernel_sizes=(2, 3),
        shared_hidden=32,
        dropout=0.0,
        embedding_dropout=0.0,
    )
    input_ids = torch.tensor(
        [
            [2, 3, 4, 5, 0, 0],
            [6, 7, 8, 9, 10, 0],
        ],
        dtype=torch.long,
    )

    sentiment_logits, topic_logits = model(input_ids)

    assert sentiment_logits.shape == (2, 3)
    assert topic_logits.shape == (2, 4)
    assert model.count_parameters() > 0
