from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    DistillTextCNNDataset,
    Vocab,
    collate_distill,
)


def test_distill_dataset_and_collate() -> None:
    frame = pd.DataFrame(
        {
            "tokens": ["giảng_viên dạy tốt", "bài_tập quá khó"],
            "sentiment": ["positive", "negative"],
            "topic": ["lecturer", "training_program"],
        }
    )
    vocab = Vocab.build([tokens.split() for tokens in frame["tokens"]], min_freq=1)
    sentiment_probs = np.array(
        [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
        dtype=np.float32,
    )
    topic_probs = np.array(
        [[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1]],
        dtype=np.float32,
    )
    dataset = DistillTextCNNDataset(
        frame=frame,
        vocab=vocab,
        sentiment2id={"negative": 0, "neutral": 1, "positive": 2},
        topic2id={"lecturer": 0, "training_program": 1, "facility": 2, "others": 3},
        sentiment_probs=sentiment_probs,
        topic_probs=topic_probs,
        max_len=4,
    )

    item = dataset[0]
    assert item["input_ids"].shape == (4,)
    assert item["sentiment_label"].item() == 2
    assert item["topic_label"].item() == 0
    assert torch.allclose(item["sentiment_probs"], torch.tensor([0.1, 0.2, 0.7]))

    batch = collate_distill([dataset[0], dataset[1]])
    input_ids, sentiment_labels, topic_labels, teacher_sentiment, teacher_topic = batch

    assert input_ids.shape == (2, 4)
    assert sentiment_labels.tolist() == [2, 0]
    assert topic_labels.tolist() == [0, 1]
    assert teacher_sentiment.shape == (2, 3)
    assert teacher_topic.shape == (2, 4)
