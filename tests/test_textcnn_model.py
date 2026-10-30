from __future__ import annotations

import torch

from src.models.textcnn import TextCNN


def test_textcnn_forward_shape() -> None:
    model = TextCNN(
        vocab_size=100,
        num_classes=3,
        embed_dim=16,
        num_filters=8,
        kernel_sizes=(2, 3),
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

    logits = model(input_ids)

    assert logits.shape == (2, 3)
    assert model.count_parameters() > 0
