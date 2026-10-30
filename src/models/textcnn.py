from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 100,
        num_filters: int = 100,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        dropout: float = 0.5,
        embedding_dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded).transpose(1, 2)

        pooled = []
        for conv in self.convs:
            features = F.relu(conv(embedded))
            pooled.append(F.adaptive_max_pool1d(features, 1).squeeze(-1))

        representation = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(representation))

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
