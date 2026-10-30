from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhoBERTMultiTaskHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        shared_hidden: int = 256,
        num_sentiment_classes: int = 3,
        num_topic_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.shared = nn.Linear(input_dim, shared_hidden)
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(shared_hidden, num_sentiment_classes)
        self.topic_head = nn.Linear(shared_hidden, num_topic_classes)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_norm(embeddings)
        hidden = self.dropout(F.relu(self.shared(hidden)))
        return self.sentiment_head(hidden), self.topic_head(hidden)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class KeywordAwarePhoBERTMultiTask(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        num_keyword_features: int = 12,
        shared_hidden: int = 256,
        num_sentiment_classes: int = 3,
        num_topic_classes: int = 4,
        dropout: float = 0.3,
        keyword_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.shared = nn.Linear(input_dim, shared_hidden)
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(shared_hidden, num_sentiment_classes)
        self.topic_head = nn.Linear(
            shared_hidden + num_keyword_features,
            num_topic_classes,
        )
        self.keyword_head = nn.Linear(num_keyword_features, num_topic_classes)
        self.keyword_scale = keyword_scale

    def forward(
        self,
        embeddings: torch.Tensor,
        keyword_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.input_norm(embeddings)
        hidden = self.dropout(F.relu(self.shared(hidden)))
        sentiment_logits = self.sentiment_head(hidden)
        topic_input = torch.cat([hidden, keyword_features], dim=1)
        keyword_logits = self.keyword_head(keyword_features)
        topic_logits = (
            self.topic_head(self.dropout(topic_input))
            + self.keyword_scale * keyword_logits
        )
        return sentiment_logits, topic_logits, keyword_logits

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
