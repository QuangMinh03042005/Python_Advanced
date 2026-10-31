from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.data.preprocess import tokenize_text  # noqa: E402
from src.features.keywords import (  # noqa: E402
    extract_topic_keyword_features,
    keyword_hits,
)
from src.models.phobert_multitask import (  # noqa: E402
    KeywordAwarePhoBERTMultiTask,
    PhoBERTMultiTaskHead,
)

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "phobert_keyword_multitask"
    / "phobert_keyword_multitask_best.pt"
)


class PhoBERTMultiTaskPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str | torch.device | None = None,
        sentiment_temperature: float = 1.0,
        topic_temperature: float = 1.0,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.config = checkpoint["config"]
        self.sentiment_labels = list(checkpoint["sentiment_labels"])
        self.topic_labels = list(checkpoint["topic_labels"])
        self.max_len = int(checkpoint["max_len"])
        self.embedding_model = str(checkpoint["embedding_model"])
        self.model_type = str(checkpoint.get("model_type", "standard"))
        self.keyword_aware = self.model_type == "keyword_aware"
        self.sentiment_temperature = float(sentiment_temperature)
        self.topic_temperature = float(topic_temperature)
        self.parameter_count = 0
        self.model_name = (
            "PhoBERT keyword-aware teacher"
            if self.keyword_aware
            else "PhoBERT teacher"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.encoder = AutoModel.from_pretrained(self.embedding_model).to(self.device)
        self.encoder.eval()

        if self.keyword_aware:
            self.head = KeywordAwarePhoBERTMultiTask(
                input_dim=int(self.encoder.config.hidden_size),
                num_keyword_features=int(checkpoint["num_keyword_features"]),
                shared_hidden=int(self.config["model"]["shared_hidden"]),
                num_sentiment_classes=len(self.sentiment_labels),
                num_topic_classes=len(self.topic_labels),
                dropout=float(self.config["model"]["dropout"]),
            ).to(self.device)
        else:
            self.head = PhoBERTMultiTaskHead(
                input_dim=int(self.encoder.config.hidden_size),
                shared_hidden=int(self.config["model"]["shared_hidden"]),
                num_sentiment_classes=len(self.sentiment_labels),
                num_topic_classes=len(self.topic_labels),
                dropout=float(self.config["model"]["dropout"]),
            ).to(self.device)
        self.head.load_state_dict(checkpoint["model_state"])
        self.head.eval()
        self.parameter_count = sum(
            parameter.numel() for parameter in self.encoder.parameters()
        ) + sum(parameter.numel() for parameter in self.head.parameters())

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        tokens = tokenize_text(text)
        if not tokens:
            raise ValueError("Câu đầu vào rỗng")

        encoded = self.tokenizer(
            " ".join(tokens),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        encoder_output = self.encoder(**encoded)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]

        if self.keyword_aware:
            keyword_features = torch.tensor(
                extract_topic_keyword_features(tokens),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            sentiment_logits, topic_logits, _ = self.head(cls_embedding, keyword_features)
        else:
            sentiment_logits, topic_logits = self.head(cls_embedding)

        sentiment_probs = torch.softmax(
            sentiment_logits / self.sentiment_temperature,
            dim=1,
        )[0]
        topic_probs = torch.softmax(
            topic_logits / self.topic_temperature,
            dim=1,
        )[0]

        sentiment_values = {
            label: float(probability)
            for label, probability in zip(self.sentiment_labels, sentiment_probs, strict=True)
        }
        topic_values = {
            label: float(probability)
            for label, probability in zip(self.topic_labels, topic_probs, strict=True)
        }

        sentiment_sorted = sorted(sentiment_values.items(), key=lambda item: item[1], reverse=True)
        topic_sorted = sorted(topic_values.items(), key=lambda item: item[1], reverse=True)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "text": text,
            "tokens": tokens,
            "topic_keywords": keyword_hits(tokens),
            "model_name": self.model_name,
            "parameter_count": self.parameter_count,
            "latency_ms": latency_ms,
            "sentiment": {
                "label": sentiment_sorted[0][0],
                "confidence": sentiment_sorted[0][1],
                "margin": sentiment_sorted[0][1] - sentiment_sorted[1][1],
                "probabilities": sentiment_values,
                "temperature": self.sentiment_temperature,
            },
            "topic": {
                "label": topic_sorted[0][0],
                "confidence": topic_sorted[0][1],
                "margin": topic_sorted[0][1] - topic_sorted[1][1],
                "probabilities": topic_values,
                "temperature": self.topic_temperature,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment and topic for one sentence.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sentiment-temperature", type=float, default=1.0)
    parser.add_argument("--topic-temperature", type=float, default=1.0)
    args = parser.parse_args()

    predictor = PhoBERTMultiTaskPredictor(
        args.checkpoint,
        device=args.device,
        sentiment_temperature=args.sentiment_temperature,
        topic_temperature=args.topic_temperature,
    )
    result = predictor.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
