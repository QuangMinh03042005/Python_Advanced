from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.data.dataset import Vocab
from src.data.preprocess import tokenize_text
from src.features.keywords import keyword_hits
from src.models.textcnn import TextCNNMultiTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "textcnn_distilled" / "textcnn_distilled_best.pt"
)


class TextCNNMultiTaskPredictor:
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
        config = checkpoint["config"]
        self.config = config
        self.sentiment_labels = list(checkpoint["sentiment_labels"])
        self.topic_labels = list(checkpoint["topic_labels"])
        self.max_len = int(config["data"]["max_len"])
        self.sentiment_temperature = float(sentiment_temperature)
        self.topic_temperature = float(topic_temperature)
        self.vocab = Vocab(token2id=checkpoint["vocab"])
        self.model = TextCNNMultiTask(
            vocab_size=len(self.vocab),
            num_sentiment_classes=len(self.sentiment_labels),
            num_topic_classes=len(self.topic_labels),
            embed_dim=config["model"]["embed_dim"],
            num_filters=config["model"]["num_filters"],
            kernel_sizes=tuple(config["model"]["kernel_sizes"]),
            shared_hidden=config["model"]["shared_hidden"],
            dropout=config["model"]["dropout"],
            embedding_dropout=config["model"]["embedding_dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.parameter_count = self.model.count_parameters()
        self.model_name = "TextCNN distilled"

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        tokens = tokenize_text(text)
        if not tokens:
            raise ValueError("Câu đầu vào rỗng")

        input_ids = torch.tensor(
            [self.vocab.encode(tokens, self.max_len)],
            dtype=torch.long,
            device=self.device,
        )
        sentiment_logits, topic_logits = self.model(input_ids)
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
            for label, probability in zip(
                self.sentiment_labels,
                sentiment_probs,
                strict=True,
            )
        }
        topic_values = {
            label: float(probability)
            for label, probability in zip(
                self.topic_labels,
                topic_probs,
                strict=True,
            )
        }
        sentiment_sorted = sorted(
            sentiment_values.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        topic_sorted = sorted(
            topic_values.items(),
            key=lambda item: item[1],
            reverse=True,
        )
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
    parser = argparse.ArgumentParser(description="Predict with distilled TextCNN student.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sentiment-temperature", type=float, default=1.0)
    parser.add_argument("--topic-temperature", type=float, default=1.0)
    args = parser.parse_args()

    predictor = TextCNNMultiTaskPredictor(
        args.checkpoint,
        device=args.device,
        sentiment_temperature=args.sentiment_temperature,
        topic_temperature=args.topic_temperature,
    )
    result = predictor.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
