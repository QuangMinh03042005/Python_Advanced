from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.data.preprocess import tokenize_text  # noqa: E402
from src.models.phobert_multitask import PhoBERTMultiTaskHead  # noqa: E402

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "phobert_multitask" / "phobert_multitask_best.pt"
)


class PhoBERTMultiTaskPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str | torch.device | None = None,
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.encoder = AutoModel.from_pretrained(self.embedding_model).to(self.device)
        self.encoder.eval()

        self.head = PhoBERTMultiTaskHead(
            input_dim=int(self.encoder.config.hidden_size),
            shared_hidden=int(self.config["model"]["shared_hidden"]),
            num_sentiment_classes=len(self.sentiment_labels),
            num_topic_classes=len(self.topic_labels),
            dropout=float(self.config["model"]["dropout"]),
        ).to(self.device)
        self.head.load_state_dict(checkpoint["model_state"])
        self.head.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
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

        sentiment_logits, topic_logits = self.head(cls_embedding)
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]
        topic_probs = torch.softmax(topic_logits, dim=1)[0]

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

        return {
            "text": text,
            "tokens": tokens,
            "sentiment": {
                "label": sentiment_sorted[0][0],
                "confidence": sentiment_sorted[0][1],
                "margin": sentiment_sorted[0][1] - sentiment_sorted[1][1],
                "probabilities": sentiment_values,
            },
            "topic": {
                "label": topic_sorted[0][0],
                "confidence": topic_sorted[0][1],
                "margin": topic_sorted[0][1] - topic_sorted[1][1],
                "probabilities": topic_values,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment and topic for one sentence.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    predictor = PhoBERTMultiTaskPredictor(args.checkpoint, device=args.device)
    result = predictor.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
