from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.dataset import Vocab
from src.data.preprocess import tokenize_text
from src.models.textcnn import TextCNNMultiTask
from src.predict import DEFAULT_CHECKPOINT_PATH, PhoBERTMultiTaskPredictor
from src.report_paths import DISTILLATION_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "textcnn_distilled" / "textcnn_distilled_best.pt"
)
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_tokenized" / "test.csv"
DEFAULT_OUTPUT = DISTILLATION_DIR / "distillation_benchmark.json"


def load_student(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vocab = Vocab(token2id=checkpoint["vocab"])
    model = TextCNNMultiTask(
        vocab_size=len(vocab),
        num_sentiment_classes=len(checkpoint["sentiment_labels"]),
        num_topic_classes=len(checkpoint["topic_labels"]),
        embed_dim=config["model"]["embed_dim"],
        num_filters=config["model"]["num_filters"],
        kernel_sizes=tuple(config["model"]["kernel_sizes"]),
        shared_hidden=config["model"]["shared_hidden"],
        dropout=config["model"]["dropout"],
        embedding_dropout=config["model"]["embedding_dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "vocab": vocab,
        "max_len": int(config["data"]["max_len"]),
        "parameter_count": model.count_parameters(),
    }


def benchmark_teacher(texts: list[str], checkpoint: Path, repeats: int) -> dict[str, Any]:
    predictor = PhoBERTMultiTaskPredictor(checkpoint)
    for text in texts[:5]:
        predictor.predict(text)

    start = time.perf_counter()
    for text in texts:
        predictor.predict(text)
    elapsed = time.perf_counter() - start
    parameters = sum(p.numel() for p in predictor.encoder.parameters()) + sum(
        p.numel() for p in predictor.head.parameters()
    )
    return {
        "model": "PhoBERT keyword-aware teacher",
        "parameter_count": int(parameters),
        "estimated_size_mb": parameters * 4 / (1024**2),
        "saved_checkpoint_mb": checkpoint.stat().st_size / (1024**2),
        "elapsed_sec": elapsed,
        "ms_per_sentence": elapsed * 1000 / repeats,
        "sentences_per_sec": repeats / elapsed,
    }


@torch.no_grad()
def benchmark_student(
    texts: list[str],
    checkpoint: Path,
    repeats: int,
) -> dict[str, Any]:
    student = load_student(checkpoint)
    model = student["model"]
    vocab = student["vocab"]
    max_len = student["max_len"]

    for text in texts[:5]:
        tokens = tokenize_text(text)
        ids = torch.tensor([vocab.encode(tokens, max_len)], dtype=torch.long)
        model(ids)

    start = time.perf_counter()
    for text in texts:
        tokens = tokenize_text(text)
        ids = torch.tensor([vocab.encode(tokens, max_len)], dtype=torch.long)
        model(ids)
    elapsed = time.perf_counter() - start
    parameters = student["parameter_count"]
    return {
        "model": "TextCNN distilled student",
        "parameter_count": int(parameters),
        "estimated_size_mb": parameters * 4 / (1024**2),
        "saved_checkpoint_mb": checkpoint.stat().st_size / (1024**2),
        "elapsed_sec": elapsed,
        "ms_per_sentence": elapsed * 1000 / repeats,
        "sentences_per_sec": repeats / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark teacher vs distilled student.")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--n-sentences", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    frame = pd.read_csv(args.test_csv)
    texts = frame["text"].astype(str).head(args.n_sentences).tolist()

    teacher = benchmark_teacher(texts, args.teacher_checkpoint, len(texts))
    student = benchmark_student(texts, args.student_checkpoint, len(texts))
    speedup = teacher["ms_per_sentence"] / student["ms_per_sentence"]
    size_reduction = teacher["estimated_size_mb"] / student["estimated_size_mb"]

    payload = {
        "n_sentences": len(texts),
        "device": "cpu",
        "teacher": teacher,
        "student": student,
        "speedup": speedup,
        "size_reduction": size_reduction,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
