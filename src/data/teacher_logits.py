from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.features.keywords import extract_topic_keyword_features  # noqa: E402
from src.models.phobert_multitask import (  # noqa: E402
    KeywordAwarePhoBERTMultiTask,
    PhoBERTMultiTaskHead,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "phobert_keyword_multitask"
    / "phobert_keyword_multitask_best.pt"
)
DEFAULT_TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_tokenized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "teacher_soft_labels"
SPLITS = ("train", "dev", "test")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_teacher(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(checkpoint["embedding_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    config = checkpoint["config"]
    model_type = str(checkpoint.get("model_type", "standard"))
    keyword_aware = model_type == "keyword_aware"
    if keyword_aware:
        head = KeywordAwarePhoBERTMultiTask(
            input_dim=int(encoder.config.hidden_size),
            num_keyword_features=int(checkpoint["num_keyword_features"]),
            shared_hidden=int(config["model"]["shared_hidden"]),
            num_sentiment_classes=len(checkpoint["sentiment_labels"]),
            num_topic_classes=len(checkpoint["topic_labels"]),
            dropout=float(config["model"]["dropout"]),
        ).to(device)
    else:
        head = PhoBERTMultiTaskHead(
            input_dim=int(encoder.config.hidden_size),
            shared_hidden=int(config["model"]["shared_hidden"]),
            num_sentiment_classes=len(checkpoint["sentiment_labels"]),
            num_topic_classes=len(checkpoint["topic_labels"]),
            dropout=float(config["model"]["dropout"]),
        ).to(device)
    head.load_state_dict(checkpoint["model_state"])
    head.eval()

    return {
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "head": head,
        "device": device,
        "model_type": model_type,
        "keyword_aware": keyword_aware,
        "model_name": model_name,
        "max_len": int(checkpoint["max_len"]),
    }


@torch.no_grad()
def extract_split(
    split: str,
    *,
    tokenized_dir: Path,
    teacher: dict[str, Any],
    batch_size: int,
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(tokenized_dir / f"{split}.csv")
    texts = frame["tokens"].astype(str).tolist()
    tokenizer = teacher["tokenizer"]
    encoder = teacher["encoder"]
    head = teacher["head"]
    device = teacher["device"]
    max_len = teacher["max_len"]
    keyword_aware = teacher["keyword_aware"]

    sentiment_logits: list[np.ndarray] = []
    topic_logits: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cls_embeddings = encoder(**encoded).last_hidden_state[:, 0, :]

        if keyword_aware:
            keyword_features = np.stack(
                [
                    extract_topic_keyword_features(text.split())
                    for text in batch_texts
                ]
            )
            keyword_tensor = torch.tensor(
                keyword_features,
                dtype=torch.float32,
                device=device,
            )
            sentiment_out, topic_out, _ = head(cls_embeddings, keyword_tensor)
        else:
            sentiment_out, topic_out = head(cls_embeddings)

        sentiment_logits.append(sentiment_out.cpu().numpy())
        topic_logits.append(topic_out.cpu().numpy())
        if start % (batch_size * 10) == 0:
            print(f"  {split}: {min(start + batch_size, len(texts))}/{len(texts)}")

    sentiment_logits_array = np.concatenate(sentiment_logits, axis=0).astype(np.float32)
    topic_logits_array = np.concatenate(topic_logits, axis=0).astype(np.float32)
    sentiment_probs = torch.softmax(
        torch.from_numpy(sentiment_logits_array),
        dim=1,
    ).numpy()
    topic_probs = torch.softmax(
        torch.from_numpy(topic_logits_array),
        dim=1,
    ).numpy()

    return {
        "sentiment_logits": sentiment_logits_array,
        "topic_logits": topic_logits_array,
        "sentiment_probs": sentiment_probs.astype(np.float32),
        "topic_probs": topic_probs.astype(np.float32),
        "frame": frame,
    }


def _summary_row(
    split: str,
    result: dict[str, np.ndarray],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    frame = result["frame"]
    sentiment2id = checkpoint["sentiment2id"]
    topic2id = checkpoint["topic2id"]
    y_sentiment = frame["sentiment"].map(sentiment2id).to_numpy()
    y_topic = frame["topic"].map(topic2id).to_numpy()
    pred_sentiment = result["sentiment_probs"].argmax(axis=1)
    pred_topic = result["topic_probs"].argmax(axis=1)

    return {
        "split": split,
        "rows": len(frame),
        "sentiment_accuracy": accuracy_score(y_sentiment, pred_sentiment),
        "sentiment_macro_f1": f1_score(y_sentiment, pred_sentiment, average="macro"),
        "topic_accuracy": accuracy_score(y_topic, pred_topic),
        "topic_macro_f1": f1_score(y_topic, pred_topic, average="macro"),
        "sentiment_shape": tuple(result["sentiment_probs"].shape),
        "topic_shape": tuple(result["topic_probs"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract teacher soft labels for distillation.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("checkpoint") == str(args.checkpoint):
            print(f"Using cached teacher labels in {output_dir}")
            return

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()
    teacher = load_teacher(args.checkpoint, device)
    print(f"Teacher: {teacher['model_type']} on {device}")

    summaries: list[dict[str, Any]] = []
    for split in SPLITS:
        print(f"Extracting {split}")
        result = extract_split(
            split,
            tokenized_dir=args.tokenized_dir,
            teacher=teacher,
            batch_size=args.batch_size,
        )
        np.save(output_dir / f"{split}_sentiment_logits.npy", result["sentiment_logits"])
        np.save(output_dir / f"{split}_topic_logits.npy", result["topic_logits"])
        np.save(output_dir / f"{split}_sentiment_probs.npy", result["sentiment_probs"])
        np.save(output_dir / f"{split}_topic_probs.npy", result["topic_probs"])
        summaries.append(
            _summary_row(split, result, teacher["checkpoint"])
        )

    elapsed = time.perf_counter() - start_time
    metadata = {
        "checkpoint": str(args.checkpoint),
        "model_type": teacher["model_type"],
        "model_name": teacher["model_name"],
        "max_len": teacher["max_len"],
        "batch_size": args.batch_size,
        "device": str(device),
        "git_commit": git_commit(),
        "extraction_time_sec": elapsed,
        "splits": {row["split"]: row["rows"] for row in summaries},
        "sentiment_labels": teacher["checkpoint"]["sentiment_labels"],
        "topic_labels": teacher["checkpoint"]["topic_labels"],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Teacher soft labels",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Model type: `{teacher['model_type']}`",
        f"- Device: `{device}`",
        f"- Max length: `{teacher['max_len']}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Extraction time: `{elapsed:.1f}s`",
        f"- Git commit: `{metadata['git_commit']}`",
        "",
        "| Split | Rows | Sent acc | Sent F1 | Topic acc | Topic F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['rows']} | "
            f"{row['sentiment_accuracy']:.4f} | {row['sentiment_macro_f1']:.4f} | "
            f"{row['topic_accuracy']:.4f} | {row['topic_macro_f1']:.4f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "```text",
        "train_sentiment_logits.npy",
        "train_topic_logits.npy",
        "train_sentiment_probs.npy",
        "train_topic_probs.npy",
        "dev_*",
        "test_*",
        "metadata.json",
        "```",
        "",
    ]
    (PROJECT_ROOT / "reports" / "teacher_soft_labels.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
