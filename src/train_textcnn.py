from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data.dataset import TextCNNDataset, Vocab, collate_batch
from src.data.load import DEFAULT_DATA_DIR, DEFAULT_LABEL_MAP_PATH, load_label_map
from src.data.preprocess import build_tokenized_splits
from src.metrics import (
    classification_summary,
    per_class_report,
    plot_confusion_matrix,
)
from src.models.textcnn import TextCNN
from src.report_paths import TEXTCNN_SINGLE_DIR, resolve_results_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "textcnn.yaml"
DEFAULT_REPORTS_DIR = TEXTCNN_SINGLE_DIR
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "textcnn"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    table = frame.reset_index() if index else frame.copy()
    headers = [str(column).replace("|", "\\|") for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in table.iterrows():
        values = [str(value).replace("|", "\\|") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_loaders(
    splits: dict[str, pd.DataFrame],
    vocab: Vocab,
    label_column: str,
    label2id: dict[str, int],
    max_len: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    smoke_test: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    frames = {split: frame.copy() for split, frame in splits.items()}
    if smoke_test:
        frames = {
            split: frame.head(100).copy()
            for split, frame in frames.items()
        }

    datasets = {
        split: TextCNNDataset(
            frame=frame,
            vocab=vocab,
            label_column=label_column,
            label2id=label2id,
            max_len=max_len,
        )
        for split, frame in frames.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_batch,
            generator=generator,
        ),
        "dev": DataLoader(
            datasets["dev"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        ),
    }
    return loaders, frames


def class_weights_for(labels: list[int], num_classes: int, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(labels),
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: TextCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_examples += len(labels)

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: TextCNN,
    loader: DataLoader,
    device: torch.device,
    labels: list[str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_confidence: list[float] = []

    for input_ids, target in loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

        all_true.extend(target.tolist())
        all_pred.extend(prediction.cpu().tolist())
        all_confidence.extend(confidence.cpu().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    true_labels = [labels[index] for index in y_true]
    pred_labels = [labels[index] for index in y_pred]

    summary = classification_summary(true_labels, pred_labels)
    per_class = per_class_report(true_labels, pred_labels, labels)

    return summary, per_class, y_true, y_pred, np.array(all_confidence)


def plot_learning_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["epoch"], history["train_loss"], marker="o", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["dev_macro_f1"], marker="s", color="tab:orange")
    ax2.set_ylabel("Dev macro-F1", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _prediction_frame(
    frame: pd.DataFrame,
    label_column: str,
    labels: list[str],
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    result = frame[["text", label_column]].copy().reset_index(drop=True)
    result["prediction"] = [labels[index] for index in y_pred]
    result["confidence"] = confidence
    return result


def run_task(
    *,
    task: str,
    label_column: str,
    labels: list[str],
    label2id: dict[str, int],
    splits: dict[str, pd.DataFrame],
    config: dict[str, Any],
    run_id: str,
    reports_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    runs_dir: Path,
    device: torch.device,
    smoke_test: bool,
) -> dict[str, Any]:
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    vocab = Vocab.build(
        (str(tokens).split() for tokens in splits["train"]["tokens"]),
        min_freq=data_config["min_freq"],
        max_size=data_config["max_vocab_size"],
    )
    loaders, frames = build_loaders(
        splits=splits,
        vocab=vocab,
        label_column=label_column,
        label2id=label2id,
        max_len=data_config["max_len"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        seed=config["seed"],
        smoke_test=smoke_test,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=len(labels),
        embed_dim=model_config["embed_dim"],
        num_filters=model_config["num_filters"],
        kernel_sizes=tuple(model_config["kernel_sizes"]),
        dropout=model_config["dropout"],
        embedding_dropout=model_config["embedding_dropout"],
    ).to(device)

    train_labels = [
        label2id[str(label)]
        for label in frames["train"][label_column].tolist()
    ]
    weights = class_weights_for(
        train_labels,
        num_classes=len(labels),
        enabled=training_config["class_weight"],
    )
    if weights is not None:
        weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{task}_best.pt"

    epochs = 2 if smoke_test else training_config["epochs"]
    best_dev_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = training_config["patience"]
    history: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            training_config["max_grad_norm"],
        )
        dev_summary, _, _, _, _ = evaluate(model, loaders["dev"], device, labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_accuracy": dev_summary["accuracy"],
                "dev_macro_f1": dev_summary["macro_f1"],
                "dev_weighted_f1": dev_summary["weighted_f1"],
            }
        )
        print(
            f"[{task}] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_macro_f1={dev_summary['macro_f1']:.4f}"
        )

        if dev_summary["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_summary["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "config": config,
                    "vocab": vocab.token2id,
                    "label2id": label2id,
                    "labels": labels,
                    "max_len": data_config["max_len"],
                    "task": task,
                },
                checkpoint_path,
            )
        elif epoch - best_epoch >= patience:
            print(f"[{task}] early stopping at epoch {epoch}")
            break

    train_time_sec = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("No checkpoint was saved")

    model.load_state_dict(best_state)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        reports_dir / f"textcnn_{task}_history.csv",
        index=False,
        encoding="utf-8",
    )
    plot_learning_curve(
        history_frame,
        figures_dir / f"textcnn_{task}_learning_curve.png",
        f"TextCNN {task} — learning curve",
    )

    result: dict[str, Any] = {
        "model": model,
        "vocab": vocab,
        "checkpoint_path": checkpoint_path,
        "train_time_sec": train_time_sec,
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "history": history_frame,
        "metrics": {},
        "per_class": {},
        "predictions": {},
        "inference_ms": {},
    }

    for split in ("dev", "test"):
        infer_start = time.perf_counter()
        summary, per_class, y_true, y_pred, confidence = evaluate(
            model,
            loaders[split],
            device,
            labels,
        )
        inference_sec = time.perf_counter() - infer_start
        result["metrics"][split] = summary
        result["per_class"][split] = per_class
        result["inference_ms"][split] = inference_sec * 1000 / max(len(frames[split]), 1)

        per_class.to_csv(
            reports_dir / f"textcnn_{task}_{split}_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        plot_confusion_matrix(
            [labels[index] for index in y_true],
            [labels[index] for index in y_pred],
            labels,
            figures_dir / f"textcnn_{task}_{split}_confusion.png",
            f"TextCNN {task} — {split}",
        )

        prediction_frame = _prediction_frame(
            frames[split],
            label_column,
            labels,
            y_pred,
            confidence,
        )
        prediction_frame.to_csv(
            run_dir / f"{task}_{split}_predictions.csv",
            index=False,
            encoding="utf-8",
        )
        result["predictions"][split] = prediction_frame

    return result


def build_results_rows(
    *,
    run_id: str,
    config: dict[str, Any],
    git_hash: str,
    task_results: dict[str, dict[str, Any]],
    joint_accuracy: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, result in task_results.items():
        for split in ("dev", "test"):
            summary = result["metrics"][split]
            rows.append(
                {
                    "run_id": run_id,
                    "model": "textcnn_single",
                    "task": task,
                    "split": split,
                    "lambda": "",
                    "seed": config["seed"],
                    "git_commit": git_hash,
                    "sent_f1": summary["macro_f1"] if task == "sentiment" else "",
                    "topic_f1": summary["macro_f1"] if task == "topic" else "",
                    "joint_accuracy": "",
                    "accuracy": summary["accuracy"],
                    "macro_f1": summary["macro_f1"],
                    "weighted_f1": summary["weighted_f1"],
                    "train_time_sec": result["train_time_sec"],
                    "inference_ms": result["inference_ms"][split],
                }
            )
    for split, value in joint_accuracy.items():
        rows.append(
            {
                "run_id": run_id,
                "model": "textcnn_single",
                "task": "joint",
                "split": split,
                "lambda": "",
                "seed": config["seed"],
                "git_commit": git_hash,
                "sent_f1": "",
                "topic_f1": "",
                "joint_accuracy": value,
                "accuracy": value,
                "macro_f1": "",
                "weighted_f1": "",
                "train_time_sec": "",
                "inference_ms": "",
            }
        )
    return rows


def _joint_accuracy(
    sentiment_frame: pd.DataFrame,
    topic_frame: pd.DataFrame,
    source: pd.DataFrame,
) -> float:
    sentiment_correct = sentiment_frame["prediction"].to_numpy() == source["sentiment"].to_numpy()
    topic_correct = topic_frame["prediction"].to_numpy() == source["topic"].to_numpy()
    return float((sentiment_correct & topic_correct).mean())


def write_report(
    *,
    run_id: str,
    config: dict[str, Any],
    task_results: dict[str, dict[str, Any]],
    joint_accuracy: dict[str, float],
    reports_dir: Path,
) -> None:
    rows = []
    for task, result in task_results.items():
        for split in ("dev", "test"):
            summary = result["metrics"][split]
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "accuracy": round(summary["accuracy"], 4),
                    "macro_f1": round(summary["macro_f1"], 4),
                    "weighted_f1": round(summary["weighted_f1"], 4),
                    "best_epoch": result["best_epoch"],
                    "train_time_sec": round(result["train_time_sec"], 1),
                }
            )

    lines = [
        "# TextCNN single-task — UIT-VSFC",
        "",
        f"- Run ID: `{run_id}`",
        f"- Seed: `{config['seed']}`",
        f"- Max length: `{config['data']['max_len']}`",
        f"- Embedding dim: `{config['model']['embed_dim']}`",
        f"- Filters per kernel: `{config['model']['num_filters']}`",
        f"- Kernel sizes: `{config['model']['kernel_sizes']}`",
        f"- Dropout: `{config['model']['dropout']}`",
        f"- Batch size: `{config['training']['batch_size']}`",
        f"- Learning rate: `{config['training']['learning_rate']}`",
        "",
        "## Kết quả",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
    ]

    if joint_accuracy:
        lines.extend(
            [
                "## Joint accuracy",
                "",
                markdown_table(
                    pd.DataFrame(
                        [
                            {"split": split, "joint_accuracy": round(value, 4)}
                            for split, value in joint_accuracy.items()
                        ]
                    )
                ),
                "",
            ]
        )

    for task, result in task_results.items():
        for split in ("dev", "test"):
            lines.extend(
                [
                    f"## {task} — {split}",
                    "",
                    markdown_table(result["per_class"][split].round(4)),
                    "",
                    f"![{task} {split}](figures/textcnn_{task}_{split}_confusion.png)",
                    "",
                ]
            )

    lines.extend(
        [
            "## Learning curve",
            "",
        ]
    )
    for task in task_results:
        lines.extend(
            [
                f"![{task} learning curve](figures/textcnn_{task}_learning_curve.png)",
                "",
            ]
        )

    lines.extend(
        [
            "## Nhận xét",
            "",
            "- TextCNN dùng token tiếng Việt đã tách từ và vocabulary xây trên train.",
            "- So sánh trực tiếp với baseline TF-IDF trong `reports/results.csv`.",
            "- Chưa có multi-task; đây là mốc cho TextCNN multi-task.",
            "",
        ]
    )

    (reports_dir / "textcnn_single_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    task: str,
    data_dir: str | Path,
    config_path: str | Path,
    reports_dir: str | Path,
    models_dir: str | Path,
    runs_dir: str | Path,
    smoke_test: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    set_seed(config["seed"])
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(DEFAULT_LABEL_MAP_PATH)

    data_path = resolve_path(config["data"]["tokenized_dir"])
    splits = build_tokenized_splits(
        data_dir=data_dir,
        output_dir=data_path,
    )

    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    models_dir = Path(models_dir)
    runs_dir = Path(runs_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("textcnn_single_%Y%m%d_%H%M%S")
    git_hash = git_commit()

    sentiment_labels = [
        label_map.sentiment_id2label[index]
        for index in sorted(label_map.sentiment_id2label)
    ]
    topic_labels = [
        label_map.topic_id2label[index]
        for index in sorted(label_map.topic_id2label)
    ]

    task_results: dict[str, dict[str, Any]] = {}
    tasks = ["sentiment", "topic"] if task == "both" else [task]

    for task_name in tasks:
        if task_name == "sentiment":
            label_column = "sentiment"
            labels = sentiment_labels
            label2id = label_map.sentiment_label2id
        else:
            label_column = "topic"
            labels = topic_labels
            label2id = label_map.topic_label2id

        print(f"=== {task_name} ===")
        task_results[task_name] = run_task(
            task=task_name,
            label_column=label_column,
            labels=labels,
            label2id=label2id,
            splits=splits,
            config=config,
            run_id=run_id,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            models_dir=models_dir,
            runs_dir=runs_dir,
            device=device,
            smoke_test=smoke_test,
        )

    joint_accuracy: dict[str, float] = {}
    if "sentiment" in task_results and "topic" in task_results:
        for split in ("dev", "test"):
            source = splits[split].head(100) if smoke_test else splits[split]
            joint_accuracy[split] = _joint_accuracy(
                task_results["sentiment"]["predictions"][split],
                task_results["topic"]["predictions"][split],
                source.reset_index(drop=True),
            )

    results_path = resolve_results_path(reports_dir)
    new_rows = build_results_rows(
        run_id=run_id,
        config=config,
        git_hash=git_hash,
        task_results=task_results,
        joint_accuracy=joint_accuracy,
    )
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "textcnn_single"]
        results = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        results = pd.DataFrame(new_rows)
    results.to_csv(results_path, index=False, encoding="utf-8")

    metrics_payload = {
        "run_id": run_id,
        "git_commit": git_hash,
        "config": config,
        "device": str(device),
        "sentiment": task_results["sentiment"]["metrics"] if "sentiment" in task_results else {},
        "topic": task_results["topic"]["metrics"] if "topic" in task_results else {},
        "joint_accuracy": joint_accuracy,
        "best_epoch": {
            task_name: result["best_epoch"]
            for task_name, result in task_results.items()
        },
        "checkpoints": {
            task_name: display_path(result["checkpoint_path"])
            for task_name, result in task_results.items()
        },
    }
    (reports_dir / "textcnn_single_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        run_id=run_id,
        config=config,
        task_results=task_results,
        joint_accuracy=joint_accuracy,
        reports_dir=reports_dir,
    )

    return {
        "run_id": run_id,
        "task_results": task_results,
        "joint_accuracy": joint_accuracy,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-task TextCNN on UIT-VSFC.")
    parser.add_argument("--task", choices=["sentiment", "topic", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    result = run(
        task=args.task,
        data_dir=args.data_dir,
        config_path=args.config,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        smoke_test=args.smoke_test,
    )
    print(f"run_id: {result['run_id']}")
    print(f"device: {result['device']}")
    for task, task_result in result["task_results"].items():
        summary = task_result["metrics"]["test"]
        print(f"{task} test macro-F1: {summary['macro_f1']:.4f}")
    if result["joint_accuracy"]:
        print(f"joint test accuracy: {result['joint_accuracy']['test']:.4f}")


if __name__ == "__main__":
    main()
