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

from src.data.dataset import (
    EmbeddingMultiTaskDataset,
    collate_embeddings,
)
from src.data.load import DEFAULT_LABEL_MAP_PATH, load_label_map
from src.data.phobert_embeddings import load_or_extract_embeddings
from src.data.preprocess import build_tokenized_splits
from src.metrics import (
    classification_summary,
    per_class_report,
    plot_confusion_matrix,
)
from src.models.phobert_multitask import PhoBERTMultiTaskHead

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "phobert_multitask.yaml"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "phobert_multitask"
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


def class_weights_for(
    labels: list[int],
    num_classes: int,
    enabled: bool | str,
) -> torch.Tensor | None:
    if enabled is False or enabled == "none":
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(labels),
    )
    if enabled == "sqrt_balanced":
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(
    *,
    embeddings: dict[str, np.ndarray],
    splits: dict[str, pd.DataFrame],
    sentiment2id: dict[str, int],
    topic2id: dict[str, int],
    batch_size: int,
    num_workers: int,
    seed: int,
    smoke_test: bool,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    frames = {split: frame.copy() for split, frame in splits.items()}
    arrays = {split: embeddings[split] for split in splits}
    if smoke_test:
        frames = {split: frame.head(100).copy() for split, frame in frames.items()}
        arrays = {split: array[:100] for split, array in arrays.items()}

    datasets = {
        split: EmbeddingMultiTaskDataset(
            embeddings=arrays[split],
            frame=frame,
            sentiment2id=sentiment2id,
            topic2id=topic2id,
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
            collate_fn=collate_embeddings,
            generator=generator,
        ),
        "dev": DataLoader(
            datasets["dev"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_embeddings,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_embeddings,
        ),
    }
    return loaders, frames


def train_one_epoch(
    model: PhoBERTMultiTaskHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    sentiment_criterion: nn.Module,
    topic_criterion: nn.Module,
    lambda_topic: float,
    device: torch.device,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_sentiment_loss = 0.0
    total_topic_loss = 0.0
    total_examples = 0

    for embeddings, sentiment_labels, topic_labels in loader:
        embeddings = embeddings.to(device)
        sentiment_labels = sentiment_labels.to(device)
        topic_labels = topic_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        sentiment_logits, topic_logits = model(embeddings)
        sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
        topic_loss = topic_criterion(topic_logits, topic_labels)
        loss = sentiment_loss + lambda_topic * topic_loss
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = len(sentiment_labels)
        total_loss += loss.item() * batch_size
        total_sentiment_loss += sentiment_loss.item() * batch_size
        total_topic_loss += topic_loss.item() * batch_size
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "sentiment_loss": total_sentiment_loss / max(total_examples, 1),
        "topic_loss": total_topic_loss / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: PhoBERTMultiTaskHead,
    loader: DataLoader,
    device: torch.device,
    sentiment_labels: list[str],
    topic_labels: list[str],
) -> dict[str, Any]:
    model.eval()
    true_sentiment: list[int] = []
    pred_sentiment: list[int] = []
    conf_sentiment: list[float] = []
    true_topic: list[int] = []
    pred_topic: list[int] = []
    conf_topic: list[float] = []

    for embeddings, sentiment_target, topic_target in loader:
        embeddings = embeddings.to(device)
        sentiment_logits, topic_logits = model(embeddings)

        sentiment_prob = torch.softmax(sentiment_logits, dim=1)
        topic_prob = torch.softmax(topic_logits, dim=1)
        sentiment_conf, sentiment_pred = sentiment_prob.max(dim=1)
        topic_conf, topic_pred = topic_prob.max(dim=1)

        true_sentiment.extend(sentiment_target.tolist())
        pred_sentiment.extend(sentiment_pred.cpu().tolist())
        conf_sentiment.extend(sentiment_conf.cpu().tolist())
        true_topic.extend(topic_target.tolist())
        pred_topic.extend(topic_pred.cpu().tolist())
        conf_topic.extend(topic_conf.cpu().tolist())

    true_sentiment_array = np.array(true_sentiment)
    pred_sentiment_array = np.array(pred_sentiment)
    true_topic_array = np.array(true_topic)
    pred_topic_array = np.array(pred_topic)

    sentiment_true_labels = [sentiment_labels[index] for index in true_sentiment_array]
    sentiment_pred_labels = [sentiment_labels[index] for index in pred_sentiment_array]
    topic_true_labels = [topic_labels[index] for index in true_topic_array]
    topic_pred_labels = [topic_labels[index] for index in pred_topic_array]

    return {
        "sentiment": classification_summary(sentiment_true_labels, sentiment_pred_labels),
        "topic": classification_summary(topic_true_labels, topic_pred_labels),
        "joint_accuracy": float(
            (
                (true_sentiment_array == pred_sentiment_array)
                & (true_topic_array == pred_topic_array)
            ).mean()
        ),
        "sentiment_per_class": per_class_report(
            sentiment_true_labels,
            sentiment_pred_labels,
            sentiment_labels,
        ),
        "topic_per_class": per_class_report(
            topic_true_labels,
            topic_pred_labels,
            topic_labels,
        ),
        "true_sentiment": true_sentiment_array,
        "pred_sentiment": pred_sentiment_array,
        "confidence_sentiment": np.array(conf_sentiment),
        "true_topic": true_topic_array,
        "pred_topic": pred_topic_array,
        "confidence_topic": np.array(conf_topic),
    }


def plot_learning_curve(history: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["epoch"], history["train_loss"], marker="o", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["dev_sentiment_f1"], marker="s", color="tab:orange")
    ax2.plot(history["epoch"], history["dev_topic_f1"], marker="^", color="tab:green")
    ax2.set_ylabel("Dev macro-F1", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("PhoBERT frozen multi-task — learning curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def prediction_frame(
    frame: pd.DataFrame,
    *,
    sentiment_labels: list[str],
    topic_labels: list[str],
    result: dict[str, Any],
) -> pd.DataFrame:
    output = frame[["text", "sentiment", "topic"]].copy().reset_index(drop=True)
    output["sentiment_prediction"] = [
        sentiment_labels[index] for index in result["pred_sentiment"]
    ]
    output["sentiment_confidence"] = result["confidence_sentiment"]
    output["topic_prediction"] = [
        topic_labels[index] for index in result["pred_topic"]
    ]
    output["topic_confidence"] = result["confidence_topic"]
    output["joint_correct"] = (
        (result["true_sentiment"] == result["pred_sentiment"])
        & (result["true_topic"] == result["pred_topic"])
    )
    return output


def comparison_table(reports_dir: Path) -> pd.DataFrame:
    results_path = reports_dir / "results.csv"
    if not results_path.exists():
        return pd.DataFrame()
    results = pd.read_csv(results_path)
    wanted = results[
        (results["task"].isin(["sentiment", "topic"]))
        & (results["split"] == "test")
        & (
            results["model"].isin(
                [
                    "tfidf_logreg",
                    "textcnn_single",
                    "textcnn_multitask",
                    "phobert_frozen_multitask",
                ]
            )
        )
    ]
    return wanted.pivot_table(
        index="task",
        columns="model",
        values="macro_f1",
        aggfunc="first",
    )


def write_report(
    *,
    run_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    best_epoch: int,
    train_time_sec: float,
    reports_dir: Path,
    comparison: pd.DataFrame,
) -> None:
    rows = []
    for task in ("sentiment", "topic"):
        for split in ("dev", "test"):
            summary = metrics[split][task]
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "accuracy": round(summary["accuracy"], 4),
                    "macro_f1": round(summary["macro_f1"], 4),
                    "weighted_f1": round(summary["weighted_f1"], 4),
                }
            )
    for split in ("dev", "test"):
        rows.append(
            {
                "task": "joint",
                "split": split,
                "accuracy": round(metrics[split]["joint_accuracy"], 4),
                "macro_f1": "",
                "weighted_f1": "",
            }
        )

    lines = [
        "# PhoBERT frozen multi-task — UIT-VSFC",
        "",
        f"- Run ID: `{run_id}`",
        f"- Seed: `{config['seed']}`",
        f"- Encoder: `{config['data']['model_name']}` (frozen)",
        "- Pooling: `[CLS]`",
        f"- Lambda topic: `{config['lambda_topic']}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Train time: `{train_time_sec:.1f}s`",
        "",
        "## Kết quả",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        "## So sánh test macro-F1",
        "",
        markdown_table(comparison.round(4), index=True),
        "",
    ]

    for task in ("sentiment", "topic"):
        for split in ("dev", "test"):
            lines.extend(
                [
                    f"## {task} — {split}",
                    "",
                    markdown_table(metrics[split][f"{task}_per_class"].round(4)),
                    "",
                    f"![{task} {split}](figures/phobert_multitask_{task}_{split}_confusion.png)",
                    "",
                ]
            )

    lines.extend(
        [
            "## Learning curve",
            "",
            "![PhoBERT learning curve](figures/phobert_multitask_learning_curve.png)",
            "",
            "## Nhận xét",
            "",
            "- PhoBERT được đóng băng; chỉ train head phân loại.",
            "- Embedding `[CLS]` được trích một lần và cache trong `data/processed/uit_vsfc_phobert/`.",
            "- Loss: `L_sentiment + lambda * L_topic`.",
            "- Chọn epoch tốt nhất theo trung bình macro-F1 của hai tác vụ trên dev.",
            "",
        ]
    )
    (reports_dir / "phobert_multitask_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    config_path: str | Path,
    reports_dir: str | Path,
    models_dir: str | Path,
    runs_dir: str | Path,
    smoke_test: bool,
    force_embeddings: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if seed is not None:
        config["seed"] = seed
    set_seed(config["seed"])
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(DEFAULT_LABEL_MAP_PATH)

    tokenized_dir = resolve_path(config["data"]["tokenized_dir"])
    embedding_dir = resolve_path(config["data"]["embedding_dir"])
    splits = build_tokenized_splits(output_dir=tokenized_dir)
    embeddings = load_or_extract_embeddings(
        tokenized_dir=tokenized_dir,
        output_dir=embedding_dir,
        model_name=config["data"]["model_name"],
        max_len=config["data"]["max_len"],
        batch_size=config["data"]["embedding_batch_size"],
        force=force_embeddings,
    )

    sentiment_labels = [
        label_map.sentiment_id2label[index]
        for index in sorted(label_map.sentiment_id2label)
    ]
    topic_labels = [
        label_map.topic_id2label[index]
        for index in sorted(label_map.topic_id2label)
    ]

    loaders, frames = build_loaders(
        embeddings=embeddings,
        splits=splits,
        sentiment2id=label_map.sentiment_label2id,
        topic2id=label_map.topic_label2id,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        seed=config["seed"],
        smoke_test=smoke_test,
    )

    model = PhoBERTMultiTaskHead(
        input_dim=int(embeddings["train"].shape[1]),
        shared_hidden=config["model"]["shared_hidden"],
        num_sentiment_classes=len(sentiment_labels),
        num_topic_classes=len(topic_labels),
        dropout=config["model"]["dropout"],
    ).to(device)

    train_sentiment = [
        label_map.sentiment_label2id[str(label)]
        for label in frames["train"]["sentiment"].tolist()
    ]
    train_topic = [
        label_map.topic_label2id[str(label)]
        for label in frames["train"]["topic"].tolist()
    ]
    sentiment_weights = class_weights_for(
        train_sentiment,
        len(sentiment_labels),
        config["training"]["class_weight"],
    )
    topic_weights = class_weights_for(
        train_topic,
        len(topic_labels),
        config["training"]["class_weight"],
    )
    if sentiment_weights is not None:
        sentiment_weights = sentiment_weights.to(device)
    if topic_weights is not None:
        topic_weights = topic_weights.to(device)

    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)
    topic_criterion = nn.CrossEntropyLoss(weight=topic_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    models_dir = Path(models_dir)
    runs_dir = Path(runs_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("phobert_multitask_%Y%m%d_%H%M%S")
    git_hash = git_commit()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    epochs = 2 if smoke_test else config["training"]["epochs"]
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = config["training"]["patience"]
    history: list[dict[str, float | int]] = []
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        losses = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            sentiment_criterion,
            topic_criterion,
            config["lambda_topic"],
            device,
            config["training"]["max_grad_norm"],
        )
        dev_result = evaluate(
            model,
            loaders["dev"],
            device,
            sentiment_labels,
            topic_labels,
        )
        score = (
            dev_result["sentiment"]["macro_f1"]
            + dev_result["topic"]["macro_f1"]
        ) / 2
        history.append(
            {
                "epoch": epoch,
                "train_loss": losses["loss"],
                "train_sentiment_loss": losses["sentiment_loss"],
                "train_topic_loss": losses["topic_loss"],
                "dev_sentiment_f1": dev_result["sentiment"]["macro_f1"],
                "dev_topic_f1": dev_result["topic"]["macro_f1"],
                "dev_joint_accuracy": dev_result["joint_accuracy"],
                "selection_score": score,
            }
        )
        print(
            f"epoch {epoch:02d} | loss={losses['loss']:.4f} | "
            f"dev_sent_f1={dev_result['sentiment']['macro_f1']:.4f} | "
            f"dev_topic_f1={dev_result['topic']['macro_f1']:.4f} | "
            f"joint={dev_result['joint_accuracy']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "config": config,
                    "sentiment2id": label_map.sentiment_label2id,
                    "topic2id": label_map.topic_label2id,
                    "sentiment_labels": sentiment_labels,
                    "topic_labels": topic_labels,
                    "embedding_model": config["data"]["model_name"],
                    "max_len": config["data"]["max_len"],
                    "lambda_topic": config["lambda_topic"],
                },
                models_dir / "phobert_multitask_best.pt",
            )
        elif epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    train_time_sec = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("No checkpoint was saved")

    model.load_state_dict(best_state)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        reports_dir / "phobert_multitask_history.csv",
        index=False,
        encoding="utf-8",
    )
    plot_learning_curve(
        history_frame,
        figures_dir / "phobert_multitask_learning_curve.png",
    )

    metrics: dict[str, Any] = {}
    inference_ms: dict[str, float] = {}
    for split in ("dev", "test"):
        infer_start = time.perf_counter()
        result = evaluate(
            model,
            loaders[split],
            device,
            sentiment_labels,
            topic_labels,
        )
        inference_sec = time.perf_counter() - infer_start
        inference_ms[split] = inference_sec * 1000 / max(len(frames[split]), 1)
        metrics[split] = result

        result["sentiment_per_class"].to_csv(
            reports_dir / f"phobert_multitask_sentiment_{split}_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        result["topic_per_class"].to_csv(
            reports_dir / f"phobert_multitask_topic_{split}_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        plot_confusion_matrix(
            [sentiment_labels[index] for index in result["true_sentiment"]],
            [sentiment_labels[index] for index in result["pred_sentiment"]],
            sentiment_labels,
            figures_dir / f"phobert_multitask_sentiment_{split}_confusion.png",
            f"PhoBERT frozen sentiment — {split}",
        )
        plot_confusion_matrix(
            [topic_labels[index] for index in result["true_topic"]],
            [topic_labels[index] for index in result["pred_topic"]],
            topic_labels,
            figures_dir / f"phobert_multitask_topic_{split}_confusion.png",
            f"PhoBERT frozen topic — {split}",
        )

        prediction_frame(
            frames[split],
            sentiment_labels=sentiment_labels,
            topic_labels=topic_labels,
            result=result,
        ).to_csv(
            run_dir / f"{split}_predictions.csv",
            index=False,
            encoding="utf-8",
        )

    result_rows: list[dict[str, Any]] = []
    for task in ("sentiment", "topic"):
        for split in ("dev", "test"):
            summary = metrics[split][task]
            result_rows.append(
                {
                    "run_id": run_id,
                    "model": "phobert_frozen_multitask",
                    "task": task,
                    "split": split,
                    "lambda": config["lambda_topic"],
                    "seed": config["seed"],
                    "git_commit": git_hash,
                    "sent_f1": summary["macro_f1"] if task == "sentiment" else "",
                    "topic_f1": summary["macro_f1"] if task == "topic" else "",
                    "joint_accuracy": "",
                    "accuracy": summary["accuracy"],
                    "macro_f1": summary["macro_f1"],
                    "weighted_f1": summary["weighted_f1"],
                    "train_time_sec": train_time_sec,
                    "inference_ms": inference_ms[split],
                }
            )
    for split in ("dev", "test"):
        value = metrics[split]["joint_accuracy"]
        result_rows.append(
            {
                "run_id": run_id,
                "model": "phobert_frozen_multitask",
                "task": "joint",
                "split": split,
                "lambda": config["lambda_topic"],
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

    results_path = reports_dir / "results.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "phobert_frozen_multitask"]
        results = pd.concat([existing, pd.DataFrame(result_rows)], ignore_index=True)
    else:
        results = pd.DataFrame(result_rows)
    results.to_csv(results_path, index=False, encoding="utf-8")

    metrics_payload = {
        "run_id": run_id,
        "git_commit": git_hash,
        "config": config,
        "device": str(device),
        "best_epoch": best_epoch,
        "train_time_sec": train_time_sec,
        "metrics": {
            split: {
                "sentiment": metrics[split]["sentiment"],
                "topic": metrics[split]["topic"],
                "joint_accuracy": metrics[split]["joint_accuracy"],
            }
            for split in ("dev", "test")
        },
        "checkpoint": display_path(models_dir / "phobert_multitask_best.pt"),
    }
    (reports_dir / "phobert_multitask_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        run_id=run_id,
        config=config,
        metrics=metrics,
        best_epoch=best_epoch,
        train_time_sec=train_time_sec,
        reports_dir=reports_dir,
        comparison=comparison_table(reports_dir),
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "train_time_sec": train_time_sec,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen PhoBERT multi-task heads.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run(
        config_path=args.config,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        smoke_test=args.smoke_test,
        force_embeddings=args.force_embeddings,
        seed=args.seed,
    )
    print(f"run_id: {result['run_id']}")
    print(f"device: {result['device']}")
    print(f"best epoch: {result['best_epoch']}")
    print(f"test sentiment macro-F1: {result['metrics']['test']['sentiment']['macro_f1']:.4f}")
    print(f"test topic macro-F1: {result['metrics']['test']['topic']['macro_f1']:.4f}")
    print(f"test joint accuracy: {result['metrics']['test']['joint_accuracy']:.4f}")


if __name__ == "__main__":
    main()
