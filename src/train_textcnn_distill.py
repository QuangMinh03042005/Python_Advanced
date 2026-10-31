from __future__ import annotations

import argparse
import copy
import json
import os
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import DistillTextCNNDataset, Vocab, collate_distill
from src.data.load import DEFAULT_DATA_DIR, DEFAULT_LABEL_MAP_PATH, load_label_map
from src.data.preprocess import build_tokenized_splits
from src.metrics import (
    classification_summary,
    per_class_report,
    plot_confusion_matrix,
)
from src.models.textcnn import TextCNNMultiTask
from src.report_paths import TEXTCNN_DISTILL_DIR, resolve_results_path
from src.train_textcnn_multitask import (
    class_weights_for,
    display_path,
    git_commit,
    load_config,
    markdown_table,
    prediction_frame,
    resolve_path,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "textcnn_distill.yaml"
DEFAULT_REPORTS_DIR = TEXTCNN_DISTILL_DIR
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "textcnn_distilled"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"


def build_loaders(
    *,
    splits: dict[str, pd.DataFrame],
    teacher_probs: dict[str, dict[str, np.ndarray]],
    vocab: Vocab,
    sentiment2id: dict[str, int],
    topic2id: dict[str, int],
    max_len: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    smoke_test: bool,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    frames = {split: frame.copy() for split, frame in splits.items()}
    probabilities = {
        split: {
            "sentiment": teacher_probs[split]["sentiment"],
            "topic": teacher_probs[split]["topic"],
        }
        for split in splits
    }
    if smoke_test:
        frames = {split: frame.head(100).copy() for split, frame in frames.items()}
        probabilities = {
            split: {
                "sentiment": values["sentiment"][:100],
                "topic": values["topic"][:100],
            }
            for split, values in probabilities.items()
        }

    datasets = {
        split: DistillTextCNNDataset(
            frame=frame,
            vocab=vocab,
            sentiment2id=sentiment2id,
            topic2id=topic2id,
            sentiment_probs=probabilities[split]["sentiment"],
            topic_probs=probabilities[split]["topic"],
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
            collate_fn=collate_distill,
            generator=generator,
        ),
        "dev": DataLoader(
            datasets["dev"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_distill,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_distill,
        ),
    }
    return loaders, frames


def train_one_epoch(
    model: TextCNNMultiTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    sentiment_criterion: nn.Module,
    topic_criterion: nn.Module,
    *,
    lambda_topic: float,
    temperature: float,
    distill_weight: float,
    device: torch.device,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "hard_loss": 0.0,
        "soft_loss": 0.0,
    }
    total_examples = 0

    for input_ids, sentiment_labels, topic_labels, sentiment_probs, topic_probs in loader:
        input_ids = input_ids.to(device)
        sentiment_labels = sentiment_labels.to(device)
        topic_labels = topic_labels.to(device)
        sentiment_probs = sentiment_probs.to(device)
        topic_probs = topic_probs.to(device)

        optimizer.zero_grad(set_to_none=True)
        sentiment_logits, topic_logits = model(input_ids)
        sentiment_hard = sentiment_criterion(sentiment_logits, sentiment_labels)
        topic_hard = topic_criterion(topic_logits, topic_labels)

        sentiment_soft = F.kl_div(
            F.log_softmax(sentiment_logits / temperature, dim=1),
            sentiment_probs,
            reduction="batchmean",
        ) * (temperature**2)
        topic_soft = F.kl_div(
            F.log_softmax(topic_logits / temperature, dim=1),
            topic_probs,
            reduction="batchmean",
        ) * (temperature**2)

        hard_loss = sentiment_hard + lambda_topic * topic_hard
        soft_loss = sentiment_soft + lambda_topic * topic_soft
        loss = hard_loss + distill_weight * soft_loss
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = len(sentiment_labels)
        totals["loss"] += loss.item() * batch_size
        totals["hard_loss"] += hard_loss.item() * batch_size
        totals["soft_loss"] += soft_loss.item() * batch_size
        total_examples += batch_size

    return {key: value / max(total_examples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: TextCNNMultiTask,
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

    for input_ids, sentiment_target, topic_target, _, _ in loader:
        input_ids = input_ids.to(device)
        sentiment_logits, topic_logits = model(input_ids)
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

    plt.title("TextCNN student distillation — learning curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def comparison_table(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    results = pd.read_csv(results_path)
    wanted = results[
        (results["task"].isin(["sentiment", "topic"]))
        & (results["split"] == "test")
        & (
            results["model"].isin(
                [
                    "textcnn_multitask",
                    "textcnn_distilled",
                    "phobert_keyword_multitask",
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
    history: pd.DataFrame,
    best_epoch: int,
    train_time_sec: float,
    parameter_count: int,
    model_size_mb: float,
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
        "# TextCNN student distillation — UIT-VSFC",
        "",
        f"- Run ID: `{run_id}`",
        f"- Seed: `{config['seed']}`",
        "- Teacher: PhoBERT keyword-aware",
        "- Student: TextCNN multi-task",
        f"- Temperature: `{config['temperature']}`",
        f"- Distill weight: `{config['distill_weight']}`",
        f"- Lambda topic: `{config['lambda_topic']}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Train time: `{train_time_sec:.1f}s`",
        f"- Student parameters: `{parameter_count:,}`",
        f"- Student size: `{model_size_mb:.2f} MB`",
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
                    f"![{task} {split}](figures/textcnn_distill_{task}_{split}_confusion.png)",
                    "",
                ]
            )

    lines.extend(
        [
            "## Learning curve",
            "",
            "![Student learning curve](figures/textcnn_distill_learning_curve.png)",
            "",
            "## Nhận xét",
            "",
            "- Student học từ hard label + soft label của teacher.",
            "- Mục tiêu là trade-off giữa F1 và chi phí suy luận, không nhất thiết vượt teacher.",
            "- So sánh trực tiếp với TextCNN hard-label và PhoBERT teacher trong `reports/results.csv`.",
            "",
        ]
    )
    (reports_dir / "textcnn_distill_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    reports_dir: str | Path,
    models_dir: str | Path,
    runs_dir: str | Path,
    smoke_test: bool,
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
    teacher_dir = resolve_path(config["data"]["teacher_dir"])
    splits = build_tokenized_splits(data_dir=data_dir, output_dir=tokenized_dir)
    teacher_probs = {
        split: {
            "sentiment": np.load(teacher_dir / f"{split}_sentiment_probs.npy"),
            "topic": np.load(teacher_dir / f"{split}_topic_probs.npy"),
        }
        for split in ("train", "dev", "test")
    }

    sentiment_labels = [
        label_map.sentiment_id2label[index]
        for index in sorted(label_map.sentiment_id2label)
    ]
    topic_labels = [
        label_map.topic_id2label[index]
        for index in sorted(label_map.topic_id2label)
    ]

    vocab = Vocab.build(
        (str(tokens).split() for tokens in splits["train"]["tokens"]),
        min_freq=config["data"]["min_freq"],
        max_size=config["data"]["max_vocab_size"],
    )
    loaders, frames = build_loaders(
        splits=splits,
        teacher_probs=teacher_probs,
        vocab=vocab,
        sentiment2id=label_map.sentiment_label2id,
        topic2id=label_map.topic_label2id,
        max_len=config["data"]["max_len"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        seed=config["seed"],
        smoke_test=smoke_test,
    )

    model = TextCNNMultiTask(
        vocab_size=len(vocab),
        num_sentiment_classes=len(sentiment_labels),
        num_topic_classes=len(topic_labels),
        embed_dim=config["model"]["embed_dim"],
        num_filters=config["model"]["num_filters"],
        kernel_sizes=tuple(config["model"]["kernel_sizes"]),
        shared_hidden=config["model"]["shared_hidden"],
        dropout=config["model"]["dropout"],
        embedding_dropout=config["model"]["embedding_dropout"],
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

    run_id = datetime.now().strftime("textcnn_distill_%Y%m%d_%H%M%S")
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
            lambda_topic=config["lambda_topic"],
            temperature=config["temperature"],
            distill_weight=config["distill_weight"],
            device=device,
            max_grad_norm=config["training"]["max_grad_norm"],
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
                "train_hard_loss": losses["hard_loss"],
                "train_soft_loss": losses["soft_loss"],
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
                    "vocab": vocab.token2id,
                    "sentiment2id": label_map.sentiment_label2id,
                    "topic2id": label_map.topic_label2id,
                    "sentiment_labels": sentiment_labels,
                    "topic_labels": topic_labels,
                    "max_len": config["data"]["max_len"],
                    "temperature": config["temperature"],
                    "distill_weight": config["distill_weight"],
                },
                models_dir / "textcnn_distilled_best.pt",
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
        reports_dir / "textcnn_distill_history.csv",
        index=False,
        encoding="utf-8",
    )
    plot_learning_curve(
        history_frame,
        figures_dir / "textcnn_distill_learning_curve.png",
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
            reports_dir / f"textcnn_distill_sentiment_{split}_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        result["topic_per_class"].to_csv(
            reports_dir / f"textcnn_distill_topic_{split}_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        plot_confusion_matrix(
            [sentiment_labels[index] for index in result["true_sentiment"]],
            [sentiment_labels[index] for index in result["pred_sentiment"]],
            sentiment_labels,
            figures_dir / f"textcnn_distill_sentiment_{split}_confusion.png",
            f"Distilled TextCNN sentiment — {split}",
        )
        plot_confusion_matrix(
            [topic_labels[index] for index in result["true_topic"]],
            [topic_labels[index] for index in result["pred_topic"]],
            topic_labels,
            figures_dir / f"textcnn_distill_topic_{split}_confusion.png",
            f"Distilled TextCNN topic — {split}",
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

    parameter_count = model.count_parameters()
    model_size_mb = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) / (1024**2)

    result_rows: list[dict[str, Any]] = []
    for task in ("sentiment", "topic"):
        for split in ("dev", "test"):
            summary = metrics[split][task]
            result_rows.append(
                {
                    "run_id": run_id,
                    "model": "textcnn_distilled",
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
                "model": "textcnn_distilled",
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

    results_path = resolve_results_path(reports_dir)
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "textcnn_distilled"]
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
        "parameter_count": parameter_count,
        "model_size_mb": model_size_mb,
        "metrics": {
            split: {
                "sentiment": metrics[split]["sentiment"],
                "topic": metrics[split]["topic"],
                "joint_accuracy": metrics[split]["joint_accuracy"],
            }
            for split in ("dev", "test")
        },
        "checkpoint": display_path(models_dir / "textcnn_distilled_best.pt"),
    }
    (reports_dir / "textcnn_distill_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        run_id=run_id,
        config=config,
        metrics=metrics,
        history=history_frame,
        best_epoch=best_epoch,
        train_time_sec=train_time_sec,
        parameter_count=parameter_count,
        model_size_mb=model_size_mb,
        reports_dir=reports_dir,
        comparison=comparison_table(results_path),
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "model_size_mb": model_size_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill PhoBERT teacher into TextCNN student.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    result = run(
        config_path=args.config,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    print(f"run_id: {result['run_id']}")
    print(f"best epoch: {result['best_epoch']}")
    print(f"student parameters: {result['parameter_count']:,}")
    print(f"student size: {result['model_size_mb']:.2f} MB")
    print(f"test sentiment macro-F1: {result['metrics']['test']['sentiment']['macro_f1']:.4f}")
    print(f"test topic macro-F1: {result['metrics']['test']['topic']['macro_f1']:.4f}")
    print(f"test joint accuracy: {result['metrics']['test']['joint_accuracy']:.4f}")


if __name__ == "__main__":
    main()
