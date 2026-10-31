from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from src.data.dataset import TextCNNMultiTaskDataset, Vocab, collate_multitask
from src.models.textcnn import TextCNNMultiTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = PROJECT_ROOT / "data" / "processed" / "teacher_soft_labels"
DEFAULT_TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_tokenized"
DEFAULT_STUDENT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "textcnn_distilled" / "textcnn_distilled_best.pt"
)
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


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
        "sentiment2id": checkpoint["sentiment2id"],
        "topic2id": checkpoint["topic2id"],
    }


@torch.no_grad()
def student_logits(
    split: str,
    *,
    tokenized_dir: Path,
    checkpoint: dict[str, Any],
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(tokenized_dir / f"{split}.csv")
    dataset = TextCNNMultiTaskDataset(
        frame=frame,
        vocab=checkpoint["vocab"],
        sentiment2id=checkpoint["sentiment2id"],
        topic2id=checkpoint["topic2id"],
        max_len=checkpoint["max_len"],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multitask,
    )
    model = checkpoint["model"]
    sentiment_logits: list[np.ndarray] = []
    topic_logits: list[np.ndarray] = []
    for input_ids, _, _ in loader:
        sentiment_out, topic_out = model(input_ids)
        sentiment_logits.append(sentiment_out.cpu().numpy())
        topic_logits.append(topic_out.cpu().numpy())

    y_sentiment = frame["sentiment"].map(checkpoint["sentiment2id"]).to_numpy()
    y_topic = frame["topic"].map(checkpoint["topic2id"]).to_numpy()
    return (
        np.concatenate(sentiment_logits, axis=0),
        np.concatenate(topic_logits, axis=0),
        y_sentiment,
        y_topic,
    )


def teacher_logits(
    split: str,
    *,
    teacher_dir: Path,
    tokenized_dir: Path,
    sentiment2id: dict[str, int],
    topic2id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(tokenized_dir / f"{split}.csv")
    sentiment = np.load(teacher_dir / f"{split}_sentiment_logits.npy")
    topic = np.load(teacher_dir / f"{split}_topic_logits.npy")
    return (
        sentiment,
        topic,
        frame["sentiment"].map(sentiment2id).to_numpy(),
        frame["topic"].map(topic2id).to_numpy(),
    )


def nll_at_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    tensor_logits = torch.from_numpy(logits.astype(np.float64)) / temperature
    tensor_labels = torch.from_numpy(labels.astype(np.int64))
    return float(F.cross_entropy(tensor_logits, tensor_labels).item())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    result = minimize_scalar(
        lambda temperature: nll_at_temperature(logits, labels, float(temperature)),
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    return torch.softmax(
        torch.from_numpy(logits.astype(np.float64)) / temperature,
        dim=1,
    ).numpy()


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict[str, float]]]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= lower) & (confidences < upper)
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_confidence = float(confidences[mask].mean())
        avg_accuracy = float(correctness[mask].mean())
        ece += count / len(labels) * abs(avg_confidence - avg_accuracy)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": avg_confidence,
                "accuracy": avg_accuracy,
            }
        )
    return ece, rows


def reliability_plot(
    series: list[tuple[str, np.ndarray, np.ndarray, float]],
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    for label, probs, labels, ece in series:
        _, rows = expected_calibration_error(probs, labels)
        confidence = [row["confidence"] for row in rows]
        accuracy = [row["accuracy"] for row in rows]
        ax.plot(
            confidence,
            accuracy,
            marker="o",
            label=f"{label} (ECE={ece:.3f})",
        )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature scaling for teacher and student.")
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    teacher_meta = json.loads((args.teacher_dir / "metadata.json").read_text(encoding="utf-8"))
    sentiment2id = {label: index for index, label in enumerate(teacher_meta["sentiment_labels"])}
    topic2id = {label: index for index, label in enumerate(teacher_meta["topic_labels"])}

    student = load_student(args.student_checkpoint)
    data: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {
        "teacher": {},
        "student": {},
    }
    for split in ("dev", "test"):
        s_logits, t_logits, y_sent, y_topic = teacher_logits(
            split,
            teacher_dir=args.teacher_dir,
            tokenized_dir=args.tokenized_dir,
            sentiment2id=sentiment2id,
            topic2id=topic2id,
        )
        data["teacher"][split] = {
            "sentiment": (s_logits, y_sent),
            "topic": (t_logits, y_topic),
        }

        s_logits, t_logits, y_sent, y_topic = student_logits(
            split,
            tokenized_dir=args.tokenized_dir,
            checkpoint=student,
            batch_size=args.batch_size,
        )
        data["student"][split] = {
            "sentiment": (s_logits, y_sent),
            "topic": (t_logits, y_topic),
        }

    rows: list[dict[str, Any]] = []
    temperatures: dict[str, dict[str, float]] = {}
    reliability_series: dict[str, list[tuple[str, np.ndarray, np.ndarray, float]]] = {
        "sentiment": [],
        "topic": [],
    }

    for model_name in ("teacher", "student"):
        temperatures[model_name] = {}
        for task in ("sentiment", "topic"):
            dev_logits, dev_labels = data[model_name]["dev"][task]
            test_logits, test_labels = data[model_name]["test"][task]
            temperature = fit_temperature(dev_logits, dev_labels)
            temperatures[model_name][task] = temperature

            for split, logits, labels in (
                ("dev", dev_logits, dev_labels),
                ("test", test_logits, test_labels),
            ):
                before = probabilities(logits, 1.0)
                after = probabilities(logits, temperature)
                ece_before, _ = expected_calibration_error(before, labels)
                ece_after, _ = expected_calibration_error(after, labels)
                nll_before = nll_at_temperature(logits, labels, 1.0)
                nll_after = nll_at_temperature(logits, labels, temperature)
                accuracy = float((before.argmax(axis=1) == labels).mean())
                rows.append(
                    {
                        "model": model_name,
                        "task": task,
                        "split": split,
                        "temperature": temperature,
                        "accuracy": accuracy,
                        "ece_before": ece_before,
                        "ece_after": ece_after,
                        "nll_before": nll_before,
                        "nll_after": nll_after,
                    }
                )
                if split == "test":
                    reliability_series[task].extend(
                        [
                            (f"{model_name} before", before, labels, ece_before),
                            (f"{model_name} after", after, labels, ece_after),
                        ]
                    )

    results = pd.DataFrame(rows)
    results.to_csv(args.reports_dir / "calibration_results.csv", index=False, encoding="utf-8")
    (args.reports_dir / "calibration_temperatures.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "temperatures": temperatures,
                "student_checkpoint": str(args.student_checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for task in ("sentiment", "topic"):
        reliability_plot(
            reliability_series[task],
            f"Reliability — {task} (test)",
            args.reports_dir / "figures" / f"calibration_{task}_reliability.png",
        )

    lines = [
        "# Calibration report",
        "",
        "- Temperature scaling fit trên dev; đánh giá trên test.",
        "- ECE: Expected Calibration Error, 15 bins.",
        "- Temperature không làm thay đổi argmax/accuracy.",
        "",
        "## Test",
        "",
        "| Model | Task | Temperature | ECE before | ECE after | NLL before | NLL after |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in results[results["split"] == "test"].iterrows():
        lines.append(
            f"| {row['model']} | {row['task']} | {row['temperature']:.4f} | "
            f"{row['ece_before']:.4f} | {row['ece_after']:.4f} | "
            f"{row['nll_before']:.4f} | {row['nll_after']:.4f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `reports/figures/calibration_sentiment_reliability.png`",
        "- `reports/figures/calibration_topic_reliability.png`",
        "",
        "## Nhận xét",
        "",
        "- Trước calibration, student overconfident hơn teacher rõ rệt.",
        "- Sau temperature scaling, ECE của student giảm mạnh ở cả sentiment và topic.",
        "- Teacher topic cải thiện rõ; teacher sentiment cải thiện ít hơn.",
        "- Accuracy không đổi vì temperature scaling chỉ sửa confidence.",
        "",
        "## Cách đọc",
        "",
        "- ECE càng thấp càng tốt.",
        "- Temperature > 1 thường làm giảm overconfidence.",
        "- Nếu ECE giảm nhưng accuracy không đổi, calibration chỉ sửa confidence, không sửa dự đoán.",
        "",
    ]
    (args.reports_dir / "calibration_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
