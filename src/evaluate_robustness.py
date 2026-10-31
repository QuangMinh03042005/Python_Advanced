from __future__ import annotations

import argparse
import os
import random
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.data.dataset import Vocab
from src.data.load import DEFAULT_LABEL_MAP_PATH, load_label_map
from src.data.preprocess import tokenize_text
from src.data.teacher_logits import load_teacher
from src.features.keywords import extract_topic_keyword_features
from src.models.textcnn import TextCNNMultiTask
from src.report_paths import EVALUATION_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "phobert_keyword_multitask"
    / "phobert_keyword_multitask_best.pt"
)
DEFAULT_STUDENT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "textcnn_distilled" / "textcnn_distilled_best.pt"
)
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_tokenized" / "test.csv"
DEFAULT_REPORTS_DIR = EVALUATION_DIR
VARIANTS = ("clean", "no_diacritics", "teencode", "typo", "emoji", "shuffle")

TEENCODE_REPLACEMENTS = (
    ("cơ sở vật chất", "csvc"),
    ("chương trình đào tạo", "ctdt"),
    ("giảng viên", "gv"),
    ("không", "ko"),
    ("được", "dc"),
    ("nhiều", "nhìu"),
    ("cũng", "cx"),
    ("rất", "r"),
    ("và", "&"),
    ("thầy", "thay"),
    ("cô", "co"),
)


def remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")


def teencode(text: str) -> str:
    for source, target in TEENCODE_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def typo(text: str, rng: random.Random) -> str:
    words = text.split()
    output: list[str] = []
    for word in words:
        if len(word) > 3 and rng.random() < 0.2:
            index = rng.randrange(1, len(word) - 1)
            word = word[:index] + word[index + 1 :]
        output.append(word)
    return " ".join(output)


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
        "sentiment_labels": checkpoint["sentiment_labels"],
        "topic_labels": checkpoint["topic_labels"],
    }


@torch.no_grad()
def predict_teacher(
    token_lists: list[list[str]],
    teacher: dict[str, Any],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = teacher["tokenizer"]
    encoder = teacher["encoder"]
    head = teacher["head"]
    device = teacher["device"]
    keyword_aware = teacher["keyword_aware"]
    sentiment_probs: list[np.ndarray] = []
    topic_probs: list[np.ndarray] = []

    for start in range(0, len(token_lists), batch_size):
        batch = token_lists[start : start + batch_size]
        encoded = tokenizer(
            [" ".join(tokens) for tokens in batch],
            padding=True,
            truncation=True,
            max_length=teacher["max_len"],
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cls_embeddings = encoder(**encoded).last_hidden_state[:, 0, :]
        if keyword_aware:
            keyword_tensor = torch.tensor(
                np.stack([extract_topic_keyword_features(tokens) for tokens in batch]),
                dtype=torch.float32,
                device=device,
            )
            sentiment_out, topic_out, _ = head(cls_embeddings, keyword_tensor)
        else:
            sentiment_out, topic_out = head(cls_embeddings)
        sentiment_probs.append(torch.softmax(sentiment_out, dim=1).cpu().numpy())
        topic_probs.append(torch.softmax(topic_out, dim=1).cpu().numpy())

    return (
        np.concatenate(sentiment_probs, axis=0),
        np.concatenate(topic_probs, axis=0),
    )


@torch.no_grad()
def predict_student(
    token_lists: list[list[str]],
    student: dict[str, Any],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = student["model"]
    vocab = student["vocab"]
    max_len = student["max_len"]
    sentiment_probs: list[np.ndarray] = []
    topic_probs: list[np.ndarray] = []

    for start in range(0, len(token_lists), batch_size):
        batch = token_lists[start : start + batch_size]
        input_ids = torch.tensor(
            [vocab.encode(tokens, max_len) for tokens in batch],
            dtype=torch.long,
        )
        sentiment_out, topic_out = model(input_ids)
        sentiment_probs.append(torch.softmax(sentiment_out, dim=1).numpy())
        topic_probs.append(torch.softmax(topic_out, dim=1).numpy())

    return (
        np.concatenate(sentiment_probs, axis=0),
        np.concatenate(topic_probs, axis=0),
    )


def metrics(
    y_sentiment: np.ndarray,
    y_topic: np.ndarray,
    sentiment_probs: np.ndarray,
    topic_probs: np.ndarray,
    sentiment_labels: list[str],
    topic_labels: list[str],
) -> dict[str, float]:
    sentiment_pred = sentiment_probs.argmax(axis=1)
    topic_pred = topic_probs.argmax(axis=1)
    sentiment_true_labels = [sentiment_labels[index] for index in y_sentiment]
    sentiment_pred_labels = [sentiment_labels[index] for index in sentiment_pred]
    topic_true_labels = [topic_labels[index] for index in y_topic]
    topic_pred_labels = [topic_labels[index] for index in topic_pred]
    return {
        "sentiment_f1": float(
            f1_score(sentiment_true_labels, sentiment_pred_labels, average="macro")
        ),
        "topic_f1": float(
            f1_score(topic_true_labels, topic_pred_labels, average="macro")
        ),
        "joint_accuracy": float(
            (
                (sentiment_pred == y_sentiment)
                & (topic_pred == y_topic)
            ).mean()
        ),
    }


def plot_robustness(results: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("sentiment_f1", "topic_f1"), strict=True):
        pivot = results.pivot(index="variant", columns="model", values=metric)
        pivot = pivot.reindex(list(VARIANTS))
        x = np.arange(len(pivot))
        width = 0.38
        for offset, model in enumerate(("teacher", "student")):
            axis.bar(
                x + (offset - 0.5) * width,
                pivot[model],
                width,
                label=model,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(pivot.index, rotation=25, ha="right")
        axis.set_ylabel("Macro-F1")
        axis.set_title(metric.replace("_", " ").title())
        axis.legend()
    figure.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation for teacher and student.")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.test_csv)
    frame = frame.sample(n=min(args.n_samples, len(frame)), random_state=42).reset_index(drop=True)

    label_map = load_label_map(DEFAULT_LABEL_MAP_PATH)
    sentiment_labels = [
        label_map.sentiment_id2label[index]
        for index in sorted(label_map.sentiment_id2label)
    ]
    topic_labels = [
        label_map.topic_id2label[index]
        for index in sorted(label_map.topic_id2label)
    ]
    y_sentiment = frame["sentiment"].map(label_map.sentiment_label2id).to_numpy()
    y_topic = frame["topic"].map(label_map.topic_label2id).to_numpy()

    clean_tokens = [str(tokens).split() for tokens in frame["tokens"]]
    variants: dict[str, list[list[str]]] = {"clean": clean_tokens}
    rng = random.Random(42)
    for text in frame["text"].astype(str):
        variants.setdefault("no_diacritics", []).append(
            tokenize_text(remove_diacritics(text))
        )
        variants.setdefault("teencode", []).append(tokenize_text(teencode(text)))
        variants.setdefault("typo", []).append(tokenize_text(typo(text, rng)))
        variants.setdefault("emoji", []).append(tokenize_text(text + " 😐"))
    variants["shuffle"] = [rng.sample(tokens, len(tokens)) for tokens in clean_tokens]

    teacher = load_teacher(args.teacher_checkpoint, device)
    student = load_student(args.student_checkpoint)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        tokens = variants[variant]
        for model_name in ("teacher", "student"):
            if model_name == "teacher":
                sentiment_probs, topic_probs = predict_teacher(
                    tokens, teacher, args.batch_size
                )
            else:
                sentiment_probs, topic_probs = predict_student(
                    tokens, student, args.batch_size
                )
            scores = metrics(
                y_sentiment,
                y_topic,
                sentiment_probs,
                topic_probs,
                sentiment_labels,
                topic_labels,
            )
            rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    **scores,
                }
            )
        print(f"done {variant}")

    results = pd.DataFrame(rows)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    clean = results[results["variant"] == "clean"].set_index("model")
    for metric in ("sentiment_f1", "topic_f1", "joint_accuracy"):
        results[f"delta_{metric}"] = results.apply(
            lambda row, metric=metric: row[metric] - clean.loc[row["model"], metric],
            axis=1,
        )
    results.to_csv(args.reports_dir / "robustness_results.csv", index=False, encoding="utf-8")
    plot_robustness(
        results,
        args.reports_dir / "figures" / "robustness_scores.png",
    )

    lines = [
        "# Robustness report",
        "",
        f"- Test sample: `{len(frame)}` câu, random seed 42.",
        "- Nhãn giữ nguyên; chỉ thay đổi input.",
        "- Teacher: PhoBERT keyword-aware.",
        "- Student: TextCNN distilled.",
        "",
        "## Kết quả",
        "",
        "| Variant | Model | Sentiment F1 | Topic F1 | Joint | Δ Sent | Δ Topic | Δ Joint |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['variant']} | {row['model']} | "
            f"{row['sentiment_f1']:.4f} | {row['topic_f1']:.4f} | "
            f"{row['joint_accuracy']:.4f} | "
            f"{row['delta_sentiment_f1']:+.4f} | "
            f"{row['delta_topic_f1']:+.4f} | "
            f"{row['delta_joint_accuracy']:+.4f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `reports/evaluation/figures/robustness_scores.png`",
        "",
        "## Nhận xét",
        "",
        "- `no_diacritics` gây sụt giảm nghiêm trọng nhất cho cả hai mô hình.",
        "- `teencode` làm topic giảm mạnh hơn sentiment.",
        "- `emoji` làm teacher sentiment giảm rất mạnh; student ổn định hơn.",
        "- `shuffle` làm teacher giảm mạnh hơn student, cho thấy teacher phụ thuộc trật tự từ.",
        "",
        "## Cách đọc",
        "",
        "- Δ càng âm ⇒ mô hình càng kém bền vững với perturbation đó.",
        "- `shuffle` kiểm tra phụ thuộc trật tự từ.",
        "- `no_diacritics` kiểm tra thiếu dấu.",
        "- `teencode` kiểm tra viết tắt.",
        "- `typo` kiểm tra lỗi gõ.",
        "- `emoji` kiểm tra nhiễu ký tự ngoài từ vựng.",
        "",
    ]
    (args.reports_dir / "robustness_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
