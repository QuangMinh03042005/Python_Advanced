from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data.load import (
    DEFAULT_DATA_DIR,
    DEFAULT_LABEL_MAP_PATH,
    LabelMap,
    load_label_map,
    load_splits,
)
from src.metrics import (
    classification_summary,
    per_class_report,
    plot_confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "baseline"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_vectorizer(config: dict[str, Any]) -> FeatureUnion:
    vectorizer_config = config["vectorizer"]
    common = {
        "min_df": vectorizer_config["min_df"],
        "max_df": vectorizer_config["max_df"],
        "sublinear_tf": vectorizer_config["sublinear_tf"],
        "lowercase": vectorizer_config["lowercase"],
    }
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(vectorizer_config["word_ngram_range"]),
        max_features=vectorizer_config["word_max_features"],
        **common,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(vectorizer_config["char_ngram_range"]),
        max_features=vectorizer_config["char_max_features"],
        **common,
    )
    return FeatureUnion([("word", word), ("char", char)])


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    model_config = config["model"]
    classifier = LogisticRegression(
        C=model_config["C"],
        max_iter=model_config["max_iter"],
        solver=model_config["solver"],
        class_weight=model_config["class_weight"],
        random_state=config["seed"],
    )
    return Pipeline([("features", build_vectorizer(config)), ("classifier", classifier)])


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
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


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _run_task(
    *,
    task: str,
    label_column: str,
    labels: list[str],
    train: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    config: dict[str, Any],
    run_id: str,
    reports_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    runs_dir: Path,
) -> dict[str, Any]:
    pipeline = build_pipeline(config)
    fit_start = time.perf_counter()
    pipeline.fit(train["text"], train[label_column])
    train_time_sec = time.perf_counter() - fit_start

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{task}_tfidf_logreg.joblib"
    joblib.dump(pipeline, model_path)

    metrics: dict[str, dict[str, float]] = {}
    inference_ms: dict[str, float] = {}
    per_class: dict[str, pd.DataFrame] = {}
    predictions: dict[str, pd.DataFrame] = {}
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for split in ("dev", "test"):
        frame = splits[split]
        infer_start = time.perf_counter()
        y_pred = pipeline.predict(frame["text"])
        inference_sec = time.perf_counter() - infer_start
        y_true = frame[label_column].tolist()

        summary = classification_summary(y_true, y_pred)
        metrics[split] = summary
        inference_ms[split] = inference_sec * 1000 / max(len(frame), 1)
        per_class[split] = per_class_report(y_true, y_pred, labels)

        per_class[split].to_csv(
            reports_dir / f"baseline_{task}_{split}_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            labels,
            figures_dir / f"baseline_{task}_{split}_confusion.png",
            f"Baseline {task} — {split}",
        )

        prediction_frame = frame[["text", label_column]].copy()
        prediction_frame["prediction"] = y_pred
        prediction_frame["confidence"] = pipeline.predict_proba(frame["text"]).max(axis=1)
        prediction_frame.to_csv(
            run_dir / f"{task}_{split}_predictions.csv",
            index=False,
            encoding="utf-8",
        )
        predictions[split] = prediction_frame

    return {
        "pipeline": pipeline,
        "model_path": model_path,
        "train_time_sec": train_time_sec,
        "inference_ms": inference_ms,
        "metrics": metrics,
        "per_class": per_class,
        "predictions": predictions,
    }


def _joint_accuracy(
    sentiment_predictions: pd.DataFrame,
    topic_predictions: pd.DataFrame,
    frame: pd.DataFrame,
) -> float:
    sentiment_correct = sentiment_predictions["prediction"].to_numpy() == frame["sentiment"].to_numpy()
    topic_correct = topic_predictions["prediction"].to_numpy() == frame["topic"].to_numpy()
    return float((sentiment_correct & topic_correct).mean())


def _write_report(
    *,
    run_id: str,
    config: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, float]]],
    per_class: dict[str, dict[str, pd.DataFrame]],
    joint_accuracy: dict[str, float],
    model_paths: dict[str, Path],
    reports_dir: Path,
) -> None:
    rows = []
    for task in ("sentiment", "topic"):
        for split in ("dev", "test"):
            summary = metrics[task][split]
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "accuracy": round(summary["accuracy"], 4),
                    "macro_f1": round(summary["macro_f1"], 4),
                    "weighted_f1": round(summary["weighted_f1"], 4),
                }
            )
    for split, value in joint_accuracy.items():
        rows.append(
            {
                "task": "joint",
                "split": split,
                "accuracy": round(value, 4),
                "macro_f1": "",
                "weighted_f1": "",
            }
        )

    lines = [
        "# Baseline — TF-IDF + Logistic Regression",
        "",
        f"- Run ID: `{run_id}`",
        f"- Seed: `{config['seed']}`",
        "- Vectorizer: word 1–2 grams + char_wb 3–5 grams",
        "- Classifier: Logistic Regression, `class_weight=balanced`",
        f"- Model sentiment: `{_display_path(model_paths['sentiment'])}`",
        f"- Model topic: `{_display_path(model_paths['topic'])}`",
        "",
        "## Kết quả",
        "",
        _markdown_table(pd.DataFrame(rows)),
        "",
        "## Per-class",
        "",
        "### Sentiment — dev",
        "",
        _markdown_table(per_class["sentiment"]["dev"].round(4)),
        "",
        "### Sentiment — test",
        "",
        _markdown_table(per_class["sentiment"]["test"].round(4)),
        "",
        "### Topic — dev",
        "",
        _markdown_table(per_class["topic"]["dev"].round(4)),
        "",
        "### Topic — test",
        "",
        _markdown_table(per_class["topic"]["test"].round(4)),
        "",
        "## Confusion matrix",
        "",
        "![Sentiment dev](figures/baseline_sentiment_dev_confusion.png)",
        "",
        "![Sentiment test](figures/baseline_sentiment_test_confusion.png)",
        "",
        "![Topic dev](figures/baseline_topic_dev_confusion.png)",
        "",
        "![Topic test](figures/baseline_topic_test_confusion.png)",
        "",
        "## Nhận xét",
        "",
        "- Đây là baseline tuyến tính; chưa dùng tách từ tiếng Việt.",
        "- `neutral` ít mẫu nên cần xem per-class F1, không chỉ accuracy.",
        "- Kết quả này là mốc so sánh cho TextCNN và PhoBERT.",
        "",
    ]
    (reports_dir / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_baseline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    label_map: LabelMap = load_label_map()
    split_frames = load_splits(data_dir=data_dir)
    train = split_frames["train"]

    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    models_dir = Path(models_dir)
    runs_dir = Path(runs_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("baseline_tfidf_%Y%m%d_%H%M%S")
    git_commit = _git_commit()

    sentiment_labels = [
        label_map.sentiment_id2label[index]
        for index in sorted(label_map.sentiment_id2label)
    ]
    topic_labels = [
        label_map.topic_id2label[index]
        for index in sorted(label_map.topic_id2label)
    ]

    sentiment_result = _run_task(
        task="sentiment",
        label_column="sentiment",
        labels=sentiment_labels,
        train=train,
        splits=split_frames,
        config=config,
        run_id=run_id,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
    )
    topic_result = _run_task(
        task="topic",
        label_column="topic",
        labels=topic_labels,
        train=train,
        splits=split_frames,
        config=config,
        run_id=run_id,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
    )

    joint_accuracy = {
        split: _joint_accuracy(
            sentiment_result["predictions"][split],
            topic_result["predictions"][split],
            split_frames[split],
        )
        for split in ("dev", "test")
    }

    result_rows = []
    for task_name, task_result in (("sentiment", sentiment_result), ("topic", topic_result)):
        for split in ("dev", "test"):
            summary = task_result["metrics"][split]
            result_rows.append(
                {
                    "run_id": run_id,
                    "model": "tfidf_logreg",
                    "task": task_name,
                    "split": split,
                    "lambda": "",
                    "seed": config["seed"],
                    "git_commit": git_commit,
                    "sent_f1": summary["macro_f1"] if task_name == "sentiment" else "",
                    "topic_f1": summary["macro_f1"] if task_name == "topic" else "",
                    "joint_accuracy": "",
                    "accuracy": summary["accuracy"],
                    "macro_f1": summary["macro_f1"],
                    "weighted_f1": summary["weighted_f1"],
                    "train_time_sec": task_result["train_time_sec"],
                    "inference_ms": task_result["inference_ms"][split],
                }
            )
    for split, value in joint_accuracy.items():
        result_rows.append(
            {
                "run_id": run_id,
                "model": "tfidf_logreg",
                "task": "joint",
                "split": split,
                "lambda": "",
                "seed": config["seed"],
                "git_commit": git_commit,
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
        existing = existing[existing["model"] != "tfidf_logreg"]
        results = pd.concat([existing, pd.DataFrame(result_rows)], ignore_index=True)
    else:
        results = pd.DataFrame(result_rows)
    results.to_csv(results_path, index=False, encoding="utf-8")

    metrics_payload = {
        "run_id": run_id,
        "git_commit": git_commit,
        "config": config,
        "sentiment": sentiment_result["metrics"],
        "topic": topic_result["metrics"],
        "joint_accuracy": joint_accuracy,
    }
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_report(
        run_id=run_id,
        config=config,
        metrics={"sentiment": sentiment_result["metrics"], "topic": topic_result["metrics"]},
        per_class={"sentiment": sentiment_result["per_class"], "topic": topic_result["per_class"]},
        joint_accuracy=joint_accuracy,
        model_paths={
            "sentiment": sentiment_result["model_path"],
            "topic": topic_result["model_path"],
        },
        reports_dir=reports_dir,
    )

    return {
        "run_id": run_id,
        "sentiment": sentiment_result["metrics"],
        "topic": topic_result["metrics"],
        "joint_accuracy": joint_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    result = run_baseline(
        data_dir=args.data_dir,
        config_path=args.config,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
    )
    print(f"run_id: {result['run_id']}")
    print(f"sentiment dev/test macro-F1: {result['sentiment']['dev']['macro_f1']:.4f} / {result['sentiment']['test']['macro_f1']:.4f}")
    print(f"topic dev/test macro-F1: {result['topic']['dev']['macro_f1']:.4f} / {result['topic']['test']['macro_f1']:.4f}")
    print(f"joint accuracy dev/test: {result['joint_accuracy']['dev']:.4f} / {result['joint_accuracy']['test']:.4f}")


if __name__ == "__main__":
    main()
