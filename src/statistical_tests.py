from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from src.report_paths import EVALUATION_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_OUTPUT = EVALUATION_DIR / "statistical_tests.csv"
DEFAULT_REPORT = EVALUATION_DIR / "statistical_tests.md"


def find_latest_predictions(root: Path) -> Path:
    files = list(root.rglob("test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No test_predictions.csv under {root}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_seed_predictions(seed: int) -> dict[str, pd.DataFrame]:
    base = PROJECT_ROOT / ".tmp" / "seeds"
    roots = {
        "keyword": base / f"phobert_keyword_s{seed}" / "runs",
        "frozen": base / f"phobert_s{seed}" / "runs",
        "textcnn": base / f"textcnn_multi_s{seed}" / "runs",
    }
    frames = {
        name: pd.read_csv(find_latest_predictions(root))
        for name, root in roots.items()
    }
    reference_text = frames["keyword"]["text"].reset_index(drop=True)
    for name, frame in frames.items():
        frame = frame.reset_index(drop=True)
        frames[name] = frame
        if not frame["text"].equals(reference_text):
            raise ValueError(f"Prediction rows are not aligned for {name}, seed {seed}")
    return frames


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    p_value = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return b, c, p_value


def bootstrap_macro_f1_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = float(
        f1_score(y_true, pred_a, average="macro")
        - f1_score(y_true, pred_b, average="macro")
    )
    differences = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.integers(0, n, size=n)
        differences[index] = (
            f1_score(y_true[sample], pred_a[sample], average="macro")
            - f1_score(y_true[sample], pred_b[sample], average="macro")
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, float(low), float(high)


def bootstrap_accuracy_diff(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    observed = float(correct_a.mean() - correct_b.mean())
    differences = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.integers(0, n, size=n)
        differences[index] = correct_a[sample].mean() - correct_b[sample].mean()
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, float(low), float(high)


def _correctness(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "sentiment":
        return (frame["sentiment"] == frame["sentiment_prediction"]).to_numpy()
    if task == "topic":
        return (frame["topic"] == frame["topic_prediction"]).to_numpy()
    if task == "joint":
        return (
            (frame["sentiment"] == frame["sentiment_prediction"])
            & (frame["topic"] == frame["topic_prediction"])
        ).to_numpy()
    raise ValueError(f"Unknown task: {task}")


def run(seeds: tuple[int, ...], n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        frames = load_seed_predictions(seed)
        reference = frames["keyword"]
        y_sentiment = reference["sentiment"].to_numpy()
        y_topic = reference["topic"].to_numpy()

        for candidate in ("frozen", "textcnn"):
            for task in ("sentiment", "topic", "joint"):
                correct_keyword = _correctness(reference, task)
                correct_candidate = _correctness(frames[candidate], task)
                b, c, p_value = mcnemar_exact(correct_keyword, correct_candidate)

                if task == "sentiment":
                    diff, low, high = bootstrap_macro_f1_diff(
                        y_sentiment,
                        reference["sentiment_prediction"].to_numpy(),
                        frames[candidate]["sentiment_prediction"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                elif task == "topic":
                    diff, low, high = bootstrap_macro_f1_diff(
                        y_topic,
                        reference["topic_prediction"].to_numpy(),
                        frames[candidate]["topic_prediction"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                else:
                    diff, low, high = bootstrap_accuracy_diff(
                        correct_keyword,
                        correct_candidate,
                        n_boot=n_boot,
                        seed=seed,
                    )

                rows.append(
                    {
                        "seed": seed,
                        "comparison": f"keyword_vs_{candidate}",
                        "task": task,
                        "b_keyword_only": b,
                        "c_candidate_only": c,
                        "mcnemar_p": p_value,
                        "diff": diff,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                    }
                )
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, path: Path, n_boot: int) -> None:
    lines = [
        "# Statistical tests",
        "",
        "- Test set: 3.166 câu.",
        f"- So sánh trên predictions của 3 seed: {', '.join(str(s) for s in sorted(results['seed'].unique()))}.",
        "- McNemar exact test cho correctness từng câu.",
        f"- Bootstrap percentile 95% CI với {n_boot} lần resample.",
        "- p-value chưa hiệu chỉnh đa so sánh; Bonferroni alpha cho 18 test là 0.0028.",
        "",
        "## McNemar",
        "",
        "| Seed | Comparison | Task | b | c | p-value |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['seed']} | {row['comparison']} | {row['task']} | "
            f"{row['b_keyword_only']} | {row['c_candidate_only']} | {row['mcnemar_p']:.4g} |"
        )

    lines += [
        "",
        "## Bootstrap difference",
        "",
        "| Seed | Comparison | Task | Difference | 95% CI | CI excludes 0 |",
        "|---:|---|---|---:|---|---|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['seed']} | {row['comparison']} | {row['task']} | "
            f"{row['diff']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{'yes' if row['ci_excludes_zero'] else 'no'} |"
        )

    lines += [
        "",
        "## Tổng hợp độ nhất quán",
        "",
        "| Comparison | Task | McNemar p<0.05 / 3 seed | CI excludes 0 / 3 seed |",
        "|---|---|---:|---:|",
    ]
    for (comparison, task), group in results.groupby(["comparison", "task"]):
        lines.append(
            f"| {comparison} | {task} | "
            f"{int((group['mcnemar_p'] < 0.05).sum())}/3 | "
            f"{int(group['ci_excludes_zero'].sum())}/3 |"
        )

    lines += [
        "",
        "## Cách đọc",
        "",
        "- `b`: keyword đúng, candidate sai.",
        "- `c`: keyword sai, candidate đúng.",
        "- McNemar p nhỏ ⇒ tỷ lệ hai loại lỗi khác nhau có ý nghĩa.",
        "- CI không chứa 0 ⇒ khác biệt có ý nghĩa ở mức 95%.",
        "- Ưu tiên kết luận nhất quán trên cả 3 seed, không chỉ một seed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="McNemar and bootstrap tests for seed runs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    results = run(tuple(args.seeds), args.n_boot)
    results.to_csv(args.output, index=False, encoding="utf-8")
    write_report(results, args.report, args.n_boot)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
