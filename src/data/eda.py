from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.load import DEFAULT_DATA_DIR, SPLITS, LabelMap, load_label_map, load_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
TOPIC_ORDER = ["lecturer", "training_program", "facility", "others"]


def add_length_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add character and whitespace-token length columns."""
    result = frame.copy()
    result["n_chars"] = result["text"].str.len().astype(int)
    result["n_tokens"] = result["text"].str.split().str.len().astype(int)
    return result


def distribution_table(
    frame: pd.DataFrame,
    column: str,
    order: list[str],
) -> pd.DataFrame:
    """Count rows by split and label, with split as index."""
    table = (
        frame.groupby(["split", column], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(SPLITS), columns=order, fill_value=0)
    )
    table.columns.name = None
    table.index.name = "split"
    return table


def cross_tab(
    frame: pd.DataFrame,
    row: str,
    column: str,
    row_order: list[str],
    column_order: list[str],
    normalize: str | None = None,
) -> pd.DataFrame:
    """Cross-tab with explicit row/column order."""
    if normalize is None:
        table = pd.crosstab(frame[row], frame[column])
    else:
        table = pd.crosstab(frame[row], frame[column], normalize=normalize)
    return table.reindex(index=row_order, columns=column_order, fill_value=0)


def length_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Length summary per split and overall."""
    rows: list[dict[str, object]] = []
    groups = [(split, frame[frame["split"] == split]) for split in SPLITS]
    groups.append(("all", frame))

    for split, group in groups:
        for column in ("n_chars", "n_tokens"):
            values = group[column]
            rows.append(
                {
                    "split": split,
                    "unit": column,
                    "count": len(values),
                    "mean": round(float(values.mean()), 2),
                    "median": round(float(values.median()), 2),
                    "p95": round(float(values.quantile(0.95)), 2),
                    "max": int(values.max()),
                }
            )
    return pd.DataFrame(rows)


def duplicate_report(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Within-split and cross-split duplicate counts."""
    within = {
        split: int(len(frame) - frame["text"].nunique())
        for split, frame in splits.items()
    }
    text_sets = {split: set(frame["text"]) for split, frame in splits.items()}
    pairwise = {
        f"{left}|{right}": len(text_sets[left] & text_sets[right])
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1 :]
    }
    return {"within": within, "pairwise": pairwise}


def _markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
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


def _plot_distribution(table: pd.DataFrame, title: str, path: Path) -> None:
    ax = table.plot(kind="bar", figsize=(10, 5), width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Split")
    ax.set_ylabel("Số câu")
    ax.legend(title="Nhãn", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_length_histogram(frame: pd.DataFrame, path: Path) -> None:
    values = frame["n_tokens"]
    median = float(values.median())
    p95 = float(values.quantile(0.95))

    plt.figure(figsize=(9, 5))
    sns.histplot(values, bins=40, kde=False)
    plt.axvline(median, color="orange", linestyle="--", label=f"median = {median:.0f}")
    plt.axvline(p95, color="red", linestyle="--", label=f"p95 = {p95:.0f}")
    plt.title("Phân bố độ dài câu (token)")
    plt.xlabel("Số token")
    plt.ylabel("Số câu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_heatmap(table: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Chủ đề")
    plt.ylabel("Cảm xúc")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_length_by_sentiment(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=frame,
        x="sentiment",
        y="n_tokens",
        order=SENTIMENT_ORDER,
        showfliers=False,
    )
    plt.title("Độ dài câu theo cảm xúc — train")
    plt.xlabel("Cảm xúc")
    plt.ylabel("Số token")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_eda(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict[str, object]:
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    label_map: LabelMap = load_label_map()
    splits = load_splits(data_dir=data_dir)
    train = add_length_features(splits["train"])
    all_data = add_length_features(pd.concat(splits.values(), ignore_index=True))

    sentiment_counts = distribution_table(all_data, "sentiment", SENTIMENT_ORDER)
    topic_counts = distribution_table(all_data, "topic", TOPIC_ORDER)
    sentiment_topic = cross_tab(
        train,
        "sentiment",
        "topic",
        SENTIMENT_ORDER,
        TOPIC_ORDER,
    )
    sentiment_topic_pct = cross_tab(
        train,
        "sentiment",
        "topic",
        SENTIMENT_ORDER,
        TOPIC_ORDER,
        normalize="index",
    )
    lengths = length_stats(all_data)
    duplicates = duplicate_report(splits)

    sentiment_counts.to_csv(reports_dir / "eda_sentiment_distribution.csv", encoding="utf-8")
    topic_counts.to_csv(reports_dir / "eda_topic_distribution.csv", encoding="utf-8")
    sentiment_topic.to_csv(reports_dir / "eda_sentiment_topic_crosstab.csv", encoding="utf-8")
    lengths.to_csv(reports_dir / "eda_length_stats.csv", index=False, encoding="utf-8")

    _plot_distribution(
        sentiment_counts,
        "Phân bố cảm xúc theo split",
        figures_dir / "sentiment_distribution.png",
    )
    _plot_distribution(
        topic_counts,
        "Phân bố chủ đề theo split",
        figures_dir / "topic_distribution.png",
    )
    _plot_length_histogram(train, figures_dir / "sentence_length_hist.png")
    _plot_heatmap(
        sentiment_topic_pct,
        "Tỷ lệ chủ đề trong từng lớp cảm xúc — train",
        figures_dir / "sentiment_topic_heatmap.png",
    )
    _plot_length_by_sentiment(train, figures_dir / "length_by_sentiment.png")

    p95_tokens = float(train["n_tokens"].quantile(0.95))
    max_len_suggestion = max(32, min(256, int((p95_tokens + 15) // 16 * 16)))
    shortest = train.nsmallest(5, "n_tokens")[["text", "n_tokens"]]
    longest = train.nlargest(5, "n_tokens")[["text", "n_tokens"]]

    report = _build_report(
        sentiment_counts=sentiment_counts,
        topic_counts=topic_counts,
        sentiment_topic=sentiment_topic,
        sentiment_topic_pct=sentiment_topic_pct,
        lengths=lengths,
        duplicates=duplicates,
        p95_tokens=p95_tokens,
        max_len_suggestion=max_len_suggestion,
        shortest=shortest,
        longest=longest,
    )
    (reports_dir / "eda_summary.md").write_text(report, encoding="utf-8")

    return {
        "sentiment_counts": sentiment_counts,
        "topic_counts": topic_counts,
        "sentiment_topic": sentiment_topic,
        "length_stats": lengths,
        "duplicates": duplicates,
        "p95_tokens": p95_tokens,
        "max_len_suggestion": max_len_suggestion,
    }


def _build_report(
    *,
    sentiment_counts: pd.DataFrame,
    topic_counts: pd.DataFrame,
    sentiment_topic: pd.DataFrame,
    sentiment_topic_pct: pd.DataFrame,
    lengths: pd.DataFrame,
    duplicates: dict[str, object],
    p95_tokens: float,
    max_len_suggestion: int,
    shortest: pd.DataFrame,
    longest: pd.DataFrame,
) -> str:
    within = duplicates["within"]
    pairwise = duplicates["pairwise"]
    neutral_train = int(sentiment_counts.loc["train", "neutral"])
    total_train = int(sentiment_counts.loc["train"].sum())
    lecturer_train = int(topic_counts.loc["train", "lecturer"])
    total_topic_train = int(topic_counts.loc["train"].sum())
    facility_total = int(sentiment_topic["facility"].sum())
    facility_negative = int(sentiment_topic.loc["negative", "facility"])
    facility_positive = int(sentiment_topic.loc["positive", "facility"])
    positive_lecturer = int(sentiment_topic.loc["positive", "lecturer"])
    positive_total = int(sentiment_topic.loc["positive"].sum())
    neutral_others = int(sentiment_topic.loc["neutral", "others"])
    neutral_total = int(sentiment_topic.loc["neutral"].sum())

    lines = [
        "# EDA — UIT-VSFC",
        "",
        "Báo cáo được sinh tự động bởi `python -m src.data.eda`.",
        "",
        "## 1. Quy mô và phân bố cảm xúc",
        "",
        _markdown_table(sentiment_counts),
        "",
        f"- `neutral` chỉ chiếm **{neutral_train / total_train:.1%}** câu train.",
        "- Cần dùng **macro-F1** và class weight thay vì chỉ accuracy.",
        "",
        "## 2. Phân bố chủ đề",
        "",
        _markdown_table(topic_counts),
        "",
        f"- `lecturer` chiếm **{lecturer_train / total_topic_train:.1%}** câu train.",
        "- `facility` và `others` ít hơn đáng kể; cần theo dõi per-class F1.",
        "",
        "## 3. Tương quan cảm xúc × chủ đề (train)",
        "",
        "### Số lượng",
        "",
        _markdown_table(sentiment_topic),
        "",
        "### Tỷ lệ theo hàng",
        "",
        _markdown_table(sentiment_topic_pct.round(3)),
        "",
        f"- `facility` gần như luôn tiêu cực: **{facility_negative}/{facility_total}** câu "
        f"({facility_negative / facility_total:.1%}), chỉ {facility_positive} câu tích cực.",
        f"- Phản hồi tích cực tập trung ở `lecturer`: **{positive_lecturer}/{positive_total}** "
        f"({positive_lecturer / positive_total:.1%}).",
        f"- Phản hồi trung tính có tỷ lệ `others` cao: **{neutral_others}/{neutral_total}** "
        f"({neutral_others / neutral_total:.1%}).",
        "- Sentiment và topic có tương quan mạnh; multi-task có thể học được tín hiệu hữu ích "
        "nhưng cũng có nguy cơ học shortcut theo topic.",
        "",
        "## 4. Độ dài câu",
        "",
        _markdown_table(lengths, index=False),
        "",
        f"- p95 số token trên train: **{p95_tokens:.0f}**.",
        f"- Gợi ý `max_len` ban đầu: **{max_len_suggestion}** token.",
        "- Cần kiểm tra lại trên dev trước khi chốt.",
        "",
        "### Câu ngắn nhất (train)",
        "",
        _markdown_table(shortest, index=False),
        "",
        "### Câu dài nhất (train)",
        "",
        _markdown_table(longest, index=False),
        "",
        "## 5. Trùng lặp",
        "",
        "### Trong từng split",
        "",
        _markdown_table(pd.DataFrame([within], index=["duplicates"])),
        "",
        "### Giữa các split",
        "",
        _markdown_table(pd.DataFrame([pairwise], index=["overlap"])),
        "",
        "## 6. Hình",
        "",
        "![Phân bố cảm xúc](figures/sentiment_distribution.png)",
        "",
        "![Phân bố chủ đề](figures/topic_distribution.png)",
        "",
        "![Độ dài câu](figures/sentence_length_hist.png)",
        "",
        "![Heatmap cảm xúc × chủ đề](figures/sentiment_topic_heatmap.png)",
        "",
        "![Độ dài theo cảm xúc](figures/length_by_sentiment.png)",
        "",
        "## 7. Kết luận cho bước modeling",
        "",
        "1. Dùng split chính thức, không trộn test.",
        "2. Ưu tiên macro-F1 cho cả hai tác vụ.",
        "3. Dùng class weight cho sentiment và topic.",
        "4. Chọn `max_len` theo p95 và kiểm tra lại trên dev.",
        "5. Ghi nhận câu trùng trong train khi làm sạch.",
        "6. Phân tích lỗi theo chủ đề, đặc biệt `facility` và `others`.",
        "7. Kiểm tra truncation: p95 là 33 token nhưng câu dài nhất là 159 token.",
        "8. Cẩn thận với shortcut sentiment–topic khi huấn luyện multi-task.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UIT-VSFC EDA.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    result = run_eda(data_dir=args.data_dir, reports_dir=args.reports_dir)
    print("EDA completed")
    print(f"p95 tokens: {result['p95_tokens']:.0f}")
    print(f"max_len suggestion: {result['max_len_suggestion']}")
    print(f"duplicates: {result['duplicates']}")


if __name__ == "__main__":
    main()
