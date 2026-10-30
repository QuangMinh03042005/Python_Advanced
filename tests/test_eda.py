from __future__ import annotations

import pandas as pd

from src.data.eda import (
    add_length_features,
    cross_tab,
    distribution_table,
    duplicate_report,
    length_stats,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "train", "dev", "test"],
            "text": ["a b", "a b c", "d", "e f"],
            "sentiment": ["positive", "negative", "neutral", "positive"],
            "topic": ["lecturer", "facility", "others", "lecturer"],
        }
    )


def test_add_length_features() -> None:
    result = add_length_features(_frame())

    assert result["n_chars"].tolist() == [3, 5, 1, 3]
    assert result["n_tokens"].tolist() == [2, 3, 1, 2]


def test_distribution_table() -> None:
    table = distribution_table(_frame(), "sentiment", ["negative", "neutral", "positive"])

    assert table.loc["train", "negative"] == 1
    assert table.loc["train", "positive"] == 1
    assert table.loc["dev", "neutral"] == 1
    assert table.loc["test", "positive"] == 1
    assert table.loc["train", "neutral"] == 0


def test_cross_tab_order_and_normalization() -> None:
    frame = _frame()
    counts = cross_tab(
        frame,
        "sentiment",
        "topic",
        ["negative", "neutral", "positive"],
        ["lecturer", "training_program", "facility", "others"],
    )
    rates = cross_tab(
        frame,
        "sentiment",
        "topic",
        ["negative", "neutral", "positive"],
        ["lecturer", "training_program", "facility", "others"],
        normalize="index",
    )

    assert counts.index.tolist() == ["negative", "neutral", "positive"]
    assert counts.columns.tolist() == [
        "lecturer",
        "training_program",
        "facility",
        "others",
    ]
    assert counts.loc["negative", "facility"] == 1
    assert abs(float(rates.loc["positive", "lecturer"]) - 1.0) < 1e-9


def test_duplicate_report() -> None:
    splits = {
        "train": pd.DataFrame({"text": ["a", "a", "b"]}),
        "dev": pd.DataFrame({"text": ["a", "c"]}),
        "test": pd.DataFrame({"text": ["d"]}),
    }

    report = duplicate_report(splits)

    assert report["within"] == {"train": 1, "dev": 0, "test": 0}
    assert report["pairwise"] == {
        "train|dev": 1,
        "train|test": 0,
        "dev|test": 0,
    }


def test_length_stats() -> None:
    frame = add_length_features(_frame())

    table = length_stats(frame)

    assert len(table) == 8
    assert set(table["unit"]) == {"n_chars", "n_tokens"}
    assert set(table["split"]) == {"train", "dev", "test", "all"}
