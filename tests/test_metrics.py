from __future__ import annotations

import pytest

from src.metrics import (
    classification_summary,
    confusion_matrix_frame,
    per_class_report,
)


def test_classification_summary_perfect() -> None:
    y_true = ["negative", "neutral", "positive", "positive"]
    y_pred = ["negative", "neutral", "positive", "positive"]

    summary = classification_summary(y_true, y_pred)

    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["weighted_f1"] == 1.0


def test_per_class_report() -> None:
    y_true = ["negative", "negative", "positive", "positive"]
    y_pred = ["negative", "positive", "positive", "positive"]

    report = per_class_report(y_true, y_pred, ["negative", "positive"])

    assert report["label"].tolist() == ["negative", "positive"]
    assert report["support"].tolist() == [2, 2]
    assert report.loc[0, "recall"] == 0.5
    assert report.loc[1, "precision"] == pytest.approx(2 / 3)


def test_confusion_matrix_frame() -> None:
    y_true = ["negative", "positive", "positive"]
    y_pred = ["negative", "negative", "positive"]

    matrix = confusion_matrix_frame(y_true, y_pred, ["negative", "positive"])

    assert matrix.shape == (2, 2)
    assert matrix.loc["true_negative", "pred_negative"] == 1
    assert matrix.loc["true_positive", "pred_negative"] == 1
    assert matrix.loc["true_positive", "pred_positive"] == 1
