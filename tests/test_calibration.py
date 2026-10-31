from __future__ import annotations

import numpy as np

from src.calibration import expected_calibration_error, fit_temperature


def test_ece_perfect_calibration() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    ece, rows = expected_calibration_error(probs, labels)
    assert ece == 0.0
    assert rows


def test_ece_detects_overconfidence() -> None:
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 1])
    ece, _ = expected_calibration_error(probs, labels)
    assert abs(ece - 0.5) < 1e-9


def test_temperature_is_positive() -> None:
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(200, 3))
    labels = rng.integers(0, 3, size=200)
    temperature = fit_temperature(logits, labels)
    assert 0.05 <= temperature <= 10.0
