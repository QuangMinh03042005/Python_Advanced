from __future__ import annotations

import pandas as pd

from src.models.baseline import build_pipeline, load_config


def _tiny_config() -> dict:
    return {
        "seed": 42,
        "vectorizer": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [2, 3],
            "word_max_features": 200,
            "char_max_features": 200,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
        },
        "model": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
        },
    }


def test_load_config_has_required_sections() -> None:
    config = load_config()

    assert "vectorizer" in config
    assert "model" in config
    assert config["seed"] == 42


def test_pipeline_fits_and_predicts() -> None:
    train = pd.DataFrame(
        {
            "text": [
                "giảng viên dạy rất tốt",
                "bài giảng hay và dễ hiểu",
                "môn học bình thường",
                "bài tập tạm được",
                "bài tập quá khó",
                "giảng viên dạy khó hiểu",
            ],
            "sentiment": [
                "positive",
                "positive",
                "neutral",
                "neutral",
                "negative",
                "negative",
            ],
        }
    )

    pipeline = build_pipeline(_tiny_config())
    pipeline.fit(train["text"], train["sentiment"])
    predictions = pipeline.predict(train["text"])

    assert len(predictions) == len(train)
    assert set(predictions) <= {"negative", "neutral", "positive"}
