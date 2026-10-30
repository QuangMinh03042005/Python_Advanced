from __future__ import annotations

from pathlib import Path

import pytest

from src.data.load import (
    DEFAULT_DATA_DIR,
    SPLITS,
    load_all,
    load_label_map,
    load_splits,
    read_split,
)


def _write_split(root: Path, split: str, texts: list[str], sentiments: list[str], topics: list[str]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    (split_dir / "sentences.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    (split_dir / "sentiments.txt").write_text("\n".join(sentiments) + "\n", encoding="utf-8")
    (split_dir / "topics.txt").write_text("\n".join(topics) + "\n", encoding="utf-8")


def test_label_map() -> None:
    label_map = load_label_map()

    assert label_map.sentiment_id2label == {0: "negative", 1: "neutral", 2: "positive"}
    assert label_map.topic_id2label == {
        0: "lecturer",
        1: "training_program",
        2: "facility",
        3: "others",
    }
    assert label_map.num_sentiment_labels == 3
    assert label_map.num_topic_labels == 4


def test_read_split_synthetic(temp_data_dir: Path) -> None:
    _write_split(
        temp_data_dir,
        "train",
        texts=["giảng viên dạy tốt", "bài tập quá khó"],
        sentiments=["2", "0"],
        topics=["0", "1"],
    )

    frame = read_split("train", data_dir=temp_data_dir)

    assert list(frame.columns) == [
        "split",
        "text",
        "sentiment_id",
        "topic_id",
        "sentiment",
        "topic",
    ]
    assert len(frame) == 2
    assert frame.loc[0, "sentiment"] == "positive"
    assert frame.loc[1, "topic"] == "training_program"


def test_line_count_mismatch(temp_data_dir: Path) -> None:
    _write_split(
        temp_data_dir,
        "train",
        texts=["một câu"],
        sentiments=["2", "0"],
        topics=["0"],
    )

    with pytest.raises(ValueError, match="Line count mismatch"):
        read_split("train", data_dir=temp_data_dir)


def test_unknown_label(temp_data_dir: Path) -> None:
    _write_split(
        temp_data_dir,
        "train",
        texts=["một câu"],
        sentiments=["9"],
        topics=["0"],
    )

    with pytest.raises(ValueError, match="Unknown sentiment labels"):
        read_split("train", data_dir=temp_data_dir)


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="UIT-VSFC raw data not downloaded")
def test_real_splits() -> None:
    splits = load_splits()

    assert list(splits) == list(SPLITS)
    assert {split: len(frame) for split, frame in splits.items()} == {
        "train": 11426,
        "dev": 1583,
        "test": 3166,
    }

    for frame in splits.values():
        assert frame["sentiment"].notna().all()
        assert frame["topic"].notna().all()
        assert set(frame["sentiment"]) <= {"negative", "neutral", "positive"}
        assert set(frame["topic"]) <= {
            "lecturer",
            "training_program",
            "facility",
            "others",
        }

    combined = load_all()
    assert len(combined) == 16175
