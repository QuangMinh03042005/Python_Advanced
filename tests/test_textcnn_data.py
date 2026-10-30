from __future__ import annotations

import pandas as pd
import torch

from src.data.dataset import PAD_ID, UNK_ID, TextCNNDataset, Vocab, collate_batch


def test_vocab_build_encode_and_roundtrip(temp_data_dir) -> None:
    vocab = Vocab.build(
        [["giảng_viên", "dạy", "tốt"], ["giảng_viên", "dạy", "hay"]],
        min_freq=2,
    )

    assert len(vocab) == 4
    assert vocab.token2id["<pad>"] == PAD_ID
    assert vocab.token2id["<unk>"] == UNK_ID
    expected = [
        vocab.token2id["giảng_viên"],
        vocab.token2id["dạy"],
        UNK_ID,
        PAD_ID,
    ]
    assert vocab.encode(["giảng_viên", "dạy", "tốt"], max_len=4) == expected

    path = temp_data_dir / "vocab.json"
    vocab.save(path)
    loaded = Vocab.load(path)
    assert loaded.token2id == vocab.token2id


def test_dataset_and_collate() -> None:
    frame = pd.DataFrame(
        {
            "tokens": ["giảng_viên dạy tốt", "bài_tập khó"],
            "sentiment": ["positive", "negative"],
        }
    )
    vocab = Vocab.build([tokens.split() for tokens in frame["tokens"]], min_freq=1)
    dataset = TextCNNDataset(
        frame=frame,
        vocab=vocab,
        label_column="sentiment",
        label2id={"negative": 0, "positive": 1},
        max_len=4,
    )

    item = dataset[0]
    assert item["input_ids"].shape == (4,)
    assert item["label"].item() == 1

    input_ids, labels = collate_batch([dataset[0], dataset[1]])
    assert input_ids.shape == (2, 4)
    assert labels.shape == (2,)
    assert torch.equal(input_ids[0], item["input_ids"])
