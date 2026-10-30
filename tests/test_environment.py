import torch
from sklearn.metrics import f1_score
from torch import nn


def test_torch_cpu_forward() -> None:
    embedding = nn.Embedding(20, 8, padding_idx=0)
    conv = nn.Conv1d(8, 4, kernel_size=3)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])

    hidden = embedding(input_ids).transpose(1, 2)
    output = conv(hidden)

    assert output.shape == (1, 4, 4)


def test_macro_f1() -> None:
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    score = f1_score(y_true, y_pred, average="macro")

    assert 0.0 <= score <= 1.0
