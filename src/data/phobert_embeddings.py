from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_tokenized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_phobert"
DEFAULT_MODEL_NAME = "vinai/phobert-base"
SPLITS = ("train", "dev", "test")


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_is_valid(output_dir: Path, metadata: dict, expected: dict) -> bool:
    if not metadata:
        return False
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return all((output_dir / f"{split}_cls.npy").exists() for split in SPLITS)


@torch.no_grad()
def extract_embeddings(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    *,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)
        cls_embeddings = output.last_hidden_state[:, 0, :]
        batches.append(cls_embeddings.cpu().numpy().astype(np.float32))

        if start % (batch_size * 20) == 0:
            print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(batches, axis=0)


def load_or_extract_embeddings(
    *,
    tokenized_dir: str | Path = DEFAULT_TOKENIZED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    max_len: int = 128,
    batch_size: int = 16,
    force: bool = False,
) -> dict[str, np.ndarray]:
    tokenized_dir = Path(tokenized_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    expected = {
        "model_name": model_name,
        "max_len": max_len,
        "pooling": "cls",
    }

    metadata = _load_metadata(metadata_path)
    if not force and _cache_is_valid(output_dir, metadata, expected):
        print(f"Using cached embeddings in {output_dir}")
        return {
            split: np.load(output_dir / f"{split}_cls.npy")
            for split in SPLITS
        }

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings: dict[str, np.ndarray] = {}
    for split in SPLITS:
        frame = pd.read_csv(tokenized_dir / f"{split}.csv")
        texts = frame["tokens"].astype(str).tolist()
        print(f"Extracting {split}: {len(texts)} sentences")
        embeddings[split] = extract_embeddings(
            texts,
            tokenizer,
            model,
            max_len=max_len,
            batch_size=batch_size,
            device=device,
        )
        np.save(output_dir / f"{split}_cls.npy", embeddings[split])

    metadata = {
        **expected,
        "hidden_size": int(embeddings["train"].shape[1]),
        "splits": {split: int(embeddings[split].shape[0]) for split in SPLITS},
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen PhoBERT CLS embeddings.")
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    embeddings = load_or_extract_embeddings(
        tokenized_dir=args.tokenized_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        force=args.force,
    )
    for split, array in embeddings.items():
        print(f"{split}: {array.shape}")


if __name__ == "__main__":
    main()
