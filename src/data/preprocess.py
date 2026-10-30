from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from underthesea import word_tokenize

from src.data.load import DEFAULT_DATA_DIR, SPLITS, load_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "uit_vsfc_tokenized"


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without removing sentiment cues."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def tokenize_text(text: str) -> list[str]:
    """Segment Vietnamese text into word tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return word_tokenize(normalized, format="text").split()


def preprocess_split(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["text"] = result["text"].map(normalize_text)
    result["tokens"] = result["text"].map(tokenize_text)
    result["n_tokens"] = result["tokens"].map(len)
    return result


def build_tokenized_splits(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Tokenize and cache all splits as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached = {
        split: output_dir / f"{split}.csv"
        for split in SPLITS
    }
    if not force and all(path.exists() for path in cached.values()):
        return {
            split: pd.read_csv(path)
            for split, path in cached.items()
        }

    splits = load_splits(data_dir=data_dir)
    result: dict[str, pd.DataFrame] = {}
    for split, frame in splits.items():
        processed = preprocess_split(frame)
        processed["tokens"] = processed["tokens"].map(lambda tokens: " ".join(tokens))
        processed.to_csv(cached[split], index=False, encoding="utf-8")
        result[split] = processed

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize and cache UIT-VSFC.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    splits = build_tokenized_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    for split, frame in splits.items():
        print(f"{split}: {len(frame)} rows")


if __name__ == "__main__":
    main()
