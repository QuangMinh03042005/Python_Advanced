from __future__ import annotations

import argparse
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "uit_vsfc"

FILES: dict[str, dict[str, str]] = {
    "train": {
        "sentences.txt": "1nzak5OkrheRV1ltOGCXkT671bmjODLhP",
        "sentiments.txt": "1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv",
        "topics.txt": "14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C",
    },
    "dev": {
        "sentences.txt": "1sMJSR3oRfPc3fe1gK-V3W5F24tov_517",
        "sentiments.txt": "1GiY1AOp41dLXIIkgES4422AuDwmbUseL",
        "topics.txt": "1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W",
    },
    "test": {
        "sentences.txt": "1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n",
        "sentiments.txt": "1vkQS5gI0is4ACU58-AbWusnemw7KZNfO",
        "topics.txt": "1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB",
    },
}


def download_file(file_id: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"skip {target}")
        return

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, timeout=120, allow_redirects=True)
    response.raise_for_status()
    content = response.content

    if not content or content.lstrip().startswith(b"<"):
        raise RuntimeError(f"Invalid response for {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(f"wrote {target} ({len(content)} bytes)")


def check_files(data_dir: Path) -> None:
    missing: list[Path] = []
    for split, files in FILES.items():
        for filename in files:
            path = data_dir / split / filename
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path)
            else:
                print(f"ok {path} ({path.stat().st_size} bytes)")

    if missing:
        raise SystemExit(f"Missing or empty files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UIT-VSFC raw files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--check", action="store_true", help="Only verify existing files.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if args.check:
        check_files(data_dir)
        return

    for split, files in FILES.items():
        for filename, file_id in files.items():
            target = data_dir / split / filename
            if args.force and target.exists():
                target.unlink()
            download_file(file_id, target)

    check_files(data_dir)


if __name__ == "__main__":
    main()
