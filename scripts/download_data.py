"""Fetch the ETTh1 and ILI benchmark files into data/.

ETTh1 comes from the ETDataset GitHub repository. The ILI file is the
``national_illness.csv`` of the Autoformer dataset folder, mirrored on the
Hugging Face hub by the Time-Series-Library authors. If a download fails,
place the file by hand as described in data/README.md.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

FILES = {
    "ETT-small/ETTh1.csv": [
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    ],
    "illness/national_illness.csv": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv",
    ],
}


def fetch(url: str, destination: Path) -> bool:
    """Download ``url`` to ``destination``; return whether it looks like a CSV."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as error:  # noqa: BLE001 - report and try the next mirror
        print(f"  failed: {error}")
        return False
    if not payload.startswith(b"date"):
        print("  response is not a benchmark CSV")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    missing = []
    for relative, urls in FILES.items():
        destination = Path(args.data_dir) / relative
        if destination.exists() and not args.force:
            print(f"{relative}: present")
            continue
        print(f"{relative}: downloading")
        if not any(fetch(url, destination) for url in urls):
            missing.append(relative)
    if missing:
        print("could not fetch: " + ", ".join(missing))
        print("see data/README.md for the manual download steps")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
