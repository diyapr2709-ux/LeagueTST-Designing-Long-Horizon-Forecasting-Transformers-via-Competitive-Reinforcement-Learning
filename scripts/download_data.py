"""Fetch the ETTh1 and ILI benchmark files into data/ and check their digests.

ETTh1 comes from the ETDataset GitHub repository. The ILI file is the
``national_illness.csv`` of the Autoformer dataset folder, mirrored on the
Hugging Face hub by the Time-Series-Library authors. Each file is pinned to
the SHA-256 digest of the copy behind every result in this repository, so a
changed mirror or a tampered download is refused instead of silently
shifting the numbers. Files already present are checked too. If a download
fails, place the file by hand as described in data/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaguetst.utils import atomic_write, sha256_file  # noqa: E402

# Digests of the files fetched from these URLs on 2026-10-29 and confirmed
# byte for byte against a fresh download on 2026-10-30.
FILES = {
    "ETT-small/ETTh1.csv": (
        "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        ["https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"],
    ),
    "illness/national_illness.csv": (
        "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
        ["https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv"],
    ),
}
MAX_BYTES = 50 * 2**20


def fetch(url: str, destination: Path, expected_sha256: str) -> bool:
    """Download ``url`` to ``destination`` if it is the pinned file; return success."""
    if not url.startswith("https://"):
        print("  refused: only https sources are allowed")
        return False
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            final_url = response.geturl()
            payload = response.read(MAX_BYTES + 1)
    except Exception as error:  # noqa: BLE001 - report and try the next mirror
        print(f"  failed: {error}")
        return False
    if not final_url.startswith("https://"):
        print(f"  refused: redirected to a non-https address {final_url}")
        return False
    if len(payload) > MAX_BYTES:
        print(f"  refused: larger than {MAX_BYTES} bytes")
        return False
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        print(f"  refused: SHA-256 {digest[:16]} is not the pinned {expected_sha256[:16]}")
        return False
    atomic_write(destination, lambda target: target.write_bytes(payload))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true", help="re-download files even if present")
    args = parser.parse_args()

    problems = []
    for relative, (expected, urls) in FILES.items():
        destination = Path(args.data_dir) / relative
        if destination.exists() and not args.force:
            if sha256_file(destination) == expected:
                print(f"{relative}: present, digest matches")
            else:
                print(f"{relative}: present but its digest differs from the pinned copy; rerun with --force")
                problems.append(relative)
            continue
        print(f"{relative}: downloading")
        if not any(fetch(url, destination, expected) for url in urls):
            problems.append(relative)
    if problems:
        print("not usable: " + ", ".join(problems))
        print("see data/README.md for the manual download steps")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
