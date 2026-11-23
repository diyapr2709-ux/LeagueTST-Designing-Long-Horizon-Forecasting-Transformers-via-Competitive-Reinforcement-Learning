"""Train and validation rows only, with split borders from a manifest.

``make_trainval`` copies the leading rows of each full benchmark file, the
training block and the validation block, byte for byte into
``data/trainval/`` and writes ``MANIFEST.json`` with the row borders that
the full file implies. Every later run loads from that folder, so no test
row is on disk where the search can read it.

Borders come from the manifest, never from the length of the file being
read. The ILI split is a ratio of the full file's length, so recomputing it
on the shorter copy would silently move both borders.

Asking for the test split here raises :class:`TestSplitLocked`. The only
way to the test rows is :mod:`leaguetst.data.test_split`, which logs every
access.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from leaguetst.data.datasets import (
    BENCHMARKS,
    Standardizer,
    WindowDataset,
    load_benchmark_frame,
    split_borders,
)

MANIFEST_NAME = "MANIFEST.json"
TRAINVAL_SPLITS = ("train", "val")


class TestSplitLocked(RuntimeError):
    """Raised when code on the search path asks for test rows."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_trainval(src_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Write the train and validation rows of both benchmarks plus the manifest.

    The full file is checked with the usual loader first. The copy keeps the
    header and the first ``val_end`` data lines unchanged, so parsing it
    gives exactly the values the full pipeline uses.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "train and validation rows only; borders are those of the full file",
        "datasets": {},
    }
    for name, spec in BENCHMARKS.items():
        source = src_dir / spec.relative_path
        frame = load_benchmark_frame(src_dir, name)
        n_rows = len(frame)
        # The input length only shifts where val and test start, not where
        # any block ends, so the spec's own length is used here.
        borders = split_borders(n_rows, spec.input_len, spec.split_scheme)
        train_end = borders["train"][1]
        val_end = borders["val"][1]

        raw = source.read_bytes()
        lines = raw.splitlines(keepends=True)
        if len(lines) != n_rows + 1:
            raise ValueError(f"{source}: {len(lines)} lines for {n_rows} rows plus a header")
        kept = b"".join(lines[: val_end + 1])
        target = out_dir / spec.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(kept)

        manifest["datasets"][name] = {
            "file": spec.relative_path,
            "sha256": _sha256_bytes(kept),
            "rows": val_end,
            "train_end": train_end,
            "val_end": val_end,
            "source_rows": n_rows,
            "source_sha256": _sha256_bytes(raw),
            "split_scheme": spec.split_scheme,
        }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_manifest(trainval_dir: str | Path) -> dict[str, Any]:
    path = Path(trainval_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/make_trainval.py")
    return json.loads(path.read_text(encoding="utf-8"))


def check_trainval_file(trainval_dir: str | Path, name: str) -> dict[str, Any]:
    """Return the manifest entry for ``name`` after checking the file's digest."""
    entry = read_manifest(trainval_dir)["datasets"][name]
    path = Path(trainval_dir) / entry["file"]
    digest = _sha256_bytes(path.read_bytes())
    if digest != entry["sha256"]:
        raise ValueError(f"{path}: SHA-256 {digest[:12]} differs from the manifest's {entry['sha256'][:12]}")
    return entry


def trainval_borders(entry: dict[str, Any], input_len: int) -> dict[str, tuple[int, int]]:
    """Row ranges of train and val inside the trainval file."""
    train_end, val_end = int(entry["train_end"]), int(entry["val_end"])
    if train_end <= input_len:
        raise ValueError("the training block is shorter than the input window")
    return {"train": (0, train_end), "val": (train_end - input_len, val_end)}


def build_trainval_splits(
    trainval_dir: str | Path,
    name: str,
    input_len: int,
    horizon: int,
    splits: tuple[str, ...] = TRAINVAL_SPLITS,
) -> tuple[dict[str, WindowDataset], Standardizer]:
    """Train and validation window datasets built from the trainval copy.

    The values, borders and standardizer equal those of
    :func:`leaguetst.data.build_splits` on the full file.
    """
    for split in splits:
        if split == "test":
            raise TestSplitLocked("the test split is not on the search path; see scripts/eval_test.py")
        if split not in TRAINVAL_SPLITS:
            raise ValueError(f"unknown split {split!r}")
    entry = check_trainval_file(trainval_dir, name)
    values = load_benchmark_frame(trainval_dir, name).to_numpy()
    if len(values) != int(entry["rows"]):
        raise ValueError(f"{name}: {len(values)} rows on disk, manifest says {entry['rows']}")
    borders = trainval_borders(entry, input_len)
    standardizer = Standardizer.fit(values[borders["train"][0] : borders["train"][1]])
    scaled = standardizer.transform(values)
    datasets = {s: WindowDataset(scaled[borders[s][0] : borders[s][1]], input_len, horizon) for s in splits}
    return datasets, standardizer


def window_counts(entry: dict[str, Any], input_len: int, horizon: int) -> dict[str, int]:
    """Windows per split: train N - L - T + 1, val (val rows) - T + 1."""
    train_end, val_end = int(entry["train_end"]), int(entry["val_end"])
    return {
        "train": train_end - input_len - horizon + 1,
        "val": (val_end - train_end) - horizon + 1,
    }


def scaled_values(trainval_dir: str | Path, name: str) -> np.ndarray:
    """All trainval rows standardised with the training block's statistics."""
    entry = check_trainval_file(trainval_dir, name)
    values = load_benchmark_frame(trainval_dir, name).to_numpy()
    train_end = int(entry["train_end"])
    return Standardizer.fit(values[:train_end]).transform(values)
