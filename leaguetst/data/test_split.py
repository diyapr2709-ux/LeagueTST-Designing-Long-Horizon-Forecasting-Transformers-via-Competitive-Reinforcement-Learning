"""The one door to the test rows, used only by ``scripts/eval_test.py``.

The full benchmark files live outside the repository in the vault, which
only the project owner decrypts. Every call appends a line to
``results/test_access.log`` before any test row is returned, so gate G3
can show whether the test split was touched and by which prereg file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from leaguetst.data.datasets import BENCHMARKS, Standardizer, WindowDataset, load_benchmark_frame, split_borders
from leaguetst.data.trainval import read_manifest

ACCESS_LOG = Path("results/test_access.log")


def load_test_split(
    vault_dir: str | Path,
    trainval_dir: str | Path,
    name: str,
    input_len: int,
    horizon: int,
    prereg_id: str,
    log_path: str | Path = ACCESS_LOG,
) -> tuple[WindowDataset, Standardizer]:
    """Test windows of one benchmark, after logging the access.

    The vault file must be the exact file the trainval copy was cut from
    (same SHA-256 as the manifest's ``source_sha256``).
    """
    if not prereg_id:
        raise ValueError("a prereg id is required to open the test split")
    spec = BENCHMARKS[name]
    entry = read_manifest(trainval_dir)["datasets"][name]
    source = Path(vault_dir) / spec.relative_path
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != entry["source_sha256"]:
        raise ValueError(f"{source} is not the file the trainval copy was cut from")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prereg": prereg_id,
        "dataset": name,
        "horizon": horizon,
        "source_sha256": digest,
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")

    values = load_benchmark_frame(vault_dir, name).to_numpy()
    borders = split_borders(len(values), input_len, spec.split_scheme)
    if borders["val"][1] != int(entry["val_end"]):
        raise ValueError("the vault file's borders differ from the manifest")
    standardizer = Standardizer.fit(values[borders["train"][0] : borders["train"][1]])
    scaled = standardizer.transform(values)
    start, end = borders["test"]
    return WindowDataset(scaled[start:end], input_len, horizon), standardizer
