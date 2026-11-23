"""Append-only run ledger, results/ledger.jsonl.

One line per trained (item, setting, seed, config). A row's id is a digest
of its identifying fields and its result, so the same record collected
twice is skipped and a changed result can never overwrite an old row.
Rows are only ever appended; nothing here edits or deletes a line.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

LEDGER = Path("results/ledger.jsonl")
KEY_FIELDS = ("item", "dataset", "horizon", "seed", "config_id")


def row_from_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    """The ledger row for one table record (per-epoch curves kept, train loss included)."""
    history = record.get("history", [])
    row = {
        "item": record["item"],
        "dataset": record["dataset"],
        "horizon": int(record["horizon"]),
        "input_len": int(record["input_len"]),
        "seed": int(record["seed"]),
        "config_id": record["config_id"],
        "model_config": record["model_config"],
        "n_params": int(record["n_params"]),
        "budget": record.get("budget"),
        "training": record["training"],
        "best_val_mse": record["best_val_mse"],
        "best_epoch": int(record["best_epoch"]),
        "epochs_run": int(record["epochs_run"]),
        "epochs_charged": int(record["epochs_run"]),
        "diverged": bool(record.get("diverged", False)),
        "val": record["val"],
        "val_curve": [h["val_mse"] for h in history],
        "train_curve": [h["train_loss"] for h in history],
        "epoch_seconds": [round(h["seconds"], 4) for h in history],
        "train_seconds": record.get("train_seconds"),
        "wall_seconds": record.get("wall_seconds"),
        "checkpoint_sha256": record.get("checkpoint_sha256", ""),
        "perwindow_sha256": record.get("perwindow_sha256", ""),
        "git_hash": record.get("git_hash", ""),
        "device": record.get("environment", {}).get("device", ""),
        "gpu": record.get("environment", {}).get("gpu", ""),
        "finished_at": record.get("environment", {}).get("finished_at", ""),
        "source": source,
    }
    identity = {k: row[k] for k in KEY_FIELDS}
    identity["val_mse"] = row["val"]["mse"]
    identity["best_val_mse"] = row["best_val_mse"]
    identity["checkpoint_sha256"] = row["checkpoint_sha256"]
    row["row_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return row


def read_rows(path: str | Path = LEDGER) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: unreadable ledger line") from error


def append_rows(rows: Iterable[dict[str, Any]], path: str | Path = LEDGER) -> dict[str, int]:
    """Append rows whose id is new. Returns counts of appended and skipped rows.

    A row with the same key fields as an existing row but a different id
    (a different result for the same run) is refused, because a run must
    not be silently replaced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        ids, keys = set(), {}
        for line in handle:
            if line.strip():
                existing = json.loads(line)
                ids.add(existing["row_id"])
                keys[tuple(existing[k] for k in KEY_FIELDS)] = existing["row_id"]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Check the whole batch before writing anything, so a refused batch
        # leaves the ledger as it was.
        new_rows, skipped = [], 0
        for row in rows:
            if row["row_id"] in ids:
                skipped += 1
                continue
            key = tuple(row[k] for k in KEY_FIELDS)
            if key in keys:
                raise ValueError(f"ledger already holds {key} with a different result ({keys[key]}); refusing to replace it")
            ids.add(row["row_id"])
            keys[key] = row["row_id"]
            new_rows.append({**row, "appended_at": stamp})
        for row in new_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        appended = len(new_rows)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return {"appended": appended, "skipped": skipped}


def table(item: str, dataset: str, horizon: int, path: str | Path = LEDGER) -> list[dict[str, Any]]:
    """All ledger rows of one item and setting."""
    return [r for r in read_rows(path) if r["item"] == item and r["dataset"] == dataset and int(r["horizon"]) == horizon]
