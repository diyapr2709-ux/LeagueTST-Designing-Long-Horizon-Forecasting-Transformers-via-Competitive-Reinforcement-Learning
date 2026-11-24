"""YAML, JSON and checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping with the safe loader, which builds no Python objects."""
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a mapping at the top level, not {type(loaded).__name__}")
    return loaded


def atomic_write(path: str | Path, write: Callable[[Path], None]) -> None:
    """Call ``write`` on a temporary sibling of ``path``, then rename it into place.

    A crash or interrupt mid-write leaves the previous file, or none, but
    never a truncated one; a reader never sees a half-written record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A random suffix keeps concurrent writers, threads included, off each
    # other's temporary file; open() keeps the usual umask permissions.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_json(data: Any, path: str | Path) -> None:
    """Write ``data`` as indented JSON, atomically, creating parent directories."""

    def write(target: Path) -> None:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

    atomic_write(path, write)


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_state_dict(state: dict[str, torch.Tensor], path: str | Path) -> str:
    """Save model weights atomically and return the file's SHA-256 digest."""
    atomic_write(path, lambda target: torch.save(state, target))
    return sha256_file(path)
