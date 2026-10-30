"""Device selection."""

from __future__ import annotations

import torch


def pick_device(name: str = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA, then MPS, then CPU; pass other names through."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
