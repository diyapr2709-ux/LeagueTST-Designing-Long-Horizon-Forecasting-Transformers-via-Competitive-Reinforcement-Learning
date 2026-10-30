"""DataLoader construction for the benchmark splits."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from leaguetst.data.datasets import Standardizer, build_splits


def build_dataloaders(
    data_dir: str | Path,
    name: str,
    input_len: int,
    horizon: int,
    batch_size: int,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], Standardizer]:
    """Return shuffled training and ordered validation and test loaders.

    The training loader drops its last partial batch, as the official code
    does. The validation and test loaders keep every window so that the
    reported errors cover the whole split.
    """
    datasets, standardizer = build_splits(data_dir, name, input_len, horizon)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, drop_last=False
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, drop_last=False
        ),
    }
    return loaders, standardizer
