"""Batches gathered on the GPU from one resident copy of the series.

The standard loader slices every window on the CPU and collates it, which
costs more than the forward pass of a small forecaster. Here the scaled
series sits on the device once and each batch is gathered with one indexing
operation. The order of windows comes from a ``DataLoader`` over the window
indices with the same sampler, generator and flags as
:func:`leaguetst.data.build_dataloaders`, so the batches are the same
windows in the same order.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from leaguetst.data.datasets import Standardizer, WindowDataset
from leaguetst.data.trainval import build_trainval_splits


class DeviceWindowLoader:
    """Iterable of (inputs, targets) batches already on ``device``."""

    def __init__(
        self,
        dataset: WindowDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: torch.device,
        generator: torch.Generator | None = None,
    ):
        self.dataset = dataset
        self.device = device
        self.values = dataset.values.to(device)
        self.index_loader = DataLoader(
            range(len(dataset)),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )
        self.input_offsets = torch.arange(dataset.input_len, device=device)
        self.target_offsets = torch.arange(dataset.input_len, dataset.input_len + dataset.horizon, device=device)

    def __len__(self) -> int:
        return len(self.index_loader)

    def __iter__(self):
        for index in self.index_loader:
            start = index.to(self.device)[:, None]
            yield self.values[start + self.input_offsets], self.values[start + self.target_offsets]


def build_trainval_loaders(
    trainval_dir: str | Path,
    name: str,
    input_len: int,
    horizon: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, DeviceWindowLoader], Standardizer]:
    """Shuffled drop-last training batches and ordered, complete validation batches."""
    datasets, standardizer = build_trainval_splits(trainval_dir, name, input_len, horizon)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DeviceWindowLoader(datasets["train"], batch_size, True, True, device, generator),
        "val": DeviceWindowLoader(datasets["val"], batch_size, False, False, device),
    }
    return loaders, standardizer
