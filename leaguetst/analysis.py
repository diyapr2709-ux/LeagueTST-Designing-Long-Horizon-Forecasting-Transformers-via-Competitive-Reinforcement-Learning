"""Independent re-evaluation of saved runs and detailed error statistics.

The training loop scores a model with a running accumulator on the training
device. This module scores a saved checkpoint again on the CPU: it collects
every forecast in a float64 NumPy array and computes each statistic from
that array directly, so an error in either path shows up as a disagreement.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from leaguetst.data import WindowDataset

# Dominant period of each benchmark in steps: a day of hours, a year of weeks.
SEASON_LENGTH = {"ETTh1": 24, "ILI": 52}


@torch.no_grad()
def collect_forecasts(
    model: nn.Module,
    dataset: WindowDataset,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return forecasts and targets of every window, each (windows, horizon, vars)."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    forecasts, targets = [], []
    for inputs, target in loader:
        forecasts.append(model(inputs.to(device)).cpu().double().numpy())
        targets.append(target.double().numpy())
    return np.concatenate(forecasts), np.concatenate(targets)


def window_targets(dataset: WindowDataset) -> np.ndarray:
    """Targets of every window, (windows, horizon, vars), straight from the series."""
    values = dataset.values.double().numpy()
    rows = np.arange(len(dataset))[:, None] + dataset.input_len + np.arange(dataset.horizon)[None, :]
    return values[rows]


def naive_forecasts(dataset: WindowDataset, kind: str, season: int | None = None) -> np.ndarray:
    """Forecasts that need no training, to put the learned errors on a scale.

    ``"last"`` repeats the final input step over the whole horizon.
    ``"seasonal"`` repeats the final ``season`` input steps, cycling as often
    as the horizon needs.
    """
    values = dataset.values.double().numpy()
    starts = np.arange(len(dataset))[:, None]
    steps = np.arange(dataset.horizon)[None, :]
    if kind == "last":
        rows = starts + dataset.input_len - 1 + 0 * steps
    elif kind == "seasonal":
        if season is None or not 0 < season <= dataset.input_len:
            raise ValueError("seasonal forecasts need 0 < season <= input_len")
        rows = starts + dataset.input_len - season + steps % season
    else:
        raise ValueError(f"unknown naive forecast {kind!r}")
    return values[rows]


def error_profile(forecasts: np.ndarray, targets: np.ndarray, batch_size: int | None = None) -> dict[str, Any]:
    """Exact and drop-last errors plus their breakdown by window, step and variable.

    ``batch_size`` sets the drop-last protocol: only the first
    ``floor(n / batch_size) * batch_size`` windows count, as in a test loader
    that drops its last partial batch. ``mse_dropped`` is the mean over the
    windows that protocol leaves out.
    """
    if forecasts.shape != targets.shape or forecasts.ndim != 3:
        raise ValueError("forecasts and targets must share a (windows, horizon, vars) shape")
    error = forecasts - targets
    squared = error**2
    absolute = np.abs(error)
    per_window_mse = squared.mean(axis=(1, 2))
    per_window_mae = absolute.mean(axis=(1, 2))
    n = len(per_window_mse)
    n_keep = (n // batch_size) * batch_size if batch_size else n
    profile: dict[str, Any] = {
        "n_windows": n,
        "mse": float(per_window_mse.mean()),
        "mae": float(per_window_mae.mean()),
        "n_keep": n_keep,
        "mse_drop_last": float(per_window_mse[:n_keep].mean()) if n_keep else math.nan,
        "mae_drop_last": float(per_window_mae[:n_keep].mean()) if n_keep else math.nan,
        "mse_dropped": float(per_window_mse[n_keep:].mean()) if n_keep < n else math.nan,
        "per_step_mse": squared.mean(axis=(0, 2)).tolist(),
        "per_variable_mse": squared.mean(axis=(0, 1)).tolist(),
        "per_window_mse": per_window_mse,
    }
    return profile


def block_bootstrap_ci(
    values: np.ndarray,
    block: int,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for the mean of a serially dependent sequence.

    Circular block bootstrap: each resample joins randomly placed runs of
    ``block`` consecutive values, so the overlap between neighbouring
    forecast windows stays inside each run instead of being averaged away.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("no values to resample")
    block = max(1, min(int(block), n))
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = values[index.reshape(n_resamples, -1)[:, :n]].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)
