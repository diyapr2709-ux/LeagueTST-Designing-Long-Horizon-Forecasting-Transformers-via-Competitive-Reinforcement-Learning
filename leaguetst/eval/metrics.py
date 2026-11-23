"""Exact and drop-last means from per-window errors."""

from __future__ import annotations

import numpy as np


def exact_and_drop_last(forecasts: np.ndarray, targets: np.ndarray, batch_size: int) -> dict[str, float | int]:
    """MSE and MAE over every window, and over the full batches only.

    ``forecasts`` and ``targets`` have shape (windows, horizon, variables).
    The drop-last values keep the first ``floor(n / batch_size) * batch_size``
    windows, which is what a test loader with ``drop_last=True`` scores.
    """
    if forecasts.shape != targets.shape:
        raise ValueError(f"shapes differ: {forecasts.shape} vs {targets.shape}")
    error = forecasts.astype(np.float64) - targets.astype(np.float64)
    per_window_sq = (error**2).reshape(len(error), -1).mean(axis=1)
    per_window_abs = np.abs(error).reshape(len(error), -1).mean(axis=1)
    n = len(error)
    keep = (n // batch_size) * batch_size
    return {
        "mse": float(per_window_sq.mean()),
        "mae": float(per_window_abs.mean()),
        "n_samples": n,
        "mse_drop_last": float(per_window_sq[:keep].mean()) if keep else float("nan"),
        "mae_drop_last": float(per_window_abs[:keep].mean()) if keep else float("nan"),
        "n_samples_drop_last": keep,
    }
