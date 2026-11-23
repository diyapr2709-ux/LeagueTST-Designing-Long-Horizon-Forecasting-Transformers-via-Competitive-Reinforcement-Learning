"""Paired comparisons over seeds and forecast windows, and Holm correction."""

from __future__ import annotations

import math

import numpy as np


def paired_two_level_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    block: int,
    n_resamples: int = 4000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Interval for mean(a) - mean(b) when both are (seeds, windows) arrays.

    Seeds of each arm are resampled with replacement, then a circular block
    bootstrap over windows uses the same window blocks for both arms, so the
    pairing by window is kept and seed variance is not ignored.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must be (seeds, windows) with the same windows")
    rng = np.random.default_rng(seed)
    n_windows = a.shape[1]
    block = max(1, min(int(block), n_windows))
    n_blocks = math.ceil(n_windows / block)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        sa = rng.integers(0, a.shape[0], a.shape[0])
        sb = rng.integers(0, b.shape[0], b.shape[0])
        starts = rng.integers(0, n_windows, n_blocks)
        index = ((starts[:, None] + np.arange(block)[None, :]) % n_windows).ravel()[:n_windows]
        diffs[r] = a[sa][:, index].mean() - b[sb][:, index].mean()
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(diffs, [tail, 1.0 - tail])
    observed = a.mean() - b.mean()
    p_two_sided = 2.0 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return {"diff": float(observed), "low": float(low), "high": float(high), "p": float(min(1.0, p_two_sided))}


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise level ``alpha``."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected: dict[str, bool] = {}
    still = True
    for i, (name, p) in enumerate(ordered):
        still = still and p <= alpha / (m - i)
        rejected[name] = still
    return rejected
