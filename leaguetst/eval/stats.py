"""Seed-level and window-level comparisons of two arms, and Holm correction."""

from __future__ import annotations

import itertools
import math

import numpy as np

EXACT_LIMIT = 250_000


def _checked(values, name: str) -> np.ndarray:
    """A 1-D float array with at least two finite values, or ValueError."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if len(array) < 2:
        raise ValueError(f"{name} needs at least two values, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} holds non-finite values")
    return array


def permutation_test(a: np.ndarray, b: np.ndarray, n_permutations: int = 100_000, seed: int = 0) -> dict[str, float]:
    """Two-sided test of mean(a) = mean(b) for independent per-seed values.

    Exact over every relabelling when there are at most ``EXACT_LIMIT`` of
    them (10 seeds against 10 gives 184,756); otherwise Monte Carlo with
    the observed labelling counted, so p is never zero.
    """
    a = _checked(a, "a")
    b = _checked(b, "b")
    pooled = np.concatenate([a, b])
    n, k = len(pooled), len(a)
    observed = a.mean() - b.mean()
    total = pooled.sum()
    tolerance = 1e-12 * max(1.0, abs(observed))
    if math.comb(n, k) <= EXACT_LIMIT:
        extreme = count = 0
        for subset in itertools.combinations(range(n), k):
            sum_a = pooled[list(subset)].sum()
            diff = sum_a / k - (total - sum_a) / (n - k)
            extreme += abs(diff) >= abs(observed) - tolerance
            count += 1
        p = extreme / count
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            diff = pooled[perm[:k]].mean() - pooled[perm[k:]].mean()
            extreme += abs(diff) >= abs(observed) - tolerance
        p = (extreme + 1) / (n_permutations + 1)
    return {"diff": float(observed), "p": float(p), "n_a": k, "n_b": n - k}


def _block_index(rng: np.random.Generator, n_windows: int, block: int) -> np.ndarray:
    block = max(1, min(int(block), n_windows))
    n_blocks = math.ceil(n_windows / block)
    starts = rng.integers(0, n_windows, n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n_windows).ravel()[:n_windows]


def endpoint_bootstrap(
    a: dict[int, np.ndarray],
    b: dict[int, np.ndarray],
    blocks: dict[int, int],
    n_resamples: int = 4000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Interval for the difference in the mean over horizons of per-window MSE.

    ``a[h]`` and ``b[h]`` are (seeds, windows) arrays at horizon h whose rows
    are in the same seed order at every horizon. Each resample draws seed
    rows per arm (shared across horizons) and, per horizon, one circular
    block resample of windows shared by both arms, so the pairing by window
    is kept and seed variance is included.
    """
    horizons = sorted(a)
    if not horizons or sorted(b) != horizons:
        raise ValueError("arms must cover the same, non-empty set of horizons")
    for arm in (a, b):
        for h in horizons:
            values = np.asarray(arm[h], dtype=np.float64)
            if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"horizon {h}: need a finite (seeds >= 2, windows) array")
    n_a = {a[h].shape[0] for h in horizons}
    n_b = {b[h].shape[0] for h in horizons}
    if len(n_a) != 1 or len(n_b) != 1:
        raise ValueError("each arm needs the same seeds at every horizon")
    n_a, n_b = n_a.pop(), n_b.pop()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        sa = rng.integers(0, n_a, n_a)
        sb = rng.integers(0, n_b, n_b)
        total = 0.0
        for h in horizons:
            index = _block_index(rng, a[h].shape[1], blocks[h])
            total += a[h][sa][:, index].mean() - b[h][sb][:, index].mean()
        diffs[r] = total / len(horizons)
    observed = float(np.mean([a[h].mean() - b[h].mean() for h in horizons]))
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(diffs, [tail, 1.0 - tail])
    below = int((diffs <= 0).sum())
    above = int((diffs >= 0).sum())
    p = min(1.0, 2.0 * (min(below, above) + 1) / (n_resamples + 1))
    return {"diff": observed, "low": float(low), "high": float(high), "p": float(p)}


def paired_two_level_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    block: int,
    n_resamples: int = 4000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """One-horizon case of :func:`endpoint_bootstrap` for (seeds, windows) arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must be (seeds, windows) with the same windows")
    return endpoint_bootstrap({0: a}, {0: b}, {0: block}, n_resamples, level, seed)


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
