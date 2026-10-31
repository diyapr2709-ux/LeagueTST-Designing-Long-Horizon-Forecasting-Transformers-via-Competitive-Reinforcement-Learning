"""The independent evaluator used by scripts/revalidate.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from leaguetst.analysis import (
    block_bootstrap_ci,
    collect_forecasts,
    error_profile,
    naive_forecasts,
    window_targets,
)
from leaguetst.data import WindowDataset, build_dataloaders
from leaguetst.models import DLinear
from leaguetst.training import ErrorAccumulator, Trainer, TrainingConfig


def test_error_profile_matches_the_training_accumulator():
    rng = np.random.default_rng(0)
    forecasts, targets = rng.normal(size=(37, 5, 3)), rng.normal(size=(37, 5, 3))
    accumulator = ErrorAccumulator()
    for start in range(0, 37, 8):
        accumulator.update(torch.tensor(forecasts[start : start + 8]), torch.tensor(targets[start : start + 8]))
    expected = accumulator.summary(batch_size=8)
    profile = error_profile(forecasts, targets, batch_size=8)
    for key in ("mse", "mae", "mse_drop_last", "mae_drop_last"):
        assert profile[key] == pytest.approx(expected[key], rel=1e-12)
    assert profile["n_keep"] == expected["n_samples_drop_last"] == 32
    dropped = ((forecasts[32:] - targets[32:]) ** 2).mean()
    assert profile["mse_dropped"] == pytest.approx(dropped)
    assert np.mean(profile["per_step_mse"]) == pytest.approx(profile["mse"])
    assert np.mean(profile["per_variable_mse"]) == pytest.approx(profile["mse"])


def test_collect_forecasts_agrees_with_the_trainer(synthetic_data_dir):
    loaders, _ = build_dataloaders(synthetic_data_dir, "ILI", 104, 24, batch_size=16)
    model = DLinear(n_vars=7, input_len=104, horizon=24)
    recorded = Trainer(model, loaders, TrainingConfig(epochs=1, batch_size=16), torch.device("cpu")).evaluate("test")
    forecasts, targets = collect_forecasts(model, loaders["test"].dataset, batch_size=100)
    assert np.array_equal(targets, window_targets(loaders["test"].dataset))
    profile = error_profile(forecasts, targets, batch_size=16)
    assert profile["mse"] == pytest.approx(recorded["mse"], rel=1e-6)
    assert profile["mse_drop_last"] == pytest.approx(recorded["mse_drop_last"], rel=1e-6)


def test_naive_forecasts_on_a_known_series():
    series = np.arange(30, dtype=np.float64).reshape(15, 2)
    dataset = WindowDataset(series, input_len=6, horizon=4)
    last = naive_forecasts(dataset, "last")
    assert last.shape == (len(dataset), 4, 2)
    assert np.all(last[0] == series[5])
    seasonal = naive_forecasts(dataset, "seasonal", season=3)
    assert np.array_equal(seasonal[0], series[[3, 4, 5, 3]])
    with pytest.raises(ValueError):
        naive_forecasts(dataset, "seasonal", season=7)


def test_block_bootstrap_interval_is_deterministic_and_widens_with_dependence():
    rng = np.random.default_rng(1)
    independent = rng.normal(size=2000)
    low, high = block_bootstrap_ci(independent, block=1, seed=3)
    assert low < independent.mean() < high
    assert block_bootstrap_ci(independent, block=1, seed=3) == (low, high)
    dependent = np.repeat(rng.normal(size=100), 20)
    narrow = np.subtract(*block_bootstrap_ci(dependent, block=1)[::-1])
    wide = np.subtract(*block_bootstrap_ci(dependent, block=20)[::-1])
    assert wide > 2 * narrow
