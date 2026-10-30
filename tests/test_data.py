"""Split borders, scaling and windowing of the data pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from leaguetst.data import (
    BENCHMARKS,
    Standardizer,
    WindowDataset,
    build_dataloaders,
    build_splits,
    load_benchmark_frame,
    split_borders,
)


def test_ett_borders_match_official_protocol():
    borders = split_borders(17420, input_len=336, scheme="ett")
    assert borders["train"] == (0, 8640)
    assert borders["val"] == (8640 - 336, 11520)
    assert borders["test"] == (11520 - 336, 14400)


def test_ratio_borders_match_official_protocol():
    borders = split_borders(966, input_len=104, scheme="ratio")
    n_train, n_test = int(966 * 0.7), int(966 * 0.2)
    assert (n_train, n_test) == (676, 193)
    assert borders["train"] == (0, 676)
    assert borders["val"] == (676 - 104, 966 - 193)
    assert borders["test"] == (966 - 193 - 104, 966)


def test_borders_reject_short_files():
    with pytest.raises(ValueError):
        split_borders(1000, input_len=336, scheme="ett")
    with pytest.raises(ValueError):
        split_borders(100, input_len=336, scheme="ratio")
    with pytest.raises(ValueError):
        split_borders(17420, input_len=336, scheme="monthly")


def test_standardizer_uses_population_std_and_guards_zero_variance():
    values = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    scaler = Standardizer.fit(values)
    np.testing.assert_allclose(scaler.mean, [3.0, 5.0])
    np.testing.assert_allclose(scaler.scale, [np.sqrt(8 / 3), 1.0])
    scaled = scaler.transform(values)
    np.testing.assert_allclose(scaled.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(scaler.inverse(scaled), values)


def test_window_dataset_slices_contiguous_rows():
    values = np.arange(20, dtype=np.float64).reshape(10, 2)
    dataset = WindowDataset(values, input_len=3, horizon=2)
    assert len(dataset) == 10 - 3 - 2 + 1
    x, y = dataset[4]
    assert x.shape == (3, 2) and y.shape == (2, 2)
    assert torch.equal(x, torch.tensor(values[4:7], dtype=torch.float32))
    assert torch.equal(y, torch.tensor(values[7:9], dtype=torch.float32))
    with pytest.raises(ValueError):
        WindowDataset(values, input_len=9, horizon=2)


def test_frame_moves_target_last(synthetic_data_dir):
    frame = load_benchmark_frame(synthetic_data_dir, "ILI")
    assert list(frame.columns)[-1] == "OT"
    assert "date" not in frame.columns
    assert frame.shape == (966, 7)


@pytest.mark.parametrize("name,horizon", [("ETTh1", 96), ("ILI", 24)])
def test_split_window_counts(synthetic_data_dir, name, horizon):
    spec = BENCHMARKS[name]
    datasets, _ = build_splits(synthetic_data_dir, name, spec.input_len, horizon)
    n_rows = 17420 if name == "ETTh1" else 966
    borders = split_borders(n_rows, spec.input_len, spec.split_scheme)
    for split, (start, end) in borders.items():
        assert len(datasets[split]) == end - start - spec.input_len - horizon + 1
    assert datasets["train"].n_vars == 7


def test_scaling_is_fit_on_training_block_only(synthetic_data_dir):
    datasets, scaler = build_splits(synthetic_data_dir, "ETTh1", 336, 96)
    train_values = datasets["train"].values.numpy()
    np.testing.assert_allclose(train_values.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(train_values.std(axis=0), 1.0, atol=1e-4)
    raw = load_benchmark_frame(synthetic_data_dir, "ETTh1").to_numpy()
    np.testing.assert_allclose(scaler.mean, raw[:8640].mean(axis=0))


def test_val_and_test_loaders_keep_every_window(synthetic_data_dir):
    loaders, _ = build_dataloaders(synthetic_data_dir, "ILI", 104, 24, batch_size=16)
    for split in ("val", "test"):
        seen = sum(x.shape[0] for x, _ in loaders[split])
        assert seen == len(loaders[split].dataset)
    train_seen = sum(x.shape[0] for x, _ in loaders["train"])
    assert train_seen == (len(loaders["train"].dataset) // 16) * 16


def test_real_files_have_expected_shape(real_data_dir):
    assert load_benchmark_frame(real_data_dir, "ETTh1").shape == (17420, 7)
    assert load_benchmark_frame(real_data_dir, "ILI").shape == (966, 7)
