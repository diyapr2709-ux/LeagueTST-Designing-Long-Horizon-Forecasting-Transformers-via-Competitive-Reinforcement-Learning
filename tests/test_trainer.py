"""The training loop, schedule and end-to-end table runs on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from leaguetst.data.device_loader import build_trainval_loaders
from leaguetst.ledger import row_from_record
from leaguetst.models import ForecasterConfig, PatchTST
from leaguetst.table import recipe, run_entries
from leaguetst.training import ErrorAccumulator, Trainer, TrainingConfig, learning_rate_at
from leaguetst.utils import sha256_file

TINY = {
    "patch_len": 8, "stride": 4, "n_layers": 1, "n_heads": 2, "d_model": 16,
    "ff_ratio": 1, "dropout": 0.0, "instance_norm": True, "channel_mode": "independent",
}


def train_tiny(trainval_dir, out, seed=3, **training):
    """Train TINY on synthetic ILI at T=24 through the table path; return its record."""
    job = {"item": "T", "dataset": "ILI", "horizon": 24, "seed": seed, "trainval_dir": str(trainval_dir),
           "training": {**recipe("ILI", epochs=2, patience=10), **training}, "git_hash": "test"}
    assert run_entries(job, [("tiny", TINY)], out, "cpu") == {"done": 1, "skipped": 0, "failed": 0}
    return json.loads((out / "records" / "tiny.json").read_text())


def test_error_accumulator_matches_global_mean():
    pred = torch.randn(10, 4, 3)
    true = torch.randn(10, 4, 3)
    accumulator = ErrorAccumulator()
    for start in (0, 3, 6):
        accumulator.update(pred[start : start + 3], true[start : start + 3])
    accumulator.update(pred[9:], true[9:])
    assert accumulator.n_samples == 10
    assert accumulator.mse == pytest.approx(float((pred - true).pow(2).mean()))
    assert accumulator.mae == pytest.approx(float((pred - true).abs().mean()))

    summary = accumulator.summary(batch_size=3)
    assert summary["n_samples_drop_last"] == 9
    assert summary["mse_drop_last"] == pytest.approx(float((pred[:9] - true[:9]).pow(2).mean()))
    assert summary["mae_drop_last"] == pytest.approx(float((pred[:9] - true[:9]).abs().mean()))
    assert accumulator.summary(batch_size=10)["mse_drop_last"] == pytest.approx(accumulator.mse)


def test_learning_rate_schedules():
    assert [learning_rate_at(1.0, e, "constant") for e in range(1, 4)] == [1.0, 1.0, 1.0]
    assert [learning_rate_at(1.0, e, "type1") for e in range(1, 5)] == [1.0, 1.0, 0.5, 0.25]
    type3 = [learning_rate_at(1.0, e, "type3") for e in range(1, 7)]
    assert type3[:4] == [1.0, 1.0, 1.0, 1.0]
    assert type3[4] == pytest.approx(0.9)
    assert type3[5] == pytest.approx(0.81)
    with pytest.raises(ValueError):
        TrainingConfig(lr_schedule="cosine")
    assert learning_rate_at(1.0, 1, "type3", first_epoch_factor=0.04) == pytest.approx(0.04)
    assert learning_rate_at(1.0, 2, "constant", first_epoch_factor=0.04) == 1.0


def test_official_onecycle_side_effects_are_reproduced():
    from torch.optim import lr_scheduler

    for lr in (1e-4, 2.5e-3):
        reference = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)
        lr_scheduler.OneCycleLR(optimizer=reference, steps_per_epoch=10, epochs=100, max_lr=lr)
        group = reference.param_groups[0]
        config = TrainingConfig(learning_rate=lr, adam_beta1=0.95, first_epoch_lr_factor=0.04)
        assert group["betas"][0] == config.adam_beta1
        assert group["lr"] == pytest.approx(learning_rate_at(lr, 1, "type3", config.first_epoch_lr_factor))


def test_table_runs_are_reproducible_for_a_seed(trainval_dir, tmp_path):
    first = train_tiny(trainval_dir, tmp_path / "a")
    again = train_tiny(trainval_dir, tmp_path / "b")
    assert again["best_val_mse"] == pytest.approx(first["best_val_mse"])
    assert np.allclose(np.load(tmp_path / "a" / "perwindow" / "tiny.npy"), np.load(tmp_path / "b" / "perwindow" / "tiny.npy"))
    other = train_tiny(trainval_dir, tmp_path / "c", seed=4)
    assert other["best_val_mse"] != pytest.approx(first["best_val_mse"])


def test_table_records_carry_file_digests(trainval_dir, tmp_path):
    record = train_tiny(trainval_dir, tmp_path)
    assert record["checkpoint_sha256"] == sha256_file(tmp_path / "ckpt" / "tiny.pt")
    assert record["perwindow_sha256"] == sha256_file(tmp_path / "perwindow" / "tiny.npy")


def test_divergence_is_flagged_in_the_record_and_the_ledger(trainval_dir, tmp_path):
    record = train_tiny(trainval_dir, tmp_path, epochs=3, learning_rate=1e30)
    assert record["diverged"] is True
    assert record["epochs_run"] == 1
    assert row_from_record(record, "test")["diverged"] is True


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_evaluate_runs_on_mps(trainval_dir):
    loaders, _ = build_trainval_loaders(trainval_dir, "ILI", 104, 24, 32, 0, torch.device("cpu"))
    model = PatchTST(ForecasterConfig.from_dict(TINY), n_vars=7, input_len=104, horizon=24)
    trainer = Trainer(model, loaders, TrainingConfig(epochs=1, batch_size=32), torch.device("mps"))
    on_mps = trainer.evaluate("val")
    cpu_trainer = Trainer(model.cpu(), loaders, TrainingConfig(epochs=1, batch_size=32), torch.device("cpu"))
    on_cpu = cpu_trainer.evaluate("val")
    assert on_mps["n_samples"] == on_cpu["n_samples"] == 97 - 24 + 1
    assert on_mps["mse"] == pytest.approx(on_cpu["mse"], rel=1e-4)
