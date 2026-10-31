"""The training loop, schedule and end-to-end run on synthetic data."""

from __future__ import annotations

import json

import pytest
import torch

from leaguetst.data import build_dataloaders
from leaguetst.experiment import RunSpec, run_experiment
from leaguetst.models import DLinear
from leaguetst.training import ErrorAccumulator, Trainer, TrainingConfig, learning_rate_at


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


@pytest.mark.parametrize("model", ["patchtst", "dlinear"])
def test_run_experiment_end_to_end(synthetic_data_dir, tmp_path, model):
    model_config = {} if model == "dlinear" else {
        "patch_len": 8, "stride": 4, "n_layers": 1, "n_heads": 2, "d_model": 16,
        "ff_ratio": 1, "dropout": 0.0, "instance_norm": True, "channel_mode": "independent",
    }
    spec = RunSpec(
        model=model,
        dataset="ILI",
        horizon=24,
        seed=1,
        model_config=model_config,
        training=TrainingConfig(epochs=2, batch_size=32, learning_rate=1e-3, patience=5),
        device="cpu",
        data_dir=str(synthetic_data_dir),
        out_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    record = run_experiment(spec)
    assert record["fit"]["epochs_run"] == 2
    assert record["test"]["n_samples"] == 193 - 24 + 1
    assert record["test"]["n_samples_drop_last"] == ((193 - 24 + 1) // 32) * 32
    assert record["val"]["n_samples"] == 97 - 24 + 1
    assert (tmp_path / "ckpt" / f"{spec.run_name}.pt").exists()
    saved = json.loads((tmp_path / f"{spec.run_name}.json").read_text())
    assert saved["test"]["mse"] == pytest.approx(record["test"]["mse"])
    assert saved["n_params"] == record["n_params"]


def test_runs_are_reproducible_for_a_seed(synthetic_data_dir, tmp_path):
    def run(seed):
        spec = RunSpec(
            model="dlinear", dataset="ILI", horizon=24, seed=seed,
            training=TrainingConfig(epochs=1, batch_size=32, learning_rate=1e-3, patience=5),
            device="cpu", data_dir=str(synthetic_data_dir), out_dir=str(tmp_path),
            checkpoint_dir="",
        )
        return run_experiment(spec)["test"]["mse"]

    assert run(3) == pytest.approx(run(3))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_evaluate_runs_on_mps(synthetic_data_dir):
    loaders, _ = build_dataloaders(synthetic_data_dir, "ILI", 104, 24, batch_size=32)
    model = DLinear(n_vars=7, input_len=104, horizon=24)
    trainer = Trainer(model, loaders, TrainingConfig(epochs=1, batch_size=32), torch.device("mps"))
    on_mps = trainer.evaluate("test")
    cpu_trainer = Trainer(model.cpu(), loaders, TrainingConfig(epochs=1, batch_size=32), torch.device("cpu"))
    on_cpu = cpu_trainer.evaluate("test")
    assert on_mps["n_samples"] == on_cpu["n_samples"] == 193 - 24 + 1
    assert on_mps["mse"] == pytest.approx(on_cpu["mse"], rel=1e-4)
    assert on_mps["mse_drop_last"] == pytest.approx(on_cpu["mse_drop_last"], rel=1e-4)
