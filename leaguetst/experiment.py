"""One complete training run: data, model, fit, evaluate, save.

A run is described by a :class:`RunSpec`, usually filled from a YAML file in
``configs/`` plus command-line overrides. The result is a plain dictionary
that is also written as JSON so that ``scripts/summarize_results.py`` can
aggregate runs across seeds.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from leaguetst.data import BENCHMARKS, build_dataloaders
from leaguetst.models import DLinear, ForecasterConfig, PatchTST, count_parameters
from leaguetst.training import Trainer, TrainingConfig
from leaguetst.utils import pick_device, save_json, set_seed

log = logging.getLogger(__name__)

MODEL_NAMES = ("patchtst", "dlinear")


@dataclass
class RunSpec:
    """Everything needed to reproduce one training run."""

    model: str
    dataset: str
    horizon: int
    seed: int = 2021
    input_len: int | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"
    data_dir: str = "data"
    out_dir: str = "results/preliminary"
    checkpoint_dir: str = "checkpoints"
    tag: str = ""

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}")
        if self.dataset not in BENCHMARKS:
            raise ValueError(f"dataset must be one of {tuple(BENCHMARKS)}")
        if self.input_len is None:
            self.input_len = BENCHMARKS[self.dataset].input_len
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

    @classmethod
    def from_yaml_dict(cls, loaded: dict[str, Any], **overrides: Any) -> "RunSpec":
        """Build a spec from a loaded YAML mapping and keyword overrides.

        Overrides whose value is ``None`` are ignored so that command-line
        flags left at their default do not clobber the file.
        """
        values = dict(loaded)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        return cls(**values)

    @property
    def run_name(self) -> str:
        base = f"{self.model}_{self.dataset}_h{self.horizon}_s{self.seed}"
        return f"{base}_{self.tag}" if self.tag else base


def build_model(
    name: str,
    model_config: dict[str, Any],
    n_vars: int,
    input_len: int,
    horizon: int,
) -> nn.Module:
    """Instantiate a forecaster by name."""
    if name == "patchtst":
        config = ForecasterConfig.from_dict(model_config)
        return PatchTST(config, n_vars=n_vars, input_len=input_len, horizon=horizon)
    if name == "dlinear":
        kernel_size = int(model_config.get("kernel_size", 25))
        return DLinear(n_vars=n_vars, input_len=input_len, horizon=horizon, kernel_size=kernel_size)
    raise ValueError(f"unknown model {name!r}")


def run_experiment(spec: RunSpec) -> dict[str, Any]:
    """Train and evaluate one model; return and save the record."""
    set_seed(spec.seed)
    device = pick_device(spec.device)
    log.info("run %s on %s", spec.run_name, device)

    loaders, _ = build_dataloaders(
        spec.data_dir,
        spec.dataset,
        input_len=spec.input_len,
        horizon=spec.horizon,
        batch_size=spec.training.batch_size,
        seed=spec.seed,
    )
    n_vars = loaders["train"].dataset.n_vars
    model = build_model(spec.model, spec.model_config, n_vars, spec.input_len, spec.horizon)
    n_params = count_parameters(model)
    log.info("%s has %d parameters", spec.model, n_params)

    trainer = Trainer(model, loaders, spec.training, device)
    fit = trainer.fit()
    val = trainer.evaluate("val")
    test = trainer.evaluate("test")

    checkpoint_path = ""
    if spec.checkpoint_dir:
        path = Path(spec.checkpoint_dir) / f"{spec.run_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        checkpoint_path = str(path)
    log.info(
        "done %s  val mse %.4f  test mse %.4f  mae %.4f  (%d params, %.0fs)",
        spec.run_name, val["mse"], test["mse"], test["mae"], n_params, fit.train_seconds,
    )

    record = {
        "model": spec.model,
        "dataset": spec.dataset,
        "horizon": spec.horizon,
        "input_len": spec.input_len,
        "seed": spec.seed,
        "tag": spec.tag,
        "model_config": spec.model_config,
        "training": asdict(spec.training),
        "n_params": n_params,
        "fit": fit.to_dict(),
        "val": val,
        "test": test,
        "checkpoint": checkpoint_path,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    save_json(record, Path(spec.out_dir) / f"{spec.run_name}.json")
    return record
