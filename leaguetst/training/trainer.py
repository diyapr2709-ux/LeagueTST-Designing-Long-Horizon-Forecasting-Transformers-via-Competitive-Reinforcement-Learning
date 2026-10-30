"""Supervised training of a forecaster with early stopping on validation MSE.

The loop mirrors the official PatchTST experiment: Adam, mean squared error,
a per-epoch learning-rate schedule, and the checkpoint with the lowest
validation error is restored at the end. Validation and test errors are exact
means over every window of the split.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from leaguetst.training.metrics import ErrorAccumulator

log = logging.getLogger(__name__)

SCHEDULES = ("constant", "type1", "type3")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser and schedule settings.

    ``lr_schedule`` names the official PatchTST schedules: ``"type1"`` halves
    the rate every epoch from the third, ``"type3"`` multiplies it by 0.9 every
    epoch from the fifth, ``"constant"`` leaves it alone.
    """

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    patience: int = 100
    lr_schedule: str = "type3"

    def __post_init__(self) -> None:
        if self.lr_schedule not in SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {SCHEDULES}")
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size and patience must be positive")


@dataclass
class FitResult:
    """What a training run produced."""

    best_val_mse: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def learning_rate_at(base: float, epoch: int, schedule: str) -> float:
    """Learning rate for a 1-indexed epoch.

    The official code adjusts the rate after each epoch, so the first decayed
    value takes effect one epoch later than the schedule's name suggests.
    """
    if schedule == "constant":
        return base
    if schedule == "type1":
        return base * 0.5 ** max(0, epoch - 2)
    if schedule == "type3":
        return base * 0.9 ** max(0, epoch - 4)
    raise ValueError(f"unknown schedule {schedule!r}")


class Trainer:
    """Fit a model on the training loader and evaluate on any loader."""

    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total = 0.0
        n_batches = 0
        for inputs, targets in self.loaders["train"]:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float | int]:
        """Exact MSE and MAE over every window of ``split``.

        The summary also carries the official-style values that drop the
        last partial batch, for comparison with published numbers.
        """
        self.model.eval()
        accumulator = ErrorAccumulator()
        for inputs, targets in self.loaders[split]:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            accumulator.update(self.model(inputs), targets)
        return accumulator.summary(batch_size=self.config.batch_size)

    def fit(self) -> FitResult:
        """Train until the epoch budget or the patience runs out."""
        cfg = self.config
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        best_val = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_gain = 0
        history: list[dict[str, float]] = []
        start = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            lr = learning_rate_at(cfg.learning_rate, epoch, cfg.lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr

            epoch_start = time.perf_counter()
            train_loss = self._train_epoch(optimizer)
            val_mse = float(self.evaluate("val")["mse"])
            history.append(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_loss,
                    "val_mse": val_mse,
                    "seconds": time.perf_counter() - epoch_start,
                }
            )
            log.info(
                "epoch %3d  lr %.2e  train %.4f  val %.4f  (%.1fs)",
                epoch, lr, train_loss, val_mse, history[-1]["seconds"],
            )

            if val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= cfg.patience:
                    log.info("early stop at epoch %d", epoch)
                    break

        self.model.load_state_dict(best_state)
        return FitResult(
            best_val_mse=best_val,
            best_epoch=best_epoch,
            epochs_run=len(history),
            train_seconds=time.perf_counter() - start,
            history=history,
        )
