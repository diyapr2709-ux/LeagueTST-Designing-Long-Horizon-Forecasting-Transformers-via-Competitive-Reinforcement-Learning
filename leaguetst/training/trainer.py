"""Supervised training of a forecaster with early stopping on validation MSE.

The loop mirrors the official PatchTST experiment: Adam, mean squared error,
a per-epoch learning-rate schedule, and the checkpoint with the lowest
validation error is restored at the end. Validation and test errors are exact
means over every window of the split.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from leaguetst.training.metrics import ErrorAccumulator
from leaguetst.utils.checks import require_int, require_real

log = logging.getLogger(__name__)

SCHEDULES = ("constant", "type1", "type3")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser and schedule settings.

    ``lr_schedule`` names the official PatchTST schedules: ``"type1"`` halves
    the rate every epoch from the third, ``"type3"`` multiplies it by 0.9 every
    epoch from the fifth, ``"constant"`` leaves it alone.

    ``adam_beta1`` and ``first_epoch_lr_factor`` reproduce a side effect of
    the official PatchTST trainer. It builds a OneCycleLR scheduler before
    training even when the schedule is not ``TST``; building it sets Adam's
    first beta to 0.95 for the whole run and the first epoch's rate to
    ``learning_rate / 25``. The defaults (0.9 and 1.0) are plain Adam with
    no first-epoch change.
    """

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    patience: int = 100
    lr_schedule: str = "type3"
    adam_beta1: float = 0.9
    first_epoch_lr_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            object.__setattr__(self, name, require_int(name, getattr(self, name), 1))
        lr = require_real("learning_rate", self.learning_rate, 0.0)
        if lr == 0.0:
            raise ValueError("learning_rate must be positive")
        object.__setattr__(self, "learning_rate", lr)
        object.__setattr__(self, "adam_beta1", require_real("adam_beta1", self.adam_beta1, 0.0, 1.0, high_open=True))
        factor = require_real("first_epoch_lr_factor", self.first_epoch_lr_factor, 0.0, 1.0)
        if factor == 0.0:
            raise ValueError("first_epoch_lr_factor must be positive")
        object.__setattr__(self, "first_epoch_lr_factor", factor)
        if not isinstance(self.lr_schedule, str) or self.lr_schedule not in SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {SCHEDULES}")


@dataclass
class FitResult:
    """What a training run produced.

    ``diverged`` is set when the training loss or the validation error stops
    being finite. The restored weights are then those of the best finite
    epoch, or the initial weights if there was none (``best_epoch`` 0).
    """

    best_val_mse: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)
    diverged: bool = False


def learning_rate_at(base: float, epoch: int, schedule: str, first_epoch_factor: float = 1.0) -> float:
    """Learning rate for a 1-indexed epoch.

    The official code adjusts the rate after each epoch, so the first decayed
    value takes effect one epoch later than the schedule's name suggests.
    ``first_epoch_factor`` scales epoch 1 only; see :class:`TrainingConfig`.
    """
    if epoch == 1:
        return base * first_epoch_factor
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, 0.999))

        best_val = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_gain = 0
        diverged = False
        history: list[dict[str, float]] = []
        start = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            lr = learning_rate_at(cfg.learning_rate, epoch, cfg.lr_schedule, cfg.first_epoch_lr_factor)
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

            if not (math.isfinite(train_loss) and math.isfinite(val_mse)):
                log.warning("non-finite loss at epoch %d; training stopped", epoch)
                diverged = True
                break

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
            diverged=diverged,
        )
