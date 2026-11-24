"""Seconds per training epoch as a function of the config, and the GPU bill.

An epoch is ``n_train // B`` training steps plus ``ceil(n_val / B)``
validation steps. The model is linear with non-negative coefficients in
four features of an epoch:

* ``steps``: training steps plus a third of the validation steps (a
  forward pass is about a third of a forward, backward and update), for
  the fixed cost of a step (optimizer, loss sync, embedding, head);
* ``layer_steps``: steps times encoder layers, for the kernels each layer
  launches;
* ``gflops``: matmul work, three forward passes per training window and
  one per validation window;
* ``gelems``: elementwise work (norms, activations, dropout, softmax and
  residual scores), counted the same way.

On small models the first two dominate: a step is a few hundred short
kernels. The last two take over for wide or long inputs. The model is
fitted to the steady-state epochs (2 onward) of the E1 timings in the
ledger; the first epoch's extra time and the work outside ``fit`` (model
build, checkpoint, per-window errors) are charged per config.

E1 runs four worker processes per GPU container, so the measured epoch
times include contention and a config's share of the container is its
wall time divided by four.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import nnls

from leaguetst.data.trainval import read_manifest, window_counts
from leaguetst.ledger import read_rows
from leaguetst.models.config import ForecasterConfig
from leaguetst.models.wide import HEAD_TYPES, WideConfig
from leaguetst.search.feasible import N_VARS
from leaguetst.search.space import SEARCH_SPACE
from leaguetst.search.wide_space import shape_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINVAL_DIR = REPO_ROOT / "data" / "trainval"
A10G = "NVIDIA A10G"
# NVIDIA's FP32 peak for the GA102 A10 part, about 31 TFLOPS. The table runs
# switch TF32 off (leaguetst.table.strict_float32), so this is the ceiling.
A10G_FP32_TFLOPS = 31.2
DOLLARS_PER_GPU_HOUR = 1.42
WORKERS_PER_GPU = 4
VAL_STEP_WEIGHT = 1.0 / 3.0
BACKWARD_FACTOR = 3
FEATURES = ("steps", "layer_steps", "gflops", "gelems")


def epoch_windows(dataset: str, input_len: int, horizon: int, manifest: dict[str, Any] | None = None) -> tuple[int, int]:
    """Training and validation windows at one lookback and horizon, from the manifest borders."""
    manifest = read_manifest(TRAINVAL_DIR) if manifest is None else manifest
    counts = window_counts(manifest["datasets"][dataset], input_len, horizon)
    if counts["train"] <= 0 or counts["val"] <= 0:
        raise ValueError(f"no windows for {dataset} at input {input_len}, horizon {horizon}")
    return counts["train"], counts["val"]


def window_work(
    patch_len, stride, n_layers, n_heads, d_model, ff_ratio, mixing, head_code, head_rank, input_len, horizon,
    n_vars=N_VARS,
):
    """Matmul FLOPs and elementwise elements of one forward pass over one window.

    Works on integers and on NumPy arrays alike. FLOPs count every matmul
    (patch projection, four attention projections, the two attention
    products, feed-forward, head) as two per multiply-add, the convention
    of ``torch.utils.flop_counter``. Elements are a rough count of the
    tensors written by elementwise kernels: about eight of width
    ``d_model`` and two of width ``d_ff`` per token and layer, and three
    score tensors per head and layer.
    """
    d = np.asarray(d_model, dtype=np.float64)
    d_ff = np.asarray(ff_ratio, dtype=np.float64) * d
    n_patches = (np.asarray(input_len, dtype=np.float64) - patch_len) // stride + 2
    sequences = np.where(mixing, 1, n_vars)
    width = np.where(mixing, n_vars, 1)
    out = horizon * width
    tokens = sequences * n_patches
    rank = np.asarray(head_rank, dtype=np.float64)
    code = np.asarray(head_code)
    head = np.select(
        [code == HEAD_TYPES.index(name) for name in ("flatten", "pooled", "lowrank", "shared_linear")],
        [n_patches * d * out, d * out, rank * (n_patches * d + out), n_patches * d * out],
        default=np.nan,
    )
    if np.any(np.isnan(head)):
        raise ValueError("unknown head code")
    per_layer = tokens * (4 * d * d + 2 * d * d_ff) + 2 * sequences * n_patches * n_patches * d
    macs = tokens * patch_len * width * d + n_layers * per_layer + sequences * head
    elements = n_layers * (tokens * (8 * d + 2 * d_ff) + 3 * sequences * n_heads * n_patches * n_patches)
    return 2 * macs, elements


def config_work(config: WideConfig, horizon: int, n_vars: int = N_VARS) -> tuple[float, float]:
    """``window_work`` for one config."""
    flops, elements = window_work(
        config.patch_len, config.stride, config.n_layers, config.n_heads, config.d_model, config.ff_ratio,
        config.channel_mode == "mixing", HEAD_TYPES.index(config.head_type), config.head_rank,
        config.input_len, horizon, n_vars,
    )
    return float(flops), float(elements)


def epoch_features(flops, elements, n_layers, n_train: int, n_val: int, batch_size: int) -> np.ndarray:
    """The four features of one epoch; the last axis follows ``FEATURES``."""
    train_steps = n_train // batch_size
    val_steps = math.ceil(n_val / batch_size)
    steps = train_steps + VAL_STEP_WEIGHT * val_steps
    passes = BACKWARD_FACTOR * train_steps * batch_size + n_val
    flops, elements, n_layers = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (flops, elements, n_layers)))
    return np.stack(
        [np.full_like(flops, steps), steps * n_layers, passes * flops / 1e9, passes * elements / 1e9], axis=-1
    )


def config_features(config: WideConfig, dataset: str, horizon: int, batch_size: int, manifest: dict[str, Any] | None = None) -> np.ndarray:
    flops, elements = config_work(config, horizon)
    n_train, n_val = epoch_windows(dataset, config.input_len, horizon, manifest)
    return epoch_features(flops, elements, config.n_layers, n_train, n_val, batch_size)


@dataclass
class Timing:
    """Measured timing of one ledger row."""

    config_id: str
    config: WideConfig
    dataset: str
    gpu: str
    features: np.ndarray
    epoch_seconds: float
    epochs_run: int
    overhead_seconds: float
    wall_seconds: float


def ledger_timings(path: str | Path, item: str = "E1", gpus: Iterable[str] | None = None) -> list[Timing]:
    """Timings of every ledger row of ``item`` (on the given GPUs) with at least two epochs.

    ``epoch_seconds`` is the mean of epochs 2 onward. ``overhead_seconds``
    is the rest of the row's wall time: the first epoch's excess over that
    mean plus everything outside the epoch loop.
    """
    gpus = None if gpus is None else set(gpus)
    manifest = read_manifest(TRAINVAL_DIR)
    out = []
    for row in read_rows(path):
        if row["item"] != item or (gpus is not None and row["gpu"] not in gpus):
            continue
        seconds = row["epoch_seconds"]
        if len(seconds) < 2 or row.get("wall_seconds") is None:
            continue
        steady = float(np.mean(seconds[1:]))
        config = WideConfig.from_forecaster(ForecasterConfig.from_dict(row["model_config"]), int(row["input_len"]))
        features = config_features(config, row["dataset"], int(row["horizon"]), int(row["training"]["batch_size"]), manifest)
        out.append(
            Timing(
                config_id=row["config_id"],
                config=config,
                dataset=row["dataset"],
                gpu=row["gpu"],
                features=features,
                epoch_seconds=steady,
                epochs_run=int(row["epochs_run"]),
                overhead_seconds=float(row["wall_seconds"]) - steady * len(seconds),
                wall_seconds=float(row["wall_seconds"]),
            )
        )
    return out


@dataclass
class CostModel:
    """Seconds per steady-state epoch, ``features @ coefficients``."""

    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features) @ self.coefficients

    def describe(self) -> dict[str, float]:
        return {name: float(c) for name, c in zip(FEATURES, self.coefficients)}


def fit_cost_model(features: np.ndarray, seconds: np.ndarray) -> CostModel:
    """Non-negative least squares on relative error.

    Each row is divided by its measured time, so a 10% miss on a 0.2 s
    epoch weighs as much as on a 20 s one.
    """
    features = np.asarray(features, dtype=np.float64)
    seconds = np.asarray(seconds, dtype=np.float64)
    if len(seconds) < features.shape[1] or np.any(seconds <= 0):
        raise ValueError("need positive timings, at least one per feature")
    coefficients, _ = nnls(features / seconds[:, None], np.ones(len(seconds)))
    return CostModel(coefficients=coefficients)


def fit_quality(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """R squared on seconds and on log seconds, and absolute relative errors."""
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)

    def r2(a: np.ndarray, b: np.ndarray) -> float:
        return float(1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2))

    relative = np.abs(predicted / measured - 1)
    return {
        "rows": int(len(measured)),
        "r2": r2(predicted, measured),
        "r2_log": r2(np.log(np.maximum(predicted, 1e-9)), np.log(measured)),
        "median_rel_error": float(np.median(relative)),
        "p90_rel_error": float(np.quantile(relative, 0.9)),
        "total_ratio": float(predicted.sum() / measured.sum()),
    }


def grouped_folds(groups: Sequence[str], n_folds: int, seed: int) -> np.ndarray:
    """Fold index per row; all rows of one group (a config) share a fold."""
    names = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(names))
    fold_of = {names[i]: k % n_folds for k, i in enumerate(order)}
    return np.array([fold_of[g] for g in groups])


def held_out_predictions(timings: Sequence[Timing], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Each row's epoch seconds predicted by a fit without its config's rows."""
    x = np.stack([t.features for t in timings])
    y = np.array([t.epoch_seconds for t in timings])
    folds = grouped_folds([f"{t.dataset}/{t.config_id}" for t in timings], n_folds, seed)
    predicted = np.empty_like(y)
    for k in range(n_folds):
        held = folds == k
        predicted[held] = fit_cost_model(x[~held], y[~held]).predict(x[held])
    return predicted


def split_quality(train: Sequence[Timing], test: Sequence[Timing]) -> dict[str, float]:
    """Fit on ``train`` and score on ``test``."""
    model = fit_cost_model(np.stack([t.features for t in train]), np.array([t.epoch_seconds for t in train]))
    return fit_quality(model.predict(np.stack([t.features for t in test])), np.array([t.epoch_seconds for t in test]))


def container_seconds(timings: Sequence[Timing]) -> float:
    """Measured GPU-container seconds of the rows, assuming busy workers."""
    return sum(t.wall_seconds for t in timings) / WORKERS_PER_GPU


def grid_features(dataset: str, horizon: int, batch_size: int, n_heads: int, manifest: dict[str, Any] | None = None) -> np.ndarray:
    """Epoch features of every shape of ``dataset`` with ``n_heads`` heads."""
    g = shape_grid(dataset)
    manifest = read_manifest(TRAINVAL_DIR) if manifest is None else manifest
    out = np.empty((len(g["d_model"]), len(FEATURES)))
    for length in np.unique(g["input_len"]):
        rows = g["input_len"] == length
        flops, elements = window_work(
            g["patch_len"][rows], g["stride"][rows], g["n_layers"][rows], n_heads, g["d_model"][rows],
            g["ff_ratio"][rows], g["mixing"][rows], g["head_code"][rows], g["head_rank"][rows], length, horizon,
        )
        n_train, n_val = epoch_windows(dataset, int(length), horizon, manifest)
        out[rows] = epoch_features(flops, elements, g["n_layers"][rows], n_train, n_val, batch_size)
    return out


def grid_activation_bytes(dataset: str, horizon: int, batch_size: int, n_heads: int) -> np.ndarray:
    """Rough training activation memory per shape: four bytes per element written in one batch's forward."""
    g = shape_grid(dataset)
    _, elements = window_work(
        g["patch_len"], g["stride"], g["n_layers"], n_heads, g["d_model"], g["ff_ratio"], g["mixing"],
        g["head_code"], g["head_rank"], g["input_len"], horizon,
    )
    return 4.0 * batch_size * elements


def price_enumeration(
    model: CostModel,
    dataset: str,
    horizon: int,
    mask: np.ndarray,
    batch_size: int,
    epochs: float,
    overhead_seconds: float,
    scale: float = 1.0,
    memory_limit_bytes: float | None = None,
) -> dict[str, float]:
    """GPU-hours and dollars to train every config selected by ``mask`` once.

    Each config is charged ``epochs`` steady epochs, each the model's
    prediction times ``scale``, plus ``overhead_seconds``, divided by the
    workers sharing a GPU. With ``memory_limit_bytes``, the configs whose
    rough activation memory exceeds it are counted: they need fewer workers
    per GPU. They are not charged more, because a config that large keeps
    the GPU busy on its own, so its share of the GPU stays about its
    four-worker time over four. The floor is the matmul work alone at the
    A10G's FP32 peak, which no fit can undercut.
    """
    g = shape_grid(dataset)
    manifest = read_manifest(TRAINVAL_DIR)
    per_setting = len(SEARCH_SPACE["dropout"]) * len(SEARCH_SPACE["instance_norm"])
    flops_column = FEATURES.index("gflops")
    total = configs = epoch_sum = flops = over_memory = 0.0
    for n_heads in SEARCH_SPACE["n_heads"]:
        rows = mask & (g["d_model"] % n_heads == 0)
        features = grid_features(dataset, horizon, batch_size, n_heads, manifest)[rows]
        seconds = scale * model.predict(features)
        if memory_limit_bytes is not None:
            heavy = grid_activation_bytes(dataset, horizon, batch_size, n_heads)[rows] > memory_limit_bytes
            over_memory += per_setting * heavy.sum()
        total += per_setting * ((epochs * seconds + overhead_seconds) / WORKERS_PER_GPU).sum()
        configs += per_setting * rows.sum()
        epoch_sum += per_setting * seconds.sum()
        flops += per_setting * epochs * 1e9 * features[:, flops_column].sum()
    gpu_hours = total / 3600
    floor_hours = flops / (A10G_FP32_TFLOPS * 1e12) / 3600
    return {
        "configs": int(configs),
        "mean_epoch_seconds": epoch_sum / configs if configs else 0.0,
        "gpu_hours": gpu_hours,
        "dollars": gpu_hours * DOLLARS_PER_GPU_HOUR,
        "floor_gpu_hours": floor_hours,
        "floor_dollars": floor_hours * DOLLARS_PER_GPU_HOUR,
        "configs_over_memory": int(over_memory),
    }
