"""One entry of the tabular benchmark: train a config on trainval rows only.

Every candidate uses the audited PatchTST/42 recipe of its dataset (Adam
with first beta 0.95, epoch 1 at a twenty-fifth of the rate, the official
schedule), capped at ``TABLE_EPOCHS`` epochs with early stopping after
``TABLE_PATIENCE`` epochs without a gain. Validation MSE is exact over
every window and logged at every epoch, so screening at any epoch k can be
replayed from the record without training again.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from leaguetst.data.device_loader import build_trainval_loaders
from leaguetst.experiment import build_model
from leaguetst.models import count_parameters
from leaguetst.search.feasible import BUDGETS, INPUT_LEN
from leaguetst.training import Trainer, TrainingConfig
from leaguetst.utils import load_yaml, save_json, save_state_dict, set_seed, sha256_file

log = logging.getLogger(__name__)

TABLE_EPOCHS = 30
TABLE_PATIENCE = 10
RECIPE_FILES = {"ETTh1": "configs/patchtst42_etth1.yaml", "ILI": "configs/patchtst42_ili.yaml"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def recipe(dataset: str, epochs: int = TABLE_EPOCHS, patience: int = TABLE_PATIENCE) -> dict[str, Any]:
    """The dataset's PatchTST/42 training settings with the table's epoch cap."""
    loaded = load_yaml(REPO_ROOT / RECIPE_FILES[dataset])
    training = dict(loaded["training"])
    training["epochs"] = epochs
    training["patience"] = patience
    return asdict(TrainingConfig(**training))


def patchtst42_config(dataset: str) -> dict[str, Any]:
    """Model fields of PatchTST/42 for ``dataset`` (not a grid point)."""
    return dict(load_yaml(REPO_ROOT / RECIPE_FILES[dataset])["model_config"])


def strict_float32() -> None:
    """Full float32 matmuls on CUDA, so GPU scores match the NumPy re-scorer."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


@torch.no_grad()
def per_window_mse(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    """Mean squared error of every validation window, in loader order."""
    model.eval()
    chunks = []
    for inputs, targets in loader:
        error = (model(inputs.to(device)) - targets.to(device)).flatten(start_dim=1)
        chunks.append(error.pow(2).mean(dim=1).cpu().double().numpy())
    return np.concatenate(chunks)


def run_entry(
    *,
    item: str,
    dataset: str,
    horizon: int,
    seed: int,
    config_key: str,
    model_config: dict[str, Any],
    training: dict[str, Any],
    trainval_dir: str | Path,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    perwindow_path: str | Path | None = None,
    git_hash: str = "",
) -> dict[str, Any]:
    """Train one config at one setting and seed; return its record.

    When ``perwindow_path`` is given, the validation MSE of every window at
    the restored best epoch is saved there as a float32 ``.npy`` array.
    """
    strict_float32()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    input_len = INPUT_LEN[dataset]
    config = TrainingConfig(**training)
    set_seed(seed)
    loaders, _ = build_trainval_loaders(trainval_dir, dataset, input_len, horizon, config.batch_size, seed, device)
    model = build_model("patchtst", model_config, loaders["train"].dataset.n_vars, input_len, horizon)
    n_params = count_parameters(model)
    trainer = Trainer(model, loaders, config, device)
    fit = trainer.fit()
    val = trainer.evaluate("val")

    checkpoint_sha256 = ""
    if checkpoint_path:
        checkpoint_sha256 = save_state_dict(model.state_dict(), checkpoint_path)
    perwindow_sha256 = ""
    if perwindow_path:
        errors = per_window_mse(model, loaders["val"], device).astype(np.float32)
        perwindow_path = Path(perwindow_path)
        perwindow_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(perwindow_path, errors)
        perwindow_sha256 = sha256_file(perwindow_path)

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    return {
        "item": item,
        "dataset": dataset,
        "horizon": horizon,
        "input_len": input_len,
        "seed": seed,
        "config_id": config_key,
        "model_config": model_config,
        "n_params": n_params,
        "budget": BUDGETS.get((dataset, horizon)),
        "training": asdict(config),
        "best_val_mse": fit.best_val_mse,
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        "diverged": fit.diverged,
        "train_seconds": fit.train_seconds,
        "history": fit.history,
        "val": val,
        "checkpoint": Path(checkpoint_path).name if checkpoint_path else "",
        "checkpoint_sha256": checkpoint_sha256,
        "perwindow": Path(perwindow_path).name if perwindow_path else "",
        "perwindow_sha256": perwindow_sha256,
        "git_hash": git_hash,
        "environment": {
            "device": str(device),
            "gpu": gpu,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def run_entries(job: dict[str, Any], entries: list[tuple[str, dict[str, Any]]], out_dir: str | Path, device_name: str) -> dict[str, int]:
    """Train each (config_key, model_config) pair whose record is missing.

    Records go to ``out_dir/records/<key>.json`` and weights to
    ``out_dir/ckpt/<key>.pt``. A record that exists is skipped, so a
    restarted job resumes where it stopped.
    """
    out_dir = Path(out_dir)
    device = torch.device(device_name)
    torch.set_num_threads(max(1, int(job.get("threads_per_worker", 2))))
    done = skipped = failed = 0
    for key, model_config in entries:
        record_path = out_dir / "records" / f"{key}.json"
        if record_path.exists():
            skipped += 1
            continue
        start = time.perf_counter()
        try:
            record = run_entry(
                item=job["item"],
                dataset=job["dataset"],
                horizon=int(job["horizon"]),
                seed=int(job["seed"]),
                config_key=key,
                model_config=model_config,
                training=job["training"],
                trainval_dir=job["trainval_dir"],
                device=device,
                checkpoint_path=out_dir / "ckpt" / f"{key}.pt",
                perwindow_path=out_dir / "perwindow" / f"{key}.npy",
                git_hash=job.get("git_hash", ""),
            )
        except Exception as error:  # a failed config is logged and the chunk moves on
            failed += 1
            save_json({"config_id": key, "error": repr(error)}, out_dir / "failed" / f"{key}.json")
            log.exception("config %s failed", key)
            continue
        record["wall_seconds"] = time.perf_counter() - start
        save_json(record, record_path)
        done += 1
        log.info("%s %s best %.5f at epoch %d/%d in %.0fs", job["item"], key, record["best_val_mse"],
                 record["best_epoch"], record["epochs_run"], record["wall_seconds"])
    return {"done": done, "skipped": skipped, "failed": failed}


def worker_main(job: dict[str, Any], entries: list, out_dir: str, device_name: str) -> dict[str, int]:
    """Entry point for one worker process on a GPU container."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return run_entries(job, [tuple(e) for e in entries], out_dir, device_name)
