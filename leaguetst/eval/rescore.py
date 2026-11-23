"""Re-score a saved checkpoint on the validation windows with NumPy only.

The windows are rebuilt from the trainval CSV with pandas and the
manifest's borders; the scaler is recomputed here. Torch is used only to
read the weights file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leaguetst.eval.metrics import exact_and_drop_last
from leaguetst.eval.numpy_model import numpy_patchtst_forward

FILES = {"ETTh1": "ETT-small/ETTh1.csv", "ILI": "illness/national_illness.csv"}


def validation_windows(trainval_dir: str | Path, dataset: str, input_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(inputs, targets) of every validation window, standardised on the training rows."""
    trainval_dir = Path(trainval_dir)
    manifest = json.loads((trainval_dir / "MANIFEST.json").read_text(encoding="utf-8"))["datasets"][dataset]
    path = trainval_dir / FILES[dataset]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{path} does not match the manifest")
    frame = pd.read_csv(path).drop(columns=["date"])
    columns = [c for c in frame.columns if c != "OT"] + ["OT"]
    values = frame[columns].to_numpy(dtype=np.float64)
    train_end, val_end = int(manifest["train_end"]), int(manifest["val_end"])
    train = values[:train_end]
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (values - train.mean(axis=0)) / scale
    block = scaled[train_end - input_len : val_end]
    n = len(block) - input_len - horizon + 1
    starts = np.arange(n)
    inputs = block[starts[:, None] + np.arange(input_len)[None, :]]
    targets = block[starts[:, None] + input_len + np.arange(horizon)[None, :]]
    return inputs, targets


def load_numpy_state(path: str | Path, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    import torch

    data = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{path}: SHA-256 differs from the record")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.double().numpy() for k, v in state.items()}


def rescore_checkpoint(record: dict[str, Any], checkpoint: str | Path, trainval_dir: str | Path, chunk: int = 256) -> dict[str, Any]:
    """Validation metrics of a table record recomputed from its weights."""
    state = load_numpy_state(checkpoint, record.get("checkpoint_sha256") or None)
    horizon, input_len = int(record["horizon"]), int(record["input_len"])
    inputs, targets = validation_windows(trainval_dir, record["dataset"], input_len, horizon)
    forecasts = np.concatenate(
        [numpy_patchtst_forward(state, record["model_config"], inputs[i : i + chunk], horizon) for i in range(0, len(inputs), chunk)]
    )
    result = exact_and_drop_last(forecasts, targets, int(record["training"]["batch_size"]))
    result["abs_gap_mse"] = abs(result["mse"] - float(record["val"]["mse"]))
    return result
