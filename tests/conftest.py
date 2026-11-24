"""Shared fixtures: a synthetic copy of both benchmark files and its trainval copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leaguetst.data.trainval import make_trainval

ETT_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
ILI_COLUMNS = [
    "% WEIGHTED ILI", "%UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24",
    "ILITOTAL", "NUM. OF PROVIDERS", "OT",
]


def write_synthetic(path: Path, columns: list[str], n_rows: int, freq: str, seed: int) -> None:
    """Write a CSV with a date column and smooth random variables."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_rows)[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=(1, len(columns)))
    period = rng.uniform(24, 400, size=(1, len(columns)))
    values = np.sin(2 * np.pi * time / period + phase) * rng.uniform(1, 5, size=(1, len(columns)))
    values += rng.normal(scale=0.1, size=values.shape) + rng.uniform(-10, 10, size=(1, len(columns)))
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "date", pd.date_range("2016-07-01", periods=n_rows, freq=freq))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    write_synthetic(root / "ETT-small" / "ETTh1.csv", ETT_COLUMNS, 17420, "h", seed=1)
    write_synthetic(root / "illness" / "national_illness.csv", ILI_COLUMNS, 966, "W", seed=2)
    return root


@pytest.fixture(scope="session")
def trainval_dir(synthetic_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("trainval")
    make_trainval(synthetic_data_dir, out)
    return out


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "data"
    needed = [root / "ETT-small" / "ETTh1.csv", root / "illness" / "national_illness.csv"]
    if not all(p.exists() for p in needed):
        pytest.skip("benchmark files not downloaded")
    return root
