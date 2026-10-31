"""Windowed datasets for the ETTh1 and ILI benchmarks.

Both benchmarks follow the split and scaling protocol of the official PatchTST
code so that errors stay comparable with the published numbers:

* ETTh1 is split by calendar months: 12 for training, 4 for validation and
  4 for testing, counting 30 days per month and 24 readings per day.
* ILI is split by ratio: 70% training, 20% testing, the rest validation.
* The validation and test blocks start ``input_len`` rows early so that the
  first target of each block directly follows the previous block.
* Every variable is standardised with the mean and standard deviation of the
  training block only.

Files are checked on load: the columns must be exactly the expected ones,
rows must be in time order at a constant step, and every value must be
finite. A shuffled, truncated or edited copy fails loudly instead of
training on the wrong windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")

HOURS_PER_MONTH = 30 * 24
ETT_TRAIN_STEPS = 12 * HOURS_PER_MONTH
ETT_VAL_STEPS = 4 * HOURS_PER_MONTH
ETT_TEST_STEPS = 4 * HOURS_PER_MONTH

RATIO_TRAIN = 0.7
RATIO_TEST = 0.2

DATE_COLUMN = "date"
TARGET_COLUMN = "OT"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Where a benchmark lives and how it is split."""

    name: str
    relative_path: str
    split_scheme: str
    input_len: int
    horizons: tuple[int, ...]
    columns: tuple[str, ...]
    step: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "ETTh1": BenchmarkSpec(
        name="ETTh1",
        relative_path="ETT-small/ETTh1.csv",
        split_scheme="ett",
        input_len=336,
        horizons=(96, 192, 336, 720),
        columns=("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        step="1h",
    ),
    "ILI": BenchmarkSpec(
        name="ILI",
        relative_path="illness/national_illness.csv",
        split_scheme="ratio",
        input_len=104,
        horizons=(24, 36, 48, 60),
        columns=(
            "% WEIGHTED ILI", "%UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24",
            "ILITOTAL", "NUM. OF PROVIDERS", "OT",
        ),
        step="7D",
    ),
}


@dataclass(frozen=True)
class Standardizer:
    """Per-variable mean and standard deviation fitted on the training block.

    A variable with zero variance keeps a scale of one, as scikit-learn's
    StandardScaler does, so the transform never divides by zero.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def split_borders(n_steps: int, input_len: int, scheme: str) -> dict[str, tuple[int, int]]:
    """Return the half-open row range of each split.

    ``scheme`` is ``"ett"`` for the month-based ETT split or ``"ratio"`` for
    the 70/10/20 split used by the ILI benchmark.
    """
    if scheme == "ett":
        train_end = ETT_TRAIN_STEPS
        val_end = train_end + ETT_VAL_STEPS
        test_end = val_end + ETT_TEST_STEPS
    elif scheme == "ratio":
        n_train = int(n_steps * RATIO_TRAIN)
        n_test = int(n_steps * RATIO_TEST)
        train_end = n_train
        val_end = n_steps - n_test
        test_end = n_steps
    else:
        raise ValueError(f"unknown split scheme: {scheme!r}")

    if test_end > n_steps:
        raise ValueError(
            f"the {scheme} split needs {test_end} rows but the file has {n_steps}"
        )
    if train_end <= input_len:
        raise ValueError("the training block is shorter than the input window")

    return {
        "train": (0, train_end),
        "val": (train_end - input_len, val_end),
        "test": (val_end - input_len, test_end),
    }


def load_benchmark_frame(data_dir: str | Path, name: str) -> pd.DataFrame:
    """Read and check a benchmark CSV; return its variables with the target last.

    The date column is dropped after it has been checked. Column order
    follows the file except that the ``OT`` target is moved to the last
    position, as in the official loader.
    """
    spec = BENCHMARKS[name]
    path = Path(data_dir) / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/download_data.py or see data/README.md"
        )
    frame = pd.read_csv(path)
    expected = [DATE_COLUMN, *spec.columns]
    if list(frame.columns) != expected:
        raise ValueError(f"{path}: columns {list(frame.columns)} differ from the expected {expected}")
    try:
        dates = pd.to_datetime(frame[DATE_COLUMN])
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: unreadable date column ({error})") from error
    if len(dates) > 1 and not (dates.diff().iloc[1:] == pd.Timedelta(spec.step)).all():
        raise ValueError(f"{path}: rows are not in time order at a constant {spec.step} step")
    columns = [c for c in spec.columns if c != TARGET_COLUMN] + [TARGET_COLUMN]
    try:
        values = frame[columns].astype(np.float64)
    except ValueError as error:
        raise ValueError(f"{path}: non-numeric values ({error})") from error
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{path}: missing or infinite values")
    return values


class WindowDataset(Dataset):
    """Sliding windows of ``input_len`` inputs followed by ``horizon`` targets.

    Window ``i`` starts at row ``i`` of ``values``, so consecutive windows
    overlap by all but one row. Values are stored once as a float32 tensor and
    sliced on access.
    """

    def __init__(self, values: np.ndarray, input_len: int, horizon: int):
        if values.ndim != 2:
            raise ValueError("values must have shape (steps, variables)")
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.input_len = input_len
        self.horizon = horizon
        self.n_windows = len(values) - input_len - horizon + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"{len(values)} rows cannot hold one window of {input_len} + {horizon} steps"
            )

    @property
    def n_vars(self) -> int:
        return self.values.shape[1]

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        middle = index + self.input_len
        end = middle + self.horizon
        return self.values[index:middle], self.values[middle:end]


def build_splits(
    data_dir: str | Path,
    name: str,
    input_len: int,
    horizon: int,
) -> tuple[dict[str, WindowDataset], Standardizer]:
    """Load a benchmark and return its train, val and test window datasets.

    The standardizer is fitted on the training block and applied to every
    split. It is returned so that callers can undo the scaling if needed.
    """
    spec = BENCHMARKS[name]
    frame = load_benchmark_frame(data_dir, name)
    values = frame.to_numpy()
    borders = split_borders(len(values), input_len, spec.split_scheme)

    train_start, train_end = borders["train"]
    standardizer = Standardizer.fit(values[train_start:train_end])
    scaled = standardizer.transform(values)

    datasets = {
        split: WindowDataset(scaled[start:end], input_len, horizon)
        for split, (start, end) in borders.items()
    }
    return datasets, standardizer
