"""Data integrity, leakage and parameter-count checks on the real benchmark files.

Usage:
    .venv/bin/python -I scripts/validation/data_integrity.py "$PWD" /tmp/leaguetst_integrity

The second argument is a scratch directory for the poisoned copies.
"""
import json, sys
from pathlib import Path
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from leaguetst.data import BENCHMARKS, build_splits, load_benchmark_frame, split_borders
from leaguetst.data.datasets import Standardizer
from leaguetst.experiment import build_model
from leaguetst.models import count_parameters
data = repo / "data"; checks = []
def check(ok, text): checks.append((bool(ok), text)); print(("PASS " if ok else "FAIL ") + text)

for name, spec in BENCHMARKS.items():
    raw = pd.read_csv(data / spec.relative_path)
    dates = pd.to_datetime(raw["date"])
    steps = dates.diff().dropna().value_counts()
    check(raw.shape[0] == {"ETTh1": 17420, "ILI": 966}[name], f"{name}: {raw.shape[0]} rows")
    check(list(raw.columns)[0] == "date" and list(raw.columns)[-1] == "OT" and raw.shape[1] == 8, f"{name}: columns {list(raw.columns)}")
    check(dates.is_monotonic_increasing and dates.is_unique, f"{name}: dates strictly increasing")
    check(len(steps) == 1, f"{name}: one constant time step ({dict((str(k), int(v)) for k, v in steps.items())})")
    numeric = raw.drop(columns=["date"]).to_numpy(dtype=np.float64)
    check(np.isfinite(numeric).all(), f"{name}: all {numeric.size} values finite")
    check(not (raw.drop(columns=["date"]).std() == 0).any(), f"{name}: no constant column")
    borders = split_borders(len(raw), spec.input_len, spec.split_scheme)
    used = borders["test"][1]
    check(True, f"{name}: borders {borders}; rows used {used} of {len(raw)}")
    # targets of each split cover disjoint, contiguous row ranges
    target_ranges = {k: (s + spec.input_len, e) for k, (s, e) in borders.items()}
    check(target_ranges["train"][1] <= target_ranges["val"][0] and target_ranges["val"][1] <= target_ranges["test"][0],
          f"{name}: target rows disjoint {target_ranges}")
    # the scaler must not move when validation or test rows change, and must move when a training row changes
    # End to end: poison every row after the training block in a copy of the file.
    # The scaler and every training window must come out identical.
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp(dir=sys.argv[2]))
    (tmp / spec.relative_path).parent.mkdir(parents=True)
    poisoned = raw.copy()
    cols = [c for c in raw.columns if c != "date"]
    poisoned.loc[borders["train"][1]:, cols] = poisoned.loc[borders["train"][1]:, cols] * 1000 + 1e6
    poisoned.to_csv(tmp / spec.relative_path, index=False)
    clean_splits, clean_scaler = build_splits(data, name, spec.input_len, spec.horizons[0])
    dirty_splits, dirty_scaler = build_splits(tmp, name, spec.input_len, spec.horizons[0])
    same_scaler = np.allclose(clean_scaler.mean, dirty_scaler.mean, rtol=1e-12, atol=0) and np.allclose(clean_scaler.scale, dirty_scaler.scale, rtol=1e-12, atol=0)
    same_train = np.array_equal(clean_splits["train"].values.numpy(), dirty_splits["train"].values.numpy())
    moved_test = not np.allclose(clean_splits["test"].values.numpy()[-1], dirty_splits["test"].values.numpy()[-1])
    check(same_scaler and same_train and moved_test, f"{name}: poisoning validation and test rows leaves scaler and training windows unchanged (scaler {same_scaler}, train windows {same_train}, poison visible in test {moved_test})")
    shutil.rmtree(tmp)
    for horizon in spec.horizons:
        splits, _ = build_splits(data, name, spec.input_len, horizon)
        expected = {k: (e - s) - spec.input_len - horizon + 1 for k, (s, e) in borders.items()}
        got = {k: len(v) for k, v in splits.items()}
        check(got == expected, f"{name} T={horizon}: windows {got}")

def hand_patchtst(patch_len, n_patches, horizon, d=16, d_ff=128, layers=3):
    return (patch_len * d + d) + n_patches * d + layers * (4 * (d * d + d) + (d * d_ff + d_ff) + (d_ff * d + d) + 4 * d) + (d * n_patches * horizon + horizon)

records = {p.stem: json.loads(p.read_text()) for d in ("results/preliminary", "results/baselines") for p in (repo / d).glob("*_s*.json")}
for name, spec in BENCHMARKS.items():
    cfg = {"ETTh1": dict(patch_len=16, stride=8), "ILI": dict(patch_len=24, stride=2)}[name]
    full = dict(n_layers=3, n_heads=4, d_model=16, ff_ratio=8, dropout=0.3, instance_norm=True, channel_mode="independent", **cfg)
    for horizon in spec.horizons:
        p = count_parameters(build_model("patchtst", full, 7, spec.input_len, horizon))
        d = count_parameters(build_model("dlinear", {"kernel_size": 25}, 7, spec.input_len, horizon))
        hp, hd = hand_patchtst(cfg["patch_len"], 42, horizon), 2 * (spec.input_len * horizon + horizon)
        rec = [r["n_params"] for k, r in records.items() if r["dataset"] == name and r["horizon"] == horizon and r["model"] == "patchtst"]
        check(p == hp and d == hd and all(x == p for x in rec), f"{name} T={horizon}: PatchTST {p:,} = hand {hp:,}; DLinear {d:,} = hand {hd:,}; records {sorted(set(rec))}")
print(f"\n{sum(o for o, _ in checks)} of {len(checks)} data checks passed")
