"""E3b, table validity, and the ETTh1 top-100 reseed list.

    select   write the 64 configs to retrain live: the top 20 by table mean and
             44 spread evenly over the remaining ranks (fixed before launch)
    top100   write the 100 best configs by the seed-2021 value, for seeds 2022
             and 2023 (ETTh1)
    analyze  compare the live runs (item E3b, seed 3001) with the table: the 95%
             bootstrap interval of the mean signed gap must contain 0 and the
             median absolute gap must be at most 1.15 seed SD (0.95 on ILI,
             whose table value is a 3-seed mean)

Examples:
    python scripts/e3b.py select --dataset ILI --horizon 24
    python scripts/e3b.py analyze --dataset ILI --horizon 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import Table  # noqa: E402

THRESHOLD_SD = {"ETTh1": 1.15, "ILI": 0.95}
LIVE_SEED = 3001
OUT = ROOT / "results" / "e3b"


def table_rows(dataset: str, horizon: int, item: str = "E1") -> list[dict]:
    return [r for r in read_rows(ROOT / "results" / "ledger.jsonl") if r["item"] == item and r["dataset"] == dataset
            and int(r["horizon"]) == horizon]


def select(dataset: str, horizon: int) -> list[str]:
    table = Table(table_rows(dataset, horizon))
    ranked = sorted(table.ids, key=table.true_value)
    rest = ranked[20:]
    picks = ranked[:20] + [rest[int(round(i))] for i in np.linspace(0, len(rest) - 1, 44)]
    if len(set(picks)) != 64:
        raise AssertionError("selection is not 64 distinct configs")
    return picks


def top100(dataset: str, horizon: int, seed: int = 2021) -> list[str]:
    rows = [r for r in table_rows(dataset, horizon) if int(r["seed"]) == seed]
    return [r["config_id"] for r in sorted(rows, key=lambda r: r["best_val_mse"])[:100]]


def analyze(dataset: str, horizon: int) -> dict:
    table = Table(table_rows(dataset, horizon))
    ids = (OUT / f"{dataset}_T{horizon}_ids.txt").read_text().split()
    live = {r["config_id"]: r["best_val_mse"] for r in table_rows(dataset, horizon, "E3b") if int(r["seed"]) == LIVE_SEED}
    missing = [c for c in ids if c not in live]
    if missing:
        raise SystemExit(f"{len(missing)} live runs missing")
    sigma = table.seed_sd()
    gaps = np.array([live[c] - table.true_value(c) for c in ids])
    rng = np.random.default_rng(0)
    means = gaps[rng.integers(0, len(gaps), (10000, len(gaps)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    median_abs_sd = float(np.median(np.abs(gaps)) / sigma)
    rho_all = spearmanr([live[c] for c in ids], [table.true_value(c) for c in ids])[0]
    top = ids[:20]
    rho_top = spearmanr([live[c] for c in top], [table.true_value(c) for c in top])[0]
    result = {"dataset": dataset, "horizon": horizon, "n": len(ids), "seed_sd": sigma,
              "mean_signed_gap": float(gaps.mean()), "gap_ci95": [float(low), float(high)],
              "median_abs_gap_sd": median_abs_sd, "threshold_sd": THRESHOLD_SD[dataset],
              "spearman_all": float(rho_all), "spearman_top20": float(rho_top),
              "bias_ok": bool(low <= 0 <= high), "spread_ok": median_abs_sd <= THRESHOLD_SD[dataset]}
    result["k7_pass"] = result["bias_ok"] and result["spread_ok"]
    (OUT / f"{dataset}_T{horizon}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "top100", "analyze"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "select":
        path = OUT / f"{args.dataset}_T{args.horizon}_ids.txt"
        if path.exists():
            print(f"{path} exists; the selection is fixed once")
            return 1
        path.write_text("\n".join(select(args.dataset, args.horizon)) + "\n")
        print(f"wrote {path}")
    elif args.command == "top100":
        path = OUT / f"{args.dataset}_T{args.horizon}_top100.txt"
        if path.exists():
            print(f"{path} exists; the list is fixed once")
            return 1
        path.write_text("\n".join(top100(args.dataset, args.horizon)) + "\n")
        print(f"wrote {path}")
    else:
        print(json.dumps(analyze(args.dataset, args.horizon), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
