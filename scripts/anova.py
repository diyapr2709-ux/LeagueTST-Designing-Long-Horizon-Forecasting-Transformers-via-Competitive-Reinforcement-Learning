"""E8 (RQ3): which design choices matter, from one table.

The 2,368 feasible configs are exactly 74 architectures (patch length,
stride, layers, width, feed-forward ratio) crossed with 32 training
settings (heads, dropout, instance normalisation), so a balanced two-way
ANOVA splits the variance of the table into architecture, training
setting and their interaction. With several seeds per config, seed noise
is separated from the interaction. Each field's share is also reported as
the variance of its marginal means (main effect) over the total.

Example:
    python scripts/anova.py --dataset ILI --horizon 24
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import MUTABLE_FIELDS, Space  # noqa: E402

ARCH = ("patch_len", "stride", "n_layers", "d_model", "ff_ratio")
SETTING = ("n_heads", "dropout", "instance_norm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="E1")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--log", action="store_true", help="analyse log validation MSE")
    args = parser.parse_args()

    rows = [r for r in read_rows(ROOT / "results" / "ledger.jsonl") if r["item"] == args.table
            and r["dataset"] == args.dataset and int(r["horizon"]) == args.horizon and np.isfinite(r["best_val_mse"])]
    per_config: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_config[row["config_id"]].append(np.log(row["best_val_mse"]) if args.log else row["best_val_mse"])
    space = Space(per_config)
    seeds = min(len(v) for v in per_config.values())
    # Use the same number of seeds for every config so the design stays balanced.
    values = {c: np.array(v[:seeds]) for c, v in per_config.items()}
    archs = sorted({space.architecture(c) for c in values})
    settings = sorted({tuple(space.choices[c][f] for f in SETTING) for c in values}, key=str)
    cube = np.full((len(archs), len(settings), seeds), np.nan)
    a_index = {a: i for i, a in enumerate(archs)}
    s_index = {s: i for i, s in enumerate(settings)}
    for cid, vals in values.items():
        cube[a_index[space.architecture(cid)], s_index[tuple(space.choices[cid][f] for f in SETTING)]] = vals
    if np.isnan(cube).any():
        raise SystemExit("the table is not a full architecture x setting product")

    grand = cube.mean()
    cell = cube.mean(axis=2)
    a_mean = cell.mean(axis=1)
    s_mean = cell.mean(axis=0)
    n_a, n_s, n_r = cube.shape
    ss_a = n_s * n_r * ((a_mean - grand) ** 2).sum()
    ss_s = n_a * n_r * ((s_mean - grand) ** 2).sum()
    ss_cells = n_r * ((cell - grand) ** 2).sum()
    ss_int = ss_cells - ss_a - ss_s
    ss_noise = ((cube - cell[:, :, None]) ** 2).sum()
    ss_total = ((cube - grand) ** 2).sum()
    result = {"dataset": args.dataset, "horizon": args.horizon, "log_scale": args.log, "configs": len(values),
              "architectures": n_a, "settings": n_s, "seeds_used": n_r,
              "share": {"architecture": ss_a / ss_total, "training_setting": ss_s / ss_total,
                        "interaction": ss_int / ss_total if n_r > 1 else None,
                        "interaction_plus_noise": (ss_int + ss_noise) / ss_total if n_r == 1 else None,
                        "seed_noise": ss_noise / ss_total if n_r > 1 else None},
              "main_effects": {}}
    for field in MUTABLE_FIELDS:
        groups: dict = defaultdict(list)
        for cid, vals in values.items():
            groups[space.choices[cid][field]].append(vals.mean())
        means = {str(k): float(np.mean(v)) for k, v in groups.items()}
        weights = {str(k): len(v) for k, v in groups.items()}
        between = sum(weights[k] * n_r * (means[k] - grand) ** 2 for k in means)
        result["main_effects"][field] = {"share": float(between / ss_total), "means": means, "counts": weights}
    best_arch = archs[int(np.argmin(a_mean))]
    best_setting = settings[int(np.argmin(s_mean))]
    result["best_architecture"] = dict(zip(ARCH, best_arch))
    result["best_setting"] = dict(zip(SETTING, best_setting))
    out = ROOT / "results" / "anova" / f"{args.dataset}_T{args.horizon}{'_log' if args.log else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "main_effects"}, indent=2, default=str))
    for field, effect in sorted(result["main_effects"].items(), key=lambda kv: -kv[1]["share"]):
        print(f"{field:14s} share {effect['share']:.3f}  means {effect['means']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
