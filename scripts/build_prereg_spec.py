"""Assemble the prereg spec for the single test evaluation from the ledger.

Every E6 arm becomes a prereg arm (its runs at every horizon and seed, with
checkpoint path, digest and validation MSE from the ledger). Each E3b config
becomes a one-run arm, and each dataset's E3b arms form a rank check
(validation against test Spearman, amendment A7). Only the primary
comparisons are listed (each dataset's winner against PatchTST/42), because
eval_test.py applies Holm over every listed comparison (A8: Holm over the
two datasets). The other arms are scored and reported descriptively.

Checkpoints must be downloaded first (collect_runs.py --with-checkpoints).

Example:
    python scripts/build_prereg_spec.py --id final_v1 --out results/final/prereg_spec.json
    python scripts/prereg.py --spec results/final/prereg_spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402

E3B_HORIZON = {"ETTh1": 336, "ILI": 24}


def checkpoint(row: dict) -> str:
    return f"results/runs/{row['item']}/{row['dataset']}_T{row['horizon']}_s{row['seed']}/ckpt/{row['config_id']}.pt"


def run_entry(row: dict) -> dict:
    return {"horizon": int(row["horizon"]), "seed": int(row["seed"]), "checkpoint": checkpoint(row),
            "sha256": row["checkpoint_sha256"], "val_mse": float(row["val"]["mse"]),
            "batch_size": int(row["training"]["batch_size"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--winner-arm", default="winner", help="E6 arm name of each dataset's LeagueTST winner")
    args = parser.parse_args()
    rows = list(read_rows(ROOT / "results" / "ledger.jsonl"))
    arms: dict[tuple, dict] = {}
    for row in rows:
        if row["item"] != "E6":
            continue
        key = (row["dataset"], row["config_id"])
        arm = arms.setdefault(key, {"name": f"{row['config_id']}_{row['dataset']}", "dataset": row["dataset"],
                                    "model_config": row["model_config"], "runs": []})
        arm["runs"].append(run_entry(row))
    rank_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["item"] == "E3b" and int(row["horizon"]) == E3B_HORIZON[row["dataset"]]:
            name = f"e3b_{row['dataset']}_{row['config_id']}"
            arms[(row["dataset"], name)] = {"name": name, "dataset": row["dataset"], "model_config": row["model_config"],
                                           "runs": [run_entry(row)]}
            rank_groups[row["dataset"]].append(name)
    missing = [r["checkpoint"] for a in arms.values() for r in a["runs"] if not (ROOT / r["checkpoint"]).exists()]
    if missing:
        print(f"{len(missing)} checkpoints not downloaded, e.g. {missing[0]}")
        return 1
    comparisons = []
    for dataset in sorted({d for d, _ in arms}):
        winner, baseline = f"{args.winner_arm}_{dataset}", f"patchtst42_{dataset}"
        names = {a["name"] for a in arms.values()}
        if winner in names and baseline in names:
            comparisons.append({"name": f"{dataset}: LeagueTST winner vs PatchTST/42", "a": winner, "b": baseline})
    spec = {"id": args.id, "alpha": 0.05, "bootstrap": {"resamples": 4000, "seed": 0},
            "primary_endpoint": "mean exact test MSE over the four horizons of each dataset",
            "arms": sorted(arms.values(), key=lambda a: a["name"]), "comparisons": comparisons,
            "rank_checks": [{"name": f"E3b {d} validation vs test", "arms": sorted(n)} for d, n in sorted(rank_groups.items())]}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(spec['arms'])} arms, {sum(len(a['runs']) for a in spec['arms'])} runs, "
          f"{len(comparisons)} comparisons, {len(spec['rank_checks'])} rank checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
