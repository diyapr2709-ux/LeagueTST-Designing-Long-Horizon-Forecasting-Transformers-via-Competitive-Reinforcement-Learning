"""Pick reseeds: score every method's picks on two fresh seeds.

    collect  list the distinct incumbents of every run (all methods, E3 and E4)
             at budgets 25, 50 and 100 that have no fresh-seed runs yet
    analyze  per method and budget, the mean over search seeds of the picks'
             fresh-seed validation MSE (mean of seeds 4001 and 4002), with a
             bootstrap interval, the median (K4 compares medians), and each
             method against random search (permutation test, Holm)
    winner   the LeagueTST winner for one dataset: among the budget-100 picks
             of every league-GRPO deployment on it, the config with the lowest
             fresh-seed mean

Examples:
    python scripts/picks.py collect --dataset ILI --horizon 24
    python scripts/picks.py analyze --dataset ILI --horizon 24
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

from leaguetst.eval.stats import holm, permutation_test  # noqa: E402
from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import FULL_RUN_EPOCHS, Table, load_runs, no_incumbent_value, regret_from_trace  # noqa: E402

BUDGETS = (25, 50, 100)
FRESH_SEEDS = (4001, 4002)
OUT = ROOT / "results" / "picks"


def setting_runs(dataset: str, horizon: int) -> list[dict]:
    runs = []
    for item_dir in sorted((ROOT / "results" / "search").glob("*")):
        runs += load_runs(item_dir / f"{dataset}_T{horizon}")
    return runs


def ledger(item: str, dataset: str, horizon: int) -> list[dict]:
    return [r for r in read_rows(ROOT / "results" / "ledger.jsonl") if r["item"] == item and r["dataset"] == dataset
            and int(r["horizon"]) == horizon]


def incumbents(runs: list[dict], table: Table) -> list[tuple[str, int, int, str | None]]:
    fallback = no_incumbent_value(table)
    out = []
    for run in runs:
        for b in BUDGETS:
            _, cid = regret_from_trace(run["trace"], table, b * FULL_RUN_EPOCHS, fallback)
            out.append((run["method"], int(run["seed"]), b, cid))
    return out


def fresh_values(dataset: str, horizon: int) -> dict[str, float]:
    by_config: dict[str, dict[int, float]] = defaultdict(dict)
    for row in ledger("PICK", dataset, horizon):
        by_config[row["config_id"]][int(row["seed"])] = row["best_val_mse"]
    return {c: float(np.mean([v[s] for s in FRESH_SEEDS])) for c, v in by_config.items() if all(s in v for s in FRESH_SEEDS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "analyze", "winner"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS),
                        help="collect: budgets whose picks are listed (batch 2 uses 25 50, as PLAN says)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table = Table(ledger("E1", args.dataset, args.horizon))
    picks = incumbents(setting_runs(args.dataset, args.horizon), table)

    if args.command == "collect":
        done = set(fresh_values(args.dataset, args.horizon))
        listed = set()
        for path in OUT.glob(f"{args.dataset}_T{args.horizon}_ids_*.txt"):
            listed |= set(path.read_text().split())
        new = sorted({c for _, _, b, c in picks if c and b in args.budgets} - done - listed)
        batch = len(list(OUT.glob(f"{args.dataset}_T{args.horizon}_ids_*.txt"))) + 1
        path = OUT / f"{args.dataset}_T{args.horizon}_ids_{batch}.txt"
        if new:
            path.write_text("\n".join(new) + "\n")
        print(f"{len(new)} new picks to reseed" + (f" in {path}" if new else ""))
        return 0

    fresh = fresh_values(args.dataset, args.horizon)
    if args.command == "winner":
        candidates = {c for m, _, b, c in picks if c and b == 100 and m.startswith("grpo_league")}
        missing = [c for c in candidates if c not in fresh]
        if not candidates or missing:
            print(f"{len(candidates)} league picks, {len(missing)} without fresh seeds")
            return 1
        winner = min(candidates, key=fresh.get)
        result = {"dataset": args.dataset, "winner": winner, "fresh_mean": fresh[winner],
                  "model_config": table.model_configs[winner], "candidates": len(candidates)}
        (OUT / f"{args.dataset}_winner.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    missing = set()
    for method, _, b, cid in sorted(picks):
        if cid is None:
            continue
        if cid not in fresh:
            missing.add(cid)
            continue
        values[(method, b)].append(fresh[cid])
    if missing:
        print(f"warning: {len(missing)} picks have no fresh-seed runs yet")
    rng = np.random.default_rng(0)
    summary = defaultdict(dict)
    p_values = {}
    for (method, b), vals in sorted(values.items()):
        arr = np.array(vals)
        boots = arr[rng.integers(0, len(arr), (4000, len(arr)))].mean(axis=1)
        summary[method][str(b)] = {"n": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
                                   "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}
        if method != "random" and ("random", b) in values and len(arr) >= 2:
            p_values[f"{method} vs random @{b}"] = permutation_test(arr, np.array(values[("random", b)]))["p"]
    rejected = holm(p_values) if p_values else {}
    result = {"dataset": args.dataset, "horizon": args.horizon, "fresh_seeds": FRESH_SEEDS, "methods": summary,
              "vs_random": {k: {"p": p, "significant_after_holm": rejected[k]} for k, p in p_values.items()}}
    (OUT / f"{args.dataset}_T{args.horizon}_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for method, by_b in sorted(summary.items(), key=lambda kv: kv[1].get("50", {}).get("mean", 9)):
        print(f"{method:28s} " + "  ".join(f"@{b}: {v['mean']:.4f} [{v['low']:.4f}, {v['high']:.4f}] med {v['median']:.4f}"
                                           for b, v in sorted(by_b.items(), key=lambda kv: int(kv[0]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
