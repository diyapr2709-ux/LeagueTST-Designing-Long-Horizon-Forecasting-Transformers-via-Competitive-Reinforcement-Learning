"""Replay search methods on one table at a fixed epoch budget.

Each method runs once per search seed with the largest budget; regret at
every smaller budget is read from its incumbent trace. Writes, under
results/search/<item>/<dataset>_T<horizon>/:

    runs.jsonl    one line per (method, seed): trace, charges, regrets
    budget.json   the charges of every run, which gate G5 recounts
    summary.json  regret per method and budget with bootstrap intervals
    summary.md    the same as a table

and docs/claims/<item>.json with each method against random search at the
headline budgets (seed-level permutation test, Holm over the file), which
gate G4 recomputes.

Example:
    python scripts/replay.py --item E3 --table E1 --dataset ILI --horizon 24 --seeds 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.eval.stats import holm, permutation_test  # noqa: E402
from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import FULL_RUN_EPOCHS, METHODS, Space, Table, regret_at, run_method  # noqa: E402

BUDGETS = (10, 25, 50, 100)
HEADLINE = (25, 50)


def iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values))
    n = len(values)
    return float(values[n // 4 : n - n // 4].mean()) if n >= 4 else float(values.mean())


def bootstrap_mean(values: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item", default="E3")
    parser.add_argument("--table", default="E1")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "search"))
    args = parser.parse_args()

    rows = [r for r in read_rows(args.ledger) if r["item"] == args.table and r["dataset"] == args.dataset
            and int(r["horizon"]) == args.horizon]
    if not rows:
        print("no table rows")
        return 1
    table = Table(rows)
    space = Space(table.ids)
    sigma = table.seed_sd()
    max_budget = BUDGETS[-1] * FULL_RUN_EPOCHS
    fallback = float(np.mean([table.true_value(c) for c in table.ids]))
    grid = list(range(FULL_RUN_EPOCHS, max_budget + 1, FULL_RUN_EPOCHS))
    arch_best: dict[tuple, float] = defaultdict(lambda: float("inf"))
    for cid in table.ids:
        arch = space.architecture(cid)
        arch_best[arch] = min(arch_best[arch], table.true_value(cid))

    out_dir = Path(args.out) / args.item / f"{args.dataset}_T{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, budget_runs = [], []
    for method in args.methods:
        start = time.time()
        for seed in range(args.seeds):
            oracle = run_method(METHODS[method](), table, space, max_budget, seed)
            regrets = {b: regret_at(oracle, b * FULL_RUN_EPOCHS, fallback) for b in BUDGETS}
            curve = [regret_at(oracle, b, fallback) for b in grid]
            incumbents = {b: next((c for s, c in reversed(oracle.trace) if s <= b * FULL_RUN_EPOCHS), None) for b in BUDGETS}
            arch_regret = {b: (arch_best[space.architecture(c)] - table.best_true) if c else fallback - table.best_true
                           for b, c in incumbents.items()}
            runs.append({"method": method, "seed": seed, "spent": oracle.spent, "lookups": len(oracle.charged),
                         "regret": regrets, "arch_regret": arch_regret, "incumbent": incumbents,
                         "regret_auc_sd": float(np.mean(curve) / sigma), "trace": oracle.trace})
            budget_runs.append({"method": method, "seed": seed, "spent": oracle.spent,
                                "charges": [list(c) for c in oracle.charges]})
        print(f"{method:18s} {args.seeds} seeds in {time.time() - start:.1f}s", flush=True)

    with open(out_dir / "runs.jsonl", "w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    (out_dir / "budget.json").write_text(json.dumps({"table_item": args.table, "dataset": args.dataset, "horizon": args.horizon,
                                                     "budget_epochs": max_budget, "runs": budget_runs}) + "\n")

    by_method: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_method[run["method"]].append(run)
    summary = {"table": {"configs": len(table.ids), "best_true": table.best_true, "best_id": table.best_id, "seed_sd": sigma,
                         "architectures": len(arch_best)}, "methods": {}}
    for method, group in by_method.items():
        entry = {}
        for b in BUDGETS:
            values = np.array([r["regret"][b] for r in group]) / sigma
            low, high = bootstrap_mean(values)
            entry[str(b)] = {"mean_sd": float(values.mean()), "low": low, "high": high, "median_sd": float(np.median(values)),
                             "iqm_sd": iqm(values), "found_best": float(np.mean([r["incumbent"][b] == table.best_id for r in group])),
                             "arch_mean_sd": float(np.mean([r["arch_regret"][b] for r in group]) / sigma)}
        entry["auc_sd"] = float(np.mean([r["regret_auc_sd"] for r in group]))
        summary["methods"][method] = entry
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [f"# {args.item} replay on {args.table} {args.dataset} T={args.horizon}", "",
             f"{len(table.ids)} configs, {len(arch_best)} architectures, seed SD {sigma:.4g}, {args.seeds} search seeds per method.",
             "Regret is in seed-SD units (mean with 95% bootstrap interval over search seeds).", "",
             "| method | " + " | ".join(f"regret @{b}" for b in BUDGETS) + " | AUC |", "|---|" + "---|" * (len(BUDGETS) + 1)]
    for method, entry in summary["methods"].items():
        cells = [f"{entry[str(b)]['mean_sd']:.2f} [{entry[str(b)]['low']:.2f}, {entry[str(b)]['high']:.2f}]" for b in BUDGETS]
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {entry['auc_sd']:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    claims = []
    if "random" in by_method:
        for method, group in by_method.items():
            if method == "random":
                continue
            for b in HEADLINE:
                a = [r["regret"][b] / sigma for r in group]
                r_ = [r["regret"][b] / sigma for r in by_method["random"]]
                claims.append({"name": f"{args.dataset} T{args.horizon} {method} vs random @{b}", "a_values": a, "b_values": r_,
                               "p": permutation_test(np.array(a), np.array(r_))["p"]})
        rejected = holm({c["name"]: c["p"] for c in claims})
        for claim in claims:
            claim["claims_win"] = bool(rejected[claim["name"]] and np.mean(claim["a_values"]) < np.mean(claim["b_values"]))
        claims_path = ROOT / "docs" / "claims" / f"{args.item}.json"
        existing = json.loads(claims_path.read_text())["claims"] if claims_path.exists() else []
        keep = [c for c in existing if not c["name"].startswith(f"{args.dataset} T{args.horizon} ")]
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        claims_path.write_text(json.dumps({"claims": keep + claims}, indent=1) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
