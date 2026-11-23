"""Replay search methods on one table at a fixed epoch budget.

Each method runs once per search seed with the largest budget; regret at
every smaller budget is read from its incumbent trace. Writes, under
results/search/<item>/<dataset>_T<horizon>/:

    cpu_runs.jsonl   one line per (method, seed) of the CPU methods: trace, charges
    *_runs.jsonl     the same for LLM methods, written by scripts/modal_llm.py
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
from leaguetst.search.replay import (  # noqa: E402
    FULL_RUN_EPOCHS, METHODS, Space, Table, claim_values, load_runs, no_incumbent_value, regret_from_trace, run_method,
)

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


def run_cpu_methods(table: Table, space: Space, methods: list[str], seeds: int, out_dir: Path) -> None:
    max_budget = BUDGETS[-1] * FULL_RUN_EPOCHS
    path = out_dir / "cpu_runs.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for method in methods:
            start = time.time()
            for seed in range(seeds):
                oracle = run_method(METHODS[method](), table, space, max_budget, seed)
                handle.write(json.dumps({"method": method, "seed": seed, "spent": oracle.spent, "lookups": len(oracle.charged),
                                         "trace": oracle.trace, "charges": [list(c) for c in oracle.charges]}) + "\n")
            print(f"{method:18s} {seeds} seeds in {time.time() - start:.1f}s", flush=True)


def summarize(table: Table, space: Space, out_dir: Path, item: str, table_item: str, dataset: str, horizon: int) -> str:
    sigma = table.seed_sd()
    max_budget = BUDGETS[-1] * FULL_RUN_EPOCHS
    fallback = no_incumbent_value(table)
    grid = list(range(FULL_RUN_EPOCHS, max_budget + 1, FULL_RUN_EPOCHS))
    arch_best: dict[tuple, float] = defaultdict(lambda: float("inf"))
    for cid in table.ids:
        arch = space.architecture(cid)
        arch_best[arch] = min(arch_best[arch], table.true_value(cid))

    runs = load_runs(out_dir)
    budget_runs, by_method = [], defaultdict(list)
    for run in runs:
        trace = [tuple(t) for t in run["trace"]]
        run["regret"], run["incumbent"], run["arch_regret"] = {}, {}, {}
        for b in BUDGETS:
            regret, cid = regret_from_trace(trace, table, b * FULL_RUN_EPOCHS, fallback)
            run["regret"][b], run["incumbent"][b] = regret, cid
            run["arch_regret"][b] = (arch_best[space.architecture(cid)] - table.best_true) if cid else fallback - table.best_true
        run["regret_auc_sd"] = float(np.mean([regret_from_trace(trace, table, g, fallback)[0] for g in grid]) / sigma)
        by_method[run["method"]].append(run)
        budget_runs.append({"method": run["method"], "seed": run["seed"], "spent": run["spent"], "charges": run["charges"]})
    (out_dir / "budget.json").write_text(json.dumps({"table_item": table_item, "dataset": dataset, "horizon": horizon,
                                                     "budget_epochs": max_budget, "runs": budget_runs}) + "\n")

    summary = {"table": {"configs": len(table.ids), "best_true": table.best_true, "best_id": table.best_id, "seed_sd": sigma,
                         "reference_seeds": table.reference_seeds, "diverged_rows": table.n_diverged,
                         "architectures": len(arch_best)}, "methods": {}}
    for method, group in by_method.items():
        entry = {"seeds": len(group)}
        for b in BUDGETS:
            values = np.array([r["regret"][b] for r in group]) / sigma
            low, high = bootstrap_mean(values)
            entry[str(b)] = {"mean_sd": float(values.mean()), "low": low, "high": high, "median_sd": float(np.median(values)),
                             "iqm_sd": iqm(values), "found_best": float(np.mean([r["incumbent"][b] == table.best_id for r in group])),
                             "arch_mean_sd": float(np.mean([r["arch_regret"][b] for r in group]) / sigma)}
        entry["auc_sd"] = float(np.mean([r["regret_auc_sd"] for r in group]))
        summary["methods"][method] = entry
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [f"# {item} replay on {table_item} {dataset} T={horizon}", "",
             f"{len(table.ids)} configs, {len(arch_best)} architectures, seed SD {sigma:.4g}.",
             "Regret of the incumbent in seed-SD units: mean over search seeds with a 95% bootstrap interval. "
             "Budgets in full-run equivalents of 30 epochs. AUC is the mean regret over budgets 1 to 100.", "",
             "| method | seeds | " + " | ".join(f"@{b}" for b in BUDGETS) + " | AUC | found best @50 |",
             "|---|---|" + "---|" * (len(BUDGETS) + 2)]
    for method, entry in sorted(summary["methods"].items(), key=lambda kv: kv[1]["auc_sd"]):
        cells = [f"{entry[str(b)]['mean_sd']:.2f} [{entry[str(b)]['low']:.2f}, {entry[str(b)]['high']:.2f}]" for b in BUDGETS]
        lines.append(f"| {method} | {entry['seeds']} | " + " | ".join(cells) + f" | {entry['auc_sd']:.2f} | {entry['50']['found_best']:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    claims = []
    if "random" in by_method:
        for method, group in by_method.items():
            if method == "random":
                continue
            for b in HEADLINE:
                a, r_ = claim_values(table, out_dir, method, "random", b)
                claims.append({"name": f"{dataset} T{horizon} {method} vs random @{b}", "a_values": a, "b_values": r_,
                               "p": permutation_test(np.array(a), np.array(r_))["p"], "lower_is_better": True,
                               "source": {"runs_dir": str(out_dir.relative_to(ROOT)), "method": method, "baseline": "random",
                                          "budget": b, "table_item": table_item, "dataset": dataset, "horizon": horizon}})
        claims_path = ROOT / "docs" / "claims" / f"{item}.json"
        existing = json.loads(claims_path.read_text())["claims"] if claims_path.exists() else []
        merged = [c for c in existing if not c["name"].startswith(f"{dataset} T{horizon} ")] + claims
        rejected = holm({c["name"]: c["p"] for c in merged})
        for claim in merged:
            claim["claims_win"] = bool(rejected[claim["name"]] and np.mean(claim["a_values"]) < np.mean(claim["b_values"]))
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        claims_path.write_text(json.dumps({"claims": merged}, indent=1) + "\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item", default="E3")
    parser.add_argument("--table", default="E1")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--summarize-only", action="store_true", help="recompute summaries from the runs files")
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
    out_dir = Path(args.out) / args.item / f"{args.dataset}_T{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        run_cpu_methods(table, space, args.methods, args.seeds, out_dir)
    print(summarize(table, space, out_dir, args.item, args.table, args.dataset, args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
