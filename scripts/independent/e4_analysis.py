"""Independent E4 analysis (docs/LEG2_soumitra.md L1), written from docs/prereg_E4.md and docs/PLAN.md alone.

Imports nothing from leaguetst/ or scripts/. Reads results/ledger.jsonl (E1 rows: the tables) and the
search runs files directly, and writes results/independent/E4_soumitra.json. It never writes the ledger.

Definitions, each from the registration it cites:

* Table (PLAN A2): E1 rows, one best_val_mse per (dataset, config, seed). ILI has seeds 2021 to 2023 for
  every config; ETTh1 has seed 2021 for every config and 2022, 2023 for its top 100.
* Pooled seed SD (PLAN A18; prereg A20): sqrt of the mean within-config sample variance (ddof 1) over the
  multi-seed configs. On ETTh1 seed 2021 is left out (its multi-seed configs were selected on it).
* What a search saw (prereg "Lookups"): every charge [config, seed, epochs, kind, ...] is one lookup of
  that config at that seed; the observed value is the table entry at that seed. Repeats cost nothing and
  are not charged. Only full-run charges occur in the E4 and compared runs; anything else is refused.
* Incumbent: the config with the lowest observed value so far (first one kept on ties). Checked against
  every record's own trace ([cumulative epochs, incumbent]) and refused on any mismatch.
* Held-out regret (PLAN A2, A18): the incumbent's mean over the table seeds the search never saw for it,
  minus the best config's mean over all its seeds, in pooled seed-SD units. ILI only: on ETTh1 a
  held-out seed exists for the top 100 alone, so ETTh1 claims come from the pick reseeds (A18, A25).
* Anytime regret and AUC (PLAN A3): budgets are counted in full-run equivalents of 30 epochs. Regret at
  budget b uses the incumbent among lookups whose cumulative spend is at most 30 b. AUC is the mean of
  regret(b) over b = 1 .. 100 (every config costs at most 30 epochs, so an incumbent exists at b = 1).
  Headline budgets 25 and 50.
* Tests (prereg "Endpoints and tests", A20): two-sample, seed-level, two-sided permutation test on the
  difference of means; exact when the number of splits is at most 2e6, otherwise 200,000 random splits
  with a fixed generator, p = (hits + 1) / (n + 1). Holm within each family.
* Families under A25 and A30: family 1 (league against the portfolio and warm-started TPE) and family 2
  (league against none and against uniform) each have six tests: two on ILI AUC (target ILI) and four on
  the ETTh1 picks at budgets 25 and 50 (target ETTh1). Tests that do not exist yet are reported as
  pending and enter Holm as not rejected (p = 1), so no family shrinks (A30, AB1 "missing arms").
* J3 (A30): runs if, on ILI AUC, the league mean is lower than the none mean and the two-sided p is at
  most 0.05 / 6.
* Worst case (A26, A29): failed seeds of the arm under test take the worst value observed in either arm
  of the test, failed seeds of the comparator the best; retried runs count as failed in the worst case.

Usage:  python scripts/independent/e4_analysis.py [--out results/independent/E4_soumitra.json]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EQUIV = 30            # epochs per full-run equivalent (PLAN A3)
BUDGET_EQ = 100       # deployment budget, full-run equivalents (prereg "Deployment")
HEADLINE = (25, 50)   # PLAN A3
ALPHA = 0.05
FAMILY_SIZE = 6       # A25 and A30
EXACT_LIMIT = 2_000_000
N_RANDOM = 200_000
RNG_SEED = 20261125

# Runs registered as retried under A29 (J1 attempt 1, named in the J1 commits); all target ETTh1.
RETRIED = {"grpo_league_from_ILI": [9], "grpo_none_from_ILI": [8]}


# ---------------------------------------------------------------- tables

def load_tables(ledger: Path) -> dict:
    """{dataset: {config: {seed: best_val_mse}}} from E1 rows."""
    tables: dict = {}
    with open(ledger) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("item") != "E1":
                continue
            cell = tables.setdefault(row["dataset"], {}).setdefault(row["config_id"], {})
            if row["seed"] in cell:
                raise ValueError(f"duplicate E1 row {row['dataset']} {row['config_id']} {row['seed']}")
            if row.get("diverged"):
                raise ValueError(f"diverged E1 row {row['row_id']}")
            cell[row["seed"]] = float(row["best_val_mse"])
    return tables


def pooled_sd(table: dict, leave_out: tuple = ()) -> tuple[float, int]:
    variances = []
    for seeds in table.values():
        values = [v for s, v in seeds.items() if s not in leave_out]
        if len(values) >= 2:
            variances.append(np.var(values, ddof=1))
    if not variances:
        raise ValueError("no multi-seed configs")
    return float(math.sqrt(np.mean(variances))), len(variances)


# ---------------------------------------------------------------- one search

def incumbent_path(record: dict, table: dict) -> list[tuple[int, str, int]]:
    """[(cumulative epochs, incumbent config, seed it was seen at)] after each lookup, checked against the trace."""
    spent, best_val, best = 0, math.inf, None
    path = []
    for charge in record["charges"]:
        config, seed, epochs, kind = charge[0], int(charge[1]), int(charge[2]), charge[3]
        if kind != "full":
            raise ValueError(f"{record['method']} seed {record['seed']}: non-full charge {charge}")
        spent += epochs
        value = table[config][seed]
        if value < best_val:
            best_val, best = value, (config, seed)
        path.append((spent, best[0], best[1]))
    trace = record["trace"]
    if len(trace) != len(path):
        raise ValueError(f"{record['method']} seed {record['seed']}: {len(trace)} trace rows, {len(path)} charges")
    for (spent, config, _), (t_spent, t_config) in zip(path, trace):
        if int(t_spent) != spent or t_config != config:
            raise ValueError(f"{record['method']} seed {record['seed']}: trace {t_spent},{t_config} "
                             f"against {spent},{config}")
    if spent != int(record["spent"]) or spent > BUDGET_EQ * EQUIV:
        raise ValueError(f"{record['method']} seed {record['seed']}: spent {spent} vs {record['spent']}")
    return path


def regret_curve(record: dict, table: dict, best_mean: float, sd: float) -> list[float]:
    """Held-out regret at b = 1 .. 100 full-run equivalents, in pooled seed-SD units."""
    path = incumbent_path(record, table)
    curve, i, current = [], 0, None
    for b in range(1, BUDGET_EQ + 1):
        while i < len(path) and path[i][0] <= b * EQUIV:
            current = path[i]
            i += 1
        if current is None:
            raise ValueError(f"{record['method']} seed {record['seed']}: no incumbent at budget {b}")
        _, config, seen = current
        held_out = [v for s, v in table[config].items() if s != seen]
        if not held_out:
            raise ValueError(f"{config}: no held-out seed")
        curve.append((float(np.mean(held_out)) - best_mean) / sd)
    return curve


def score_runs(path: Path, table: dict, best_mean: float, sd: float, methods: set | None = None) -> dict:
    """{method: {seed: {"auc", "regret_25", "regret_50"}}} for the records of a runs file (``methods`` only)."""
    out: dict = {}
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            if methods is not None and record["method"] not in methods:
                continue
            if float(record.get("noise_sd", 0.0)) != 0.0:
                raise ValueError(f"{path}: noise_sd {record['noise_sd']} on a held-out table")
            curve = regret_curve(record, table, best_mean, sd)
            seed = int(record["seed"])
            cell = out.setdefault(record["method"], {})
            if seed in cell:
                raise ValueError(f"{path}: duplicate {record['method']} seed {seed}")
            cell[seed] = {"auc": float(np.mean(curve)),
                          **{f"regret_{b}": curve[b - 1] for b in HEADLINE}}
    return out


# ---------------------------------------------------------------- tests

def permutation_p(a: list[float], b: list[float]) -> dict:
    """Two-sided two-sample permutation test on the difference of means (a minus b)."""
    x = np.asarray(a + b, dtype=float)
    n_a, n = len(a), len(a) + len(b)
    observed = float(np.mean(a) - np.mean(b))
    total = x.sum()
    tol = 1e-12 * max(1.0, abs(observed))
    n_splits = math.comb(n, n_a)
    if n_splits <= EXACT_LIMIT:
        hits = 0
        for idx in itertools.combinations(range(n), n_a):
            s = x[list(idx)].sum()
            diff = s / n_a - (total - s) / (n - n_a)
            hits += abs(diff) >= abs(observed) - tol
        return {"diff": observed, "p": hits / n_splits, "method": "exact", "splits": n_splits}
    rng = np.random.default_rng(RNG_SEED)
    hits = 0
    for _ in range(N_RANDOM):
        perm = rng.permutation(n)
        s = x[perm[:n_a]].sum()
        diff = s / n_a - (total - s) / (n - n_a)
        hits += abs(diff) >= abs(observed) - tol
    return {"diff": observed, "p": (hits + 1) / (N_RANDOM + 1), "method": "random", "splits": N_RANDOM}


def holm(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down; returns adjusted p and rejection per test."""
    order = sorted(p_values, key=lambda k: p_values[k])
    m = len(order)
    out, running, stop = {}, 0.0, False
    for rank, name in enumerate(order):
        adjusted = min(1.0, (m - rank) * p_values[name])
        running = max(running, adjusted)
        reject = (not stop) and p_values[name] <= alpha / (m - rank)
        if not reject:
            stop = True
        out[name] = {"p": p_values[name], "p_holm": running, "reject": reject}
    return out


def worst_case(test_vals: dict, comp_vals: dict, test_failed: list, comp_failed: list, key: str) -> dict:
    """A26: failed seeds of the arm under test take the worst observed value, the comparator's the best."""
    pool = [v[key] for v in test_vals.values()] + [v[key] for v in comp_vals.values()]
    worst, best = max(pool), min(pool)   # regret: higher is worse
    a = [v[key] for s, v in test_vals.items() if s not in test_failed] + [worst] * len(test_failed)
    b = [v[key] for s, v in comp_vals.items() if s not in comp_failed] + [best] * len(comp_failed)
    return permutation_p(a, b)


def compare(scores: dict, test: str, comp: str, key: str, expected: tuple[int, int]) -> dict:
    a_runs, b_runs = scores.get(test, {}), scores.get(comp, {})
    a = [a_runs[s][key] for s in sorted(a_runs)]
    b = [b_runs[s][key] for s in sorted(b_runs)]
    result = {"test": test, "comparator": comp, "endpoint": key, "n": [len(a), len(b)],
              "expected_n": list(expected), "mean": [float(np.mean(a)), float(np.mean(b))],
              "median": [float(np.median(a)), float(np.median(b))], **permutation_p(a, b)}
    failed_a = [s for s in range(expected[0]) if s not in a_runs] + RETRIED.get(test, [])
    failed_b = [s for s in range(expected[1]) if s not in b_runs] + RETRIED.get(comp, [])
    result["failed_or_retried"] = [sorted(set(failed_a)), sorted(set(failed_b))]
    if failed_a or failed_b:
        result["worst_case"] = worst_case(a_runs, b_runs, sorted(set(failed_a)), sorted(set(failed_b)), key)
    return result


# ---------------------------------------------------------------- main

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/independent/E4_soumitra.json")
    args = parser.parse_args()

    ledger = ROOT / "results/ledger.jsonl"
    tables = load_tables(ledger)
    ili = tables["ILI"]
    if len(ili) != 2368 or any(set(s) != {2021, 2022, 2023} for s in ili.values()):
        raise ValueError("ILI table is not 2,368 configs at seeds 2021 to 2023")
    sd_ili, n_sd = pooled_sd(ili)
    means = {c: float(np.mean(list(s.values()))) for c, s in ili.items()}
    best_config = min(means, key=means.get)
    best_mean = means[best_config]
    sd_etth1, n_sd_etth1 = pooled_sd(tables["ETTh1"], leave_out=(2021,))

    inputs = {
        "e4_league": ROOT / "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl",
        "e4_none": ROOT / "results/search/E4/ILI_T24/grpo_none_from_ETTh1_runs.jsonl",
        "e4_controls": ROOT / "results/search/E4/ILI_T24/transfer_from_ETTh1_runs.jsonl",
        "e3_cpu": ROOT / "results/search/E3/ILI_T24/cpu_runs.jsonl",
        "e3_prompted": ROOT / "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl",
    }
    scores: dict = {}
    for name, path in inputs.items():
        # from the E3 CPU runs only the full-run methods family 3 compares with (screening charges partial epochs)
        methods = {"random", "tpe"} if name == "e3_cpu" else None
        scores.update(score_runs(path, ili, best_mean, sd_ili, methods))

    league, none = "grpo_league_from_ETTh1", "grpo_none_from_ETTh1"
    fam1_ili = {
        "F1 ILI AUC league vs portfolio": compare(scores, league, "portfolio_from_ETTh1", "auc", (10, 30)),
        "F1 ILI AUC league vs tpe_warm": compare(scores, league, "tpe_warm_from_ETTh1", "auc", (10, 30)),
    }
    fam2_ili = {"F2 ILI AUC league vs none": compare(scores, league, none, "auc", (10, 10))}
    pending_f1 = [f"F1 ETTh1 picks b{b} league_from_ILI vs {c}" for c in ("portfolio", "tpe_warm") for b in HEADLINE]
    pending_f2 = [f"F2 ETTh1 picks b{b} league_from_ILI vs {c}" for c in ("none", "uniform") for b in HEADLINE]

    j3 = fam2_ili["F2 ILI AUC league vs none"]
    j3_runs = j3["mean"][0] < j3["mean"][1] and j3["p"] <= ALPHA / FAMILY_SIZE
    # A30: if J3 is not run, the ILI league-against-uniform test counts as not rejected.
    fam2_uniform = "F2 ILI AUC league vs uniform"

    def family(observed: dict, pending: list, extra_not_run: list) -> dict:
        p = {k: v["p"] for k, v in observed.items()}
        p.update({k: 1.0 for k in pending + extra_not_run})
        if len(p) != FAMILY_SIZE:
            raise ValueError(f"family has {len(p)} tests, not {FAMILY_SIZE}")
        return {"tests": observed, "pending_or_not_run": pending + extra_not_run,
                "holm_with_pending_as_not_rejected": holm(p)}

    fam3 = {}
    for comp in ("random", "tpe"):
        for b in HEADLINE:
            fam3[f"F3 ILI b{b} qwen_prompted vs {comp}"] = compare(scores, "qwen_prompted", comp, f"regret_{b}",
                                                                   (10, 30))

    descriptive = {m: {"n": len(v), "auc_mean": float(np.mean([r["auc"] for r in v.values()])),
                       "auc_median": float(np.median([r["auc"] for r in v.values()])),
                       **{f"regret_{b}_mean": float(np.mean([r[f"regret_{b}"] for r in v.values()]))
                          for b in HEADLINE}}
                   for m, v in sorted(scores.items())}

    out = {
        "script": "scripts/independent/e4_analysis.py",
        "script_sha256": sha256(Path(__file__)),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "inputs_sha256": {k: sha256(v) for k, v in {"ledger": ledger, **inputs}.items()},
        "table": {"ILI_configs": len(ili), "ILI_pooled_sd": sd_ili, "ILI_pooled_sd_configs": n_sd,
                  "ILI_best_config": best_config, "ILI_best_3seed_mean": best_mean,
                  "ETTh1_pooled_sd_without_2021": sd_etth1, "ETTh1_pooled_sd_configs": n_sd_etth1},
        "definitions": {"equivalent_epochs": EQUIV, "budget_equivalents": BUDGET_EQ, "headline": list(HEADLINE),
                        "auc": "mean held-out regret over b = 1..100", "alpha": ALPHA,
                        "family_size": FAMILY_SIZE, "exact_limit": EXACT_LIMIT, "n_random": N_RANDOM,
                        "rng_seed": RNG_SEED},
        "j3_decision_A30": {"runs": bool(j3_runs), "league_mean_auc": j3["mean"][0], "none_mean_auc": j3["mean"][1],
                            "p": j3["p"], "threshold": ALPHA / FAMILY_SIZE, "test": j3["method"]},
        "family_1": family(fam1_ili, pending_f1, []),
        "family_2": family(fam2_ili, pending_f2, [] if j3_runs else [fam2_uniform]),
        "family_2_note": ("ILI league vs uniform awaits J3" if j3_runs else
                          "J3 not run (A30): ILI league vs uniform counted as not rejected"),
        "family_3_ili_unadjusted": fam3,
        "family_3_note": ("Family 3 is Holm-adjusted together with the E3 CPU methods' comparisons "
                          "(prereg); that family is defined outside docs/prereg_E4.md, so only unadjusted "
                          "ILI p-values are given here. ETTh1 family-3 tests come from the pick reseeds."),
        "pending_note": ("ETTh1-target tests (A25) need pick batch 2 (E4 incumbents at budgets 25 and 50, "
                         "reseeded on Diya's workspace); they are pending until it is on origin/main."),
        "retried_runs_A29": RETRIED,
        "descriptive_ili": descriptive,
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["j3_decision_A30"], indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
