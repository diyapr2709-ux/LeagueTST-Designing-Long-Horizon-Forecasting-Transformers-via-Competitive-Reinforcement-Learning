"""AB-P and AB-G: the LLM ablation families, for whichever runs exist. CPU.

Two sources, never mixed:

* registered: results/search/ABL_P_* and ABL_G_* (Modal or the RTX, AB1 and AB2).
  Step trend reference: the E4 J2 league arm.
* mps (exploratory, AB3): results/ablations/mps/ABL_*_mps, every arm and every
  reference run on the same Mac.

AB-P on ILI held-out AUC: anon against ref, repair against ref, scale trend;
Holm over 3. AB-G on ILI held-out AUC: the Spearman correlation between the
level rank and AUC over policies, two-sided permutation test, for the step
trend (0, 30, 60, 120) and the KL trend (0, 0.04, 0.1); Holm over 2. A test
whose arms are missing counts as not rejected and keeps the family size. The
hardware check compares the Mac references with the A10 runs of the same
seeds, with no test.

    .venv/bin/python scripts/ablations/llm_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, apply_holm, by_seed, compare, load_table, mean_ci, provenance, read_runs, summarise, \
    write_json  # noqa: E402

N_PERM = 100_000
STEP_LEVELS = (0, 30, 60, 120)


def spearman_trend(levels: dict[int, list[float]], n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Spearman correlation of level rank with value, and a two-sided permutation p (observed counted)."""
    from scipy.stats import rankdata

    order = sorted(levels)
    x = rankdata(np.concatenate([[i] * len(levels[level]) for i, level in enumerate(order)]))
    y = rankdata(np.concatenate([levels[level] for level in order]))
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    observed = float(x @ y / len(x))
    rng = np.random.default_rng(seed)
    hits = sum(abs(float(x @ rng.permutation(y)) / len(x)) >= abs(observed) - 1e-12 for _ in range(n_perm))
    return {"rho": observed, "p": (hits + 1) / (n_perm + 1), "n": len(x),
            "levels": {str(level): mean_ci(levels[level]) for level in order}}


def runs_of(folder: Path) -> list[dict]:
    return [r for path in sorted(folder.glob("*_runs.jsonl")) for r in read_runs(path)]


def aucs(runs: list[dict], table, method: str) -> list[float]:
    return list(by_seed(runs, method, table).values())


def mps_source(table) -> dict:
    root = OUT / "mps"
    ili = {name: runs_of(root / f"ABL_P_{name}_mps" / "ILI_T24") for name in ("ref", "anon", "repair")}
    methods = {"ref": "qwen_prompted_mps", "anon": "qwen_prompted_anon_mps", "repair": "qwen_prompted_repair_mps"}
    values = {name: aucs(ili[name], table, methods[name]) for name in ili}
    p = {f"{arm} vs ref": compare(values[arm], values["ref"], 3) for arm in ("anon", "repair")
         if len(values[arm]) == 30 and len(values["ref"]) == 30}
    out = {"AB-P": {"complete": {n: len(v) for n, v in values.items()}, "tests": apply_holm(p, 3),
                    "not_run": ["scale trend (moved to Soumitra by AB1)"],
                    "diagnostics": {n: diagnostics(ili[n]) for n in ili if ili[n]}}}
    g_runs = runs_of(root / "ABL_G_steps_mps" / "ILI_T24")
    levels = {0: values["ref"]}
    for step in STEP_LEVELS[1:]:
        levels[step] = aucs(g_runs, table, f"grpo_league_from_ETTh1_steps{step}_mps")
    g = {}
    if all(len(levels[s]) == 10 for s in STEP_LEVELS[1:]) and len(levels[0]) == 30:
        g["step trend"] = spearman_trend(levels)
        # Sensitivity (AB3): level 0 is 30 of the 60 runs, so the trend is repeated on the first 10 seeds of each level.
        out["step_trend_10_per_level"] = spearman_trend({s: levels[s][:10] for s in STEP_LEVELS})
        out["step_trend_trained_levels_only"] = spearman_trend({s: levels[s] for s in STEP_LEVELS[1:]})
        out["step_pairs"] = {f"{s} vs 0": {k: v for k, v in compare(levels[s], levels[0], 2).items() if k != "p"} for s in STEP_LEVELS[1:]}
    out["AB-G"] = {"complete": {str(s): len(v) for s, v in levels.items()}, "tests": apply_holm(g, 2),
                   "not_run": ["KL trend (moved to Soumitra by AB1)"], "training": training(root / "ABL_G_steps_mps" / "ILI_T24")}
    return out


def noise_free_prompted() -> dict | None:
    """AB-Q beside the CPU methods: prompted Qwen on the table of 3-seed means (Mac)."""
    from plateau import curve, noise_free_rows
    from common import no_incumbent_value, runner

    runs = runs_of(OUT / "mps" / "ABL_Q_prompted_mps" / "ILI_T24")
    if not runs:
        return None
    from leaguetst.search.replay import Table

    free = Table(noise_free_rows(runner.table_rows("ILI")))
    sigma = load_table("ILI")[0].seed_sd()
    fallback = no_incumbent_value(free)
    values = [float(curve(r, free, fallback).mean() / sigma) for r in runs]
    return {"n": len(values), "auc": mean_ci(values)}


def diagnostics(runs: list[dict]) -> dict:
    total = sum(r["proposals"] for r in runs)
    return {"searches": len(runs), "steps": mean_ci([r["steps"] for r in runs]), "lookups": mean_ci([r["lookups"] for r in runs]),
            "invalid_rate": sum(r["invalid"] for r in runs) / total, "repaired_rate": sum(r["repaired"] for r in runs) / total,
            "repeat_rate": sum(r["repeats"] for r in runs) / total, "seconds": mean_ci([r["seconds"] for r in runs])}


def training(folder: Path) -> dict:
    records = read_runs(folder / "training.jsonl")
    errors = read_runs(folder / "errors.jsonl")
    return {"policies": len(records), "errors": [{"run_names": e["run_names"], "error": e["error"][:160]} for e in errors],
            "onpolicy_mean_max": max((row["onpolicy_mean"] for r in records for row in r["log"]), default=None),
            "unique_rate_mean": float(np.mean([row["unique_rate"] for r in records for row in r["log"]])) if records else None,
            "minutes_per_policy": mean_ci([r["seconds"] / 60 for r in records]) if len(records) > 1 else None}


def hardware_check(table) -> dict:
    """Mac references against the A10 runs of the same seeds: descriptive, no test."""
    out = {}
    root = OUT / "mps"
    e3 = read_runs(ROOT / "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl")
    ref = runs_of(root / "ABL_P_ref_mps" / "ILI_T24")
    if ref:
        a, b = by_seed(ref, "qwen_prompted_mps", table), by_seed(e3, "qwen_prompted", table)
        both = sorted(set(a) & set(b))
        out["prompted_ILI"] = {"seeds": len(both), "mps": mean_ci([a[s] for s in both]), "a10_E3": mean_ci([b[s] for s in both]),
                               "diff": mean_ci([a[s] - b[s] for s in both])}
    j2 = read_runs(ROOT / "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl")
    steps60 = runs_of(root / "ABL_G_steps_mps" / "ILI_T24")
    a = by_seed(steps60, "grpo_league_from_ETTh1_steps60_mps", table)
    if a:
        b = by_seed(j2, "grpo_league_from_ETTh1", table)
        both = sorted(set(a) & set(b))
        if len(both) > 1:
            out["grpo_league_60_steps"] = {"seeds": len(both), "mps": mean_ci([a[s] for s in both]),
                                           "a10_J2": mean_ci([b[s] for s in both]), "diff": mean_ci([a[s] - b[s] for s in both])}
    return out


def registered_source(table) -> dict:
    """Registered items, if any landed (Modal or RTX); missing ones are listed, not imputed."""
    root = ROOT / "results" / "search"
    items = sorted(p.name for p in root.glob("ABL_*"))
    return {"items_found": items, "note": "none launched by Anirudh on Modal (AB3)" if not items else
            "found; their families are computed by the owner of the items (AB1, AB2)"}


def main() -> int:
    table, _ = load_table("ILI")
    etth1, _ = load_table("ETTh1")
    result = {"exploratory": "the mps block is exploratory (AB1, AB3); it never replaces a registered family",
              "provenance": provenance(__file__, ["results/ledger.jsonl"]), "registered": registered_source(table),
              "mps": mps_source(table), "hardware_check": hardware_check(table)}
    root = OUT / "mps"
    result["mps"]["AB-Q_prompted"] = noise_free_prompted()
    result["mps"]["ETTh1_descriptive"] = {}
    for name in ("ref", "anon", "repair"):
        runs = runs_of(root / f"ABL_P_{name}_mps" / "ETTh1_T336")
        if runs:
            result["mps"]["ETTh1_descriptive"][name] = {"summary": summarise(runs, etth1), "diagnostics": diagnostics(runs)}
    write_json(OUT / "LLM.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
