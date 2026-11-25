"""AB-N: how much one seed per config biases a replay (descriptive). CPU, ILI.

Two one-seed versions of the ILI table (AB3), searched by every CPU method of
E3 at 30 seeds:

* etth1_like (the ETTh1 rule): seed 2021 for every config, seeds 2022 and 2023
  kept only for the 100 best configs by the seed-2021 value; lookups of one-seed
  configs get a fixed Gaussian offset with the pooled seed SD of the multi-seed
  configs, seed 2021 left out (leaguetst.search.replay, A18 and A20);
* strict: seed 2021 only; the offset's SD is the full ILI table's pooled SD,
  because a one-seed table has no SD of its own.

Every run is scored twice in the full table's pooled SD: as the cut table would
report it (in-sample where a config has one seed, as on ETTh1), and held out on
the full table (the config's seeds that the run did not see, minus the full
table's best 3-seed mean). Optimism = held-out AUC minus in-sample AUC.

    .venv/bin/python scripts/ablations/one_seed_bias.py
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, FULL_RUN_EPOCHS, OUT, ROOT, by_seed, load_table, mean_ci, no_incumbent_value, \
    observation_noise, provenance, read_runs, regret_from_trace, replay, runner, write_json, write_runs  # noqa: E402

from leaguetst.search.replay import METHODS, Space, Table, drawn_seed  # noqa: E402

SEEDS, CUT_SEED, TOP = 30, 2021, 100
VARIANTS = ("etth1_like", "strict")


def cut_rows(rows: list[dict], variant: str) -> list[dict]:
    first = [r for r in rows if int(r["seed"]) == CUT_SEED]
    if variant == "strict":
        return first
    top = {r["config_id"] for r in sorted(first, key=lambda r: r["best_val_mse"])[:TOP]}
    return first + [r for r in rows if int(r["seed"]) != CUT_SEED and r["config_id"] in top]


def noise_for(table: Table, full: Table, variant: str) -> float:
    return full.seed_sd() if variant == "strict" else observation_noise(table)


def build(dataset: str, variant: str, k: int) -> dict:
    rows = runner.table_rows(dataset)
    table, full = Table(cut_rows(rows, variant)), Table(rows)
    return {"table": table, "space": Space(table.ids), "noise": noise_for(table, full, variant),
            "factories": {name: functools.partial(factory, k=k) for name, factory in METHODS.items()}}


def heldout_on_full(cid: str, run_seed: int, cut: Table, full: Table) -> float:
    """Mean over the full table's seeds the run did not see for ``cid`` when it searched ``cut``."""
    seen = drawn_seed(sorted(cut.obs[cid]), run_seed, cid)
    return float(np.mean([o.best for s, o in full.obs[cid].items() if s != seen]))


def score(run: dict, cut: Table, full: Table, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Regret curves over budgets 1 to 100 in SD units: as reported on the cut table, and held out on the full one."""
    seed = int(run["seed"])
    cut_fallback, full_fallback = no_incumbent_value(cut), no_incumbent_value(full)
    reported, heldout = [], []
    for b in range(1, 101):
        value, cid = regret_from_trace(run["trace"], cut, b * FULL_RUN_EPOCHS, cut_fallback, seed)
        reported.append(value)
        heldout.append((heldout_on_full(cid, seed, cut, full) if cid else full_fallback) - full.best_true)
    return np.array(reported) / sigma, np.array(heldout) / sigma


def main() -> int:
    from scipy.stats import kendalltau, spearmanr

    dataset = "ILI"
    k = json.loads((ROOT / "results/fidelity/ILI_T24.json").read_text())["chosen_k"]
    full, _ = load_table(dataset)
    sigma = full.seed_sd()
    e3 = read_runs(ROOT / "results/search/E3/ILI_T24/cpu_runs.jsonl")
    e3_auc = {m: mean_ci(by_seed(e3, m, full).values()) for m in METHODS}
    result = {"analysis": "AB-N", "descriptive": True, "dataset": dataset, "k": k, "seeds": SEEDS, "sigma": sigma,
              "left_out": "prompted Qwen (GPU)", "e3_three_seed_auc": e3_auc, "variants": {},
              "provenance": provenance(__file__, ["results/ledger.jsonl", "results/fidelity/ILI_T24.json"])}
    for variant in VARIANTS:
        runs = replay(functools.partial(build, dataset, variant, k), list(METHODS), SEEDS)
        write_runs(OUT / "AB-N" / f"ILI_T24_{variant}.jsonl", runs)
        state = build(dataset, variant, k)
        cut = state["table"]
        methods = {}
        for m in METHODS:
            pairs = [score(r, cut, full, sigma) for r in runs if r["method"] == m]
            reported = np.array([p[0] for p in pairs])
            heldout = np.array([p[1] for p in pairs])
            methods[m] = {"reported_auc": mean_ci(reported.mean(axis=1)), "heldout_auc": mean_ci(heldout.mean(axis=1)),
                          "optimism_auc": mean_ci(heldout.mean(axis=1) - reported.mean(axis=1)),
                          "optimism_at": {str(b): mean_ci(heldout[:, b - 1] - reported[:, b - 1]) for b in BUDGETS}}
        order = list(METHODS)
        rep = [methods[m]["reported_auc"]["mean"] for m in order]
        held = [methods[m]["heldout_auc"]["mean"] for m in order]
        three = [e3_auc[m]["mean"] for m in order]
        result["variants"][variant] = {
            "noise_sd": state["noise"], "multi_seed_configs": int(sum(len(s) > 1 for s in cut.obs.values())),
            "best_reported": cut.best_true, "best_full": full.best_true, "best_id_cut": cut.best_id, "best_id_full": full.best_id,
            "methods": methods,
            "method_ranking": {"reported_vs_heldout_kendall": float(kendalltau(rep, held)[0]),
                               "reported_vs_heldout_spearman": float(spearmanr(rep, held)[0]),
                               "reported_vs_e3_kendall": float(kendalltau(rep, three)[0]),
                               "heldout_vs_e3_kendall": float(kendalltau(held, three)[0])}}
    write_json(OUT / "AB-N.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
