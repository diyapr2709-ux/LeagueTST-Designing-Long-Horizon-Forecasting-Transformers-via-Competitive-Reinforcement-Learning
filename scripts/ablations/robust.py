"""ST, robustness part (descriptive). CPU.

* Regret of the E3 runs in raw MSE beside the pooled seed-SD unit.
* The ETTh1 replay with a robust seed SD (AB3): d is the difference of a
  config's seed-2022 and seed-2023 values (seed 2021 selected those configs and
  is left out, as A20 does); the SD is 1.4826 x median|d| / sqrt(2). It replaces
  the pooled SD both as the observation noise of the lookups and as the unit.
  Every CPU method of E3, 30 seeds.
* Interquartile mean and probability of improvement over random search at every
  budget 1 to 100, with 95% bootstrap intervals; search seeds are resampled
  within each arm (the strata).

    .venv/bin/python scripts/ablations/robust.py
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, FULL_RUN_EPOCHS, OUT, ROOT, SETTINGS, load_table, mean_ci, no_incumbent_value, \
    provenance, read_runs, regret_from_trace, replay, setting, summarise, write_json, write_runs  # noqa: E402

from leaguetst.search.replay import METHODS, Table  # noqa: E402

SEEDS, N_BOOT = 30, 2000


def mad_sd(table: Table, skip: int = 2021) -> float:
    diffs = []
    for seeds in table.obs.values():
        values = [o.best for s, o in sorted(seeds.items()) if s != skip]
        if len(values) == 2:
            diffs.append(values[0] - values[1])
    return float(1.4826 * np.median(np.abs(diffs)) / np.sqrt(2))


def build(dataset: str, k: int) -> dict:
    table, space = load_table(dataset)
    return {"table": table, "space": space, "noise": mad_sd(table),
            "factories": {name: functools.partial(factory, k=k) for name, factory in METHODS.items()}}


def curves(runs: list[dict], table: Table) -> dict[str, np.ndarray]:
    """Per method, (seeds, 100) regret in raw MSE at budgets 1 to 100 (held out where the table allows it)."""
    fallback, out = no_incumbent_value(table), {}
    for run in runs:
        out.setdefault(run["method"], []).append(
            [regret_from_trace(run["trace"], table, b * FULL_RUN_EPOCHS, fallback, int(run["seed"]))[0] for b in range(1, 101)])
    return {m: np.array(v) for m, v in out.items()}


def iqm(x: np.ndarray) -> np.ndarray:
    x = np.sort(x, axis=0)
    n = len(x)
    return x[n // 4: n - n // 4].mean(axis=0)


def prob_better(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    """P(method's regret < random's) + half the ties, per budget."""
    return np.array([((x[:, None, j] < base[None, :, j]).mean() + 0.5 * (x[:, None, j] == base[None, :, j]).mean())
                     for j in range(x.shape[1])])


def profile(x: np.ndarray, base: np.ndarray, sigma: float) -> dict:
    rng = np.random.default_rng(0)
    b_iqm, b_pi = [], []
    for _ in range(N_BOOT):
        xs, bs = x[rng.integers(0, len(x), len(x))], base[rng.integers(0, len(base), len(base))]
        b_iqm.append(iqm(xs) / sigma)
        b_pi.append(prob_better(xs, bs))
    q = lambda a, p: np.quantile(np.array(a), p, axis=0)
    point_iqm, point_pi = iqm(x) / sigma, prob_better(x, base)
    pick = lambda arr: {str(b): float(arr[b - 1]) for b in BUDGETS}
    return {"iqm_sd": pick(point_iqm), "iqm_low": pick(q(b_iqm, 0.025)), "iqm_high": pick(q(b_iqm, 0.975)),
            "pi": pick(point_pi), "pi_low": pick(q(b_pi, 0.025)), "pi_high": pick(q(b_pi, 0.975)),
            "iqm_curve": point_iqm.tolist(), "pi_curve": point_pi.tolist(),
            "pi_curve_low": q(b_pi, 0.025).tolist(), "pi_curve_high": q(b_pi, 0.975).tolist()}


def main() -> int:
    result = {"analysis": "ST robust", "descriptive": True, "tables": {},
              "provenance": provenance(__file__, ["results/ledger.jsonl", "results/search/E3/ILI_T24/cpu_runs.jsonl",
                                                  "results/search/E3/ETTh1_T336/cpu_runs.jsonl"])}
    for dataset, _ in SETTINGS:
        table, _space = load_table(dataset)
        folder = ROOT / "results" / "search" / "E3" / setting(dataset)
        runs = read_runs(folder / "cpu_runs.jsonl") + read_runs(folder / "qwen_prompted_runs.jsonl")
        sigma = table.seed_sd()
        per = curves(runs, table)
        entry = {"pooled_sd": sigma, "summary": summarise(runs, table),
                 "profiles": {m: profile(x, per["random"], sigma) for m, x in per.items()}}
        if dataset == "ETTh1":
            k = json.loads((ROOT / "results/fidelity/ETTh1_T336.json").read_text())["chosen_k"]
            robust = mad_sd(table)
            rerun = replay(functools.partial(build, dataset, k), list(METHODS), SEEDS)
            write_runs(OUT / "ST" / "ETTh1_T336_mad.jsonl", rerun)
            again = curves(rerun, table)
            base = again["random"]
            entry["mad"] = {"sd": robust, "ratio_to_pooled": robust / sigma, "k": k,
                            "auc_sd": {m: mean_ci(x.mean(axis=1) / robust) for m, x in again.items()},
                            "auc_raw_mse": {m: mean_ci(x.mean(axis=1)) for m, x in again.items()},
                            "profiles": {m: profile(x, base, robust) for m, x in again.items()}}
        result["tables"][setting(dataset)] = entry
    write_json(OUT / "ST-robust.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
