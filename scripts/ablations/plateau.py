"""AB-Q: noise-free lookups (descriptive). Every CPU method of E3 replayed on ILI with 3-seed means.

The noise-free table has one pseudo-seed per config (AB3): the best value is
the mean of the three seeds' best values, the epochs charged are the mean
epochs run rounded half up, and the curves are seed means (a seed that stopped
earlier carries its last value forward). Regret is scored on the same means,
in the pooled seed SD of the real table. If methods separate here but not on
fresh seeds, seed noise over a plateau of near-equal configs explains the null.

    .venv/bin/python scripts/ablations/plateau.py
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, FULL_RUN_EPOCHS, OUT, ROOT, MAX_EPOCHS, load_table, mean_ci, no_incumbent_value, provenance, \
    read_runs, regret_from_trace, replay, runner, write_json, write_runs  # noqa: E402

from leaguetst.search.replay import METHODS, Space, Table  # noqa: E402

SEEDS = 30


def mean_curve(curves: list[list[float]], length: int) -> list[float]:
    return [float(np.mean([c[min(e, len(c) - 1)] for c in curves])) for e in range(length)]


def noise_free_rows(rows: list[dict]) -> list[dict]:
    """One row per config: the seed means of its best value, epochs and curves."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["config_id"], []).append(row)
    out = []
    for cid, members in sorted(groups.items()):
        epochs = int(np.floor(np.mean([int(r["epochs_run"]) for r in members]) + 0.5))
        out.append({"config_id": cid, "seed": 0, "model_config": members[0]["model_config"], "epochs_run": epochs,
                    "best_val_mse": float(np.mean([float(r["best_val_mse"]) for r in members])),
                    "val_curve": mean_curve([r["val_curve"] for r in members], epochs),
                    "train_curve": mean_curve([r["train_curve"] for r in members], epochs)})
    return out


def build(dataset: str, k: int) -> dict:
    table = Table(noise_free_rows(runner.table_rows(dataset)))
    return {"table": table, "space": Space(table.ids), "noise": 0.0,
            "factories": {name: functools.partial(factory, k=k) for name, factory in METHODS.items()}}


def curve(run: dict, table: Table, fallback: float) -> np.ndarray:
    """Regret (raw MSE) on the means at budgets 1 to 100 full-run equivalents."""
    return np.array([regret_from_trace(run["trace"], table, b * FULL_RUN_EPOCHS, fallback)[0] for b in range(1, 101)])


def summary(runs: list[dict], table: Table, sigma: float) -> dict:
    fallback = no_incumbent_value(table)
    out = {}
    for method in sorted({r["method"] for r in runs}):
        curves = np.array([curve(r, table, fallback) for r in runs if r["method"] == method]) / sigma
        out[method] = {"auc": mean_ci(curves.mean(axis=1)), **{str(b): mean_ci(curves[:, b - 1]) for b in BUDGETS},
                       "auc_values": curves.mean(axis=1).tolist()}
    return out


def main() -> int:
    from scipy.stats import mannwhitneyu

    dataset = "ILI"
    k = json.loads((ROOT / "results/fidelity/ILI_T24.json").read_text())["chosen_k"]
    runs = replay(functools.partial(build, dataset, k), list(METHODS), SEEDS)
    write_runs(OUT / "AB-Q" / "ILI_T24.jsonl", runs)
    real, _ = load_table(dataset)
    table = build(dataset, k)["table"]
    sigma = real.seed_sd()
    result = {"analysis": "AB-Q", "descriptive": True, "dataset": dataset, "k": k, "seeds": SEEDS, "sigma": sigma,
              "left_out": "prompted Qwen (GPU); see AB-Q-mps if it was run",
              "provenance": provenance(__file__, ["results/ledger.jsonl", "results/fidelity/ILI_T24.json"]),
              "noise_free": summary(runs, table, sigma)}
    # The same methods with noisy lookups: E3's committed runs, held-out AUC, for the side-by-side table.
    from common import by_seed
    e3 = read_runs(ROOT / "results/search/E3/ILI_T24/cpu_runs.jsonl")
    result["e3_noisy_heldout_auc"] = {m: mean_ci(by_seed(e3, m, real).values()) for m in METHODS}
    base = np.array(result["noise_free"]["random"]["auc_values"])
    e3_base = np.array(list(by_seed(e3, "random", real).values()))
    separation = {}
    for m in METHODS:
        if m == "random":
            continue
        x, y = np.array(result["noise_free"][m]["auc_values"]), np.array(list(by_seed(e3, m, real).values()))
        separation[m] = {"noise_free_diff_vs_random": float(x.mean() - base.mean()),
                         "noise_free_prob_better_than_random": float(mannwhitneyu(base, x).statistic / (len(base) * len(x))),
                         "e3_diff_vs_random": float(y.mean() - e3_base.mean()),
                         "e3_prob_better_than_random": float(mannwhitneyu(e3_base, y).statistic / (len(e3_base) * len(y)))}
    result["separation"] = separation
    plateau = np.sort([table.true_value(c) for c in table.ids])
    result["plateau"] = {"best_mean": float(plateau[0]), "within_1sd": int((plateau <= plateau[0] + sigma).sum()),
                         "within_2sd": int((plateau <= plateau[0] + 2 * sigma).sum()), "configs": len(plateau)}
    write_json(OUT / "AB-Q.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
