"""AB-R: referee variants (descriptive). CPU, both tables.

The draw of scripts/fidelity.py (E2): one permutation of the configs with seed
2021, the first 150 scored, the training set taken from the configs after them
(so a larger training set contains the smaller ones). Variants: features
(config fields only, curves only, both), model (gradient-boosted trees as in
E2, ridge, random forest), training size (100, 300, 1,000), k in 1 to 3. Each
predicts the log of the seed-2021 final value and is scored by Kendall tau
against the multi-seed mean, with a percentile bootstrap interval over the 150
configs. On ETTh1 the mean is one seed for all but 100 configs. No variant
replaces the E2 referee or feeds E5.

    .venv/bin/python scripts/ablations/referee.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, SETTINGS, load_table, provenance, setting, write_json  # noqa: E402

from fidelity import DRAW_SEED, N_EVAL  # noqa: E402
from leaguetst.search.replay import Space, Table, _features  # noqa: E402

TABLE_SEED = 2021
SIZES, KS = (100, 300, 1000), (1, 2, 3)
FEATURES, MODELS = ("config", "curves", "both"), ("gbt", "ridge", "forest")
N_BOOT = 2000


def make_model(name: str):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "gbt":
        return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    if name == "forest":
        return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=1)
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def draw(table: Table) -> tuple[list[str], list[str]]:
    ids = [c for c in table.ids if TABLE_SEED in table.obs[c]]
    order = np.random.default_rng(DRAW_SEED).permutation(len(ids))
    return [ids[i] for i in order[:N_EVAL]], [ids[i] for i in order[N_EVAL:N_EVAL + max(SIZES)]]


def features(table: Table, space: Space, cid: str, k: int, kind: str) -> list[float]:
    obs = table.obs[cid][TABLE_SEED]
    row = _features(space, cid, obs.val_curve[:k], obs.train_curve[:k], k)
    n_fields = len(row) - 2 * k
    return {"config": row[:n_fields], "curves": row[n_fields:], "both": row}[kind]


def tau_interval(predicted: np.ndarray, true: np.ndarray, rng: np.random.Generator) -> dict:
    from scipy.stats import kendalltau

    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(true), len(true))
        boots.append(kendalltau(predicted[i], true[i])[0])
    return {"tau": float(kendalltau(predicted, true)[0]), "low": float(np.nanquantile(boots, 0.025)),
            "high": float(np.nanquantile(boots, 0.975))}


def main() -> int:
    from scipy.stats import kendalltau

    result = {"analysis": "AB-R", "descriptive": True, "provenance": provenance(__file__, ["results/ledger.jsonl"]), "tables": {}}
    for dataset, _ in SETTINGS:
        table, space = load_table(dataset)
        chosen_k = json.loads((ROOT / "results" / "fidelity" / f"{setting(dataset)}.json").read_text())["chosen_k"]
        evaluate, train = draw(table)
        true = np.array([table.true_value(c) for c in evaluate])
        final = np.array([table.obs[c][TABLE_SEED].best for c in evaluate])
        target = np.log([table.obs[c][TABLE_SEED].best for c in train])
        rows = []
        for kind in FEATURES:
            for k in (KS if kind != "config" else KS[:1]):  # config-only features do not depend on k
                x_eval = np.array([features(table, space, c, k, kind) for c in evaluate])
                x_train = np.array([features(table, space, c, k, kind) for c in train])
                for size in SIZES:
                    for name in MODELS:
                        model = make_model(name)
                        model.fit(x_train[:size], target[:size])
                        predicted = model.predict(x_eval)
                        rows.append({"features": kind, "k": None if kind == "config" else k, "size": size, "model": name,
                                     **tau_interval(predicted, true, np.random.default_rng(0)),
                                     "tau_vs_seed2021_final": float(kendalltau(predicted, final)[0])})
        plain = {str(k): tau_interval(np.array([min(table.obs[c][TABLE_SEED].val_curve[:k]) for c in evaluate]), true,
                                      np.random.default_rng(0)) for k in KS}
        multi = [i for i, c in enumerate(evaluate) if len(table.obs[c]) >= 3]
        entry = {"n_eval": len(evaluate), "eval_configs_with_3_seeds": len(multi), "variants": rows, "epoch_k_ranking": plain,
                 "one_seed_ceiling": tau_interval(final[multi], true[multi], np.random.default_rng(0)) if len(multi) >= 10 else None,
                 "e2_referee": next(r for r in rows if (r["features"], r["k"], r["size"], r["model"]) == ("both", chosen_k, 300, "gbt"))}
        result["tables"][setting(dataset)] = entry
    write_json(OUT / "AB-R.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
