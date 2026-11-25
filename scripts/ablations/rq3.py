"""RQ3 (descriptive): pairwise interactions, cross-table agreement, RevIN validation check. CPU.

* Interactions: per table, least squares on the config means with all eight
  fields as factors; for each of the 28 field pairs, the extra share of the
  total variance explained by adding the pair's interaction to the additive
  model (the architecture fields do not form a full factorial, so shares are
  increments, not an orthogonal decomposition). Seed noise is given where a
  table has several seeds for every config.
* Cross-table: each field's main-effect share per table, the rank agreement of
  the shares, and per field the agreement of the level means.
* RevIN: the 20 twin pairs retrained for 100 epochs (item REVIN, seed 2021)
  against the same pairs in E1 at 30 epochs, overall and by validation month,
  beside results/revin_prediction.json. Validation errors only: the per-window
  files are those of the validation split. The test part is Diya's.

    .venv/bin/python scripts/ablations/rq3.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, SETTINGS, load_table, mean_ci, provenance, setting, write_json  # noqa: E402

from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import MUTABLE_FIELDS, Space, Table  # noqa: E402

MONTH = 720  # ETTh1 windows per validation month (results/revin_prediction.json)


def design(space: Space, ids: list[str], fields: tuple[str, ...]) -> np.ndarray:
    """One-hot columns of the joint levels of ``fields``."""
    keys = [tuple(str(space.choices[c][f]) for f in fields) for c in ids]
    levels = {k: i for i, k in enumerate(sorted(set(keys)))}
    x = np.zeros((len(ids), len(levels)))
    for r, k in enumerate(keys):
        x[r, levels[k]] = 1.0
    return x


def explained(x: np.ndarray, y: np.ndarray) -> float:
    """Share of the variance of ``y`` explained by least squares on ``x`` (with an intercept)."""
    x = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(1 - ((y - x @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def interactions(table: Table, space: Space) -> dict:
    ids = table.ids
    y = np.array([table.true_value(c) for c in ids])
    main = {f: design(space, ids, (f,)) for f in MUTABLE_FIELDS}
    additive = np.column_stack(list(main.values()))
    base = explained(additive, y)
    pairs = {}
    for a, b in itertools.combinations(MUTABLE_FIELDS, 2):
        pairs[f"{a} x {b}"] = explained(np.column_stack([additive, design(space, ids, (a, b))]), y) - base
    conditional = {f: base - explained(np.column_stack([x for g, x in main.items() if g != f]), y) for f in MUTABLE_FIELDS}
    by_seed = {}
    seed_counts = {len(s) for s in table.obs.values()}
    if len(seed_counts) == 1 and seed_counts.pop() >= 2:
        seeds = sorted(table.obs[ids[0]])
        for s in seeds:
            ys = np.array([table.obs[c][s].best for c in ids])
            b = explained(additive, ys)
            by_seed[str(s)] = {p: explained(np.column_stack([additive, design(space, ids, tuple(p.split(" x ")))]), ys) - b for p in pairs}
        values = np.array([[table.obs[c][s].best for s in seeds] for c in ids])
        within = values.var(axis=1, ddof=1).mean() / len(seeds)
        noise_share = float(within / np.array([table.true_value(c) for c in ids]).var())
    else:
        noise_share = None
    all_pairs = np.column_stack([additive] + [design(space, ids, p) for p in itertools.combinations(MUTABLE_FIELDS, 2)])
    return {"additive_r2": base, "all_pairs_r2": explained(all_pairs, y),
            "main_effect_share": {f: explained(x, y) for f, x in main.items()},
            "conditional_share": conditional, "seed_noise_share_of_config_means": noise_share,
            "interaction_increment": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
            "interaction_increment_by_seed": by_seed or None}


def cross_table(tables: dict[str, Table], spaces: dict[str, Space], shares: dict[str, dict]) -> dict:
    from scipy.stats import kendalltau, spearmanr

    a, b = ("ILI", "ETTh1")
    fields = list(MUTABLE_FIELDS)
    va, vb = [shares[a][f] for f in fields], [shares[b][f] for f in fields]
    out = {"share_spearman": float(spearmanr(va, vb)[0]), "share_kendall": float(kendalltau(va, vb)[0]),
           "field_rank": {d: [f for f in sorted(fields, key=lambda f: -shares[d][f])] for d in (a, b)}, "level_means": {}}
    for f in fields:
        means = {}
        for d in (a, b):
            groups: dict = {}
            for c in tables[d].ids:
                groups.setdefault(spaces[d].choices[c][f], []).append(tables[d].true_value(c))
            means[d] = {str(k): float(np.mean(v)) for k, v in groups.items()}
        levels = sorted(means[a])
        xa, xb = [means[a][k] for k in levels], [means[b][k] for k in levels]
        out["level_means"][f] = {"means": means, "best_level": {d: min(means[d], key=means[d].get) for d in (a, b)},
                                 "same_best_level": min(means[a], key=means[a].get) == min(means[b], key=means[b].get),
                                 "kendall": float(kendalltau(xa, xb)[0]) if len(levels) > 2 else float(np.sign((xa[0] - xa[1]) * (xb[0] - xb[1])))}
    return out


def revin() -> dict:
    rows = {r["config_id"]: r for r in read_rows(ROOT / "results" / "ledger.jsonl") if r["item"] == "REVIN"}
    e1 = {r["config_id"]: r for r in read_rows(ROOT / "results" / "ledger.jsonl")
          if r["item"] == "E1" and r["dataset"] == "ETTh1" and int(r["seed"]) == 2021 and r["config_id"] in rows}
    off = sorted(c for c in rows if c.endswith("n0i"))
    pairs = [(c, c[:-3] + "n1i") for c in off if c[:-3] + "n1i" in rows]

    def window_errors(item: str, cid: str) -> np.ndarray:
        x = np.load(ROOT / "results" / "runs" / item / "ETTh1_T336_s2021" / "perwindow" / f"{cid}.npy")
        return x.reshape(len(x), -1).mean(axis=1)

    out = {"pairs": len(pairs)}
    for name, item, source in (("epochs_30", "E1", e1), ("epochs_100", "REVIN", rows)):
        advantage = np.array([source[on]["best_val_mse"] - source[o]["best_val_mse"] for o, on in pairs])
        months = []
        for o, on in pairs:
            d = window_errors(item, on) - window_errors(item, o)
            months.append([float(d[m * MONTH:(m + 1) * MONTH].mean()) for m in range(int(np.ceil(len(d) / MONTH)))])
        out[name] = {"off_advantage": mean_ci(advantage), "share_pairs_off_better": float((advantage > 0).mean()),
                     "off_advantage_by_month": np.mean(months, axis=0).tolist(),
                     "share_off_better_by_month": (np.array(months) > 0).mean(axis=0).tolist(),
                     "on_configs_best_epoch_at_cap": float(np.mean([source[on]["best_epoch"] >= source[on]["epochs_run"] for _, on in pairs])),
                     "advantage_per_pair": advantage.tolist()}
    a30, a100 = np.array(out["epochs_30"]["advantage_per_pair"]), np.array(out["epochs_100"]["advantage_per_pair"])
    out["caveat"] = "E1 stops by patience 10 (RevIN-on runs there run 21 to 30 epochs); REVIN uses patience 100, so the 30-to-100 comparison mixes epochs and patience"
    out["shrinkage_30_to_100"] = {"mean_change": mean_ci(a100 - a30), "ratio_of_means": float(a100.mean() / a30.mean())}
    out["prediction"] = {k: v for k, v in json.loads((ROOT / "results/revin_prediction.json").read_text()).items()
                         if k in ("pairs", "off_advantage_by_month", "slope_per_month", "prediction", "windows_per_month")}
    return out


def _spearman(va, vb) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(va, vb)[0])


def main() -> int:
    tables, spaces, result = {}, {}, {"analysis": "RQ3", "descriptive": True, "tables": {}}
    for dataset, _ in SETTINGS:
        tables[dataset], spaces[dataset] = load_table(dataset)
        result["tables"][setting(dataset)] = {**interactions(tables[dataset], spaces[dataset]),
                                              "configs": len(tables[dataset].ids)}
    shares = {d: result["tables"][setting(d)]["main_effect_share"] for d in tables}
    result["cross_table"] = cross_table(tables, spaces, shares)
    cond = {d: result["tables"][setting(d)]["conditional_share"] for d in tables}
    result["cross_table"]["conditional_share_spearman"] = _spearman([cond["ILI"][f] for f in MUTABLE_FIELDS], [cond["ETTh1"][f] for f in MUTABLE_FIELDS])
    result["revin"] = revin()
    result["provenance"] = provenance(__file__, ["results/ledger.jsonl", "results/revin_prediction.json"])
    write_json(OUT / "RQ3.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
