"""E5 (RQ2): does the referee's ranking hold on the architect's own proposals?

Amendment A24. The referee is the E2 model: gradient-boosted trees on the
config fields and the seed-2021 curves up to the chosen k, fitted on the same
300 random configs as scripts/fidelity.py. It ranks the distinct configs each
LLM search run proposed (training configs left out). Each run's proposals are
compared with (a) quality-matched random sets: for every proposal, a random
pick among the 5 unproposed configs with the same instance_norm value whose
multi-seed mean is nearest; and (b) plain random sets of the same size.

Per run: Kendall tau of the referee's prediction against the multi-seed mean
(headline) and against the seed-2021 final value, and the top-2 regret (best
multi-seed mean among the referee's top 2 minus the best in the set, in
pooled seed-SD units). Runs are the unit: the mean over runs of proposals
minus each comparison, with a t interval at level 1 - 0.05/k (k methods on
the dataset). The referee is non-inferior on a method's proposals
when that interval for tau_true against matched sets lies above -0.05.

Example:
    python scripts/e5_referee.py --dataset ILI --horizon 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fidelity import DRAW_SEED, N_EVAL, N_TRAIN  # noqa: E402
from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import Space, Table, _features  # noqa: E402

SEED = 2021
N_MATCHED, NEAREST, MARGIN = 200, 5, 0.05
METRICS = ("tau_true", "tau_final", "top2_regret_sd")
CPU_FILES = ("cpu_runs.jsonl",)


def metrics(ids: list[str], predicted: dict[str, float], table: Table, sd: float) -> dict[str, float]:
    pred = [predicted[c] for c in ids]
    final = [table.obs[c][SEED].best for c in ids]
    true = [table.true_value(c) for c in ids]
    top2 = sorted(ids, key=predicted.get)[:2]
    return {"tau_final": float(kendalltau(pred, final)[0]), "tau_true": float(kendalltau(pred, true)[0]),
            "top2_regret_sd": float((min(table.true_value(c) for c in top2) - min(true)) / sd)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    from sklearn.ensemble import GradientBoostingRegressor

    setting = f"{args.dataset}_T{args.horizon}"
    rows = [r for r in read_rows(ROOT / "results" / "ledger.jsonl") if r["item"] == "E1"
            and r["dataset"] == args.dataset and int(r["horizon"]) == args.horizon]
    table = Table(rows)
    space = Space(table.ids)
    sd = table.seed_sd()
    k = json.loads((ROOT / "results" / "fidelity" / f"{setting}.json").read_text())["chosen_k"]
    ids = [c for c in table.ids if SEED in table.obs[c]]
    order = np.random.default_rng(DRAW_SEED).permutation(len(ids))
    train = {ids[i] for i in order[N_EVAL:N_EVAL + N_TRAIN]}

    def feats(cid: str) -> list[float]:
        obs = table.obs[cid][SEED]
        return _features(space, cid, obs.val_curve[:k], obs.train_curve[:k], k)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    fit_ids = sorted(train)
    model.fit(np.array([feats(c) for c in fit_ids]), np.log([table.obs[c][SEED].best for c in fit_ids]))
    pool = [c for c in ids if c not in train]
    predicted = dict(zip(pool, model.predict(np.array([feats(c) for c in pool]))))
    groups: dict = {}
    for c in pool:
        groups.setdefault(space.choices[c]["instance_norm"], []).append(c)
    for members in groups.values():
        members.sort(key=table.true_value)
    pos = {c: i for members in groups.values() for i, c in enumerate(members)}

    def matched_set(proposed: list[str]) -> list[str]:
        taken, pick = set(proposed), set()
        for c in proposed:
            members, i = groups[space.choices[c]["instance_norm"]], pos[c]
            near = [x for x in members[max(0, i - 3 * NEAREST): i + 3 * NEAREST + 1] if x not in taken and x not in pick]
            if not near:  # every neighbour is taken: widen to the whole group
                near = [x for x in members if x not in taken and x not in pick]
            near.sort(key=lambda x: abs(pos[x] - i))
            pick.add(near[int(rng.integers(min(NEAREST, len(near))))])
        return sorted(pick)

    runs = []
    for item_dir in sorted((ROOT / "results" / "search").glob("*")):
        for path in sorted((item_dir / setting).glob("*_runs.jsonl")):
            if path.name not in CPU_FILES and not path.name.startswith("transfer_from_"):
                runs += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(0)
    result = {"dataset": args.dataset, "horizon": args.horizon, "k": k, "seed_sd": sd, "methods": {}}
    by_method: dict[str, list[dict]] = {}
    for run in runs:
        proposed = sorted({c for c, *_ in run["charges"] if c in predicted})
        if len(proposed) < 5:
            continue
        own = metrics(proposed, predicted, table, sd)
        others = [c for c in pool if c not in set(proposed)]
        matched = [metrics(matched_set(proposed), predicted, table, sd) for _ in range(N_MATCHED)]
        plain = [metrics(sorted(rng.choice(others, len(proposed), replace=False)), predicted, table, sd)
                 for _ in range(N_MATCHED)]
        by_method.setdefault(run["method"], []).append(
            {"seed": run["seed"], "n": len(proposed), "proposals": own,
             "matched": {m: float(np.mean([x[m] for x in matched])) for m in METRICS},
             "plain": {m: float(np.mean([x[m] for x in plain])) for m in METRICS}})
    alpha = 0.05 / max(1, len(by_method))
    result.update({"margin": MARGIN, "interval_level": 1 - alpha})
    for method, entries in sorted(by_method.items()):
        summary = {"runs": len(entries), "mean_proposals": float(np.mean([e["n"] for e in entries]))}
        for m in METRICS:
            own = np.array([e["proposals"][m] for e in entries])
            summary[m] = {"proposals": float(own.mean())}
            for kind in ("matched", "plain"):
                diff = own - np.array([e[kind][m] for e in entries])
                # t interval over runs: a percentile bootstrap over about 10 runs is too narrow.
                half = float(student_t.ppf(1 - alpha / 2, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
                summary[m][kind] = {"mean": float(np.mean([e[kind][m] for e in entries])), "diff": float(diff.mean()),
                                    "low": float(diff.mean() - half), "high": float(diff.mean() + half)}
        summary["non_inferior"] = summary["tau_true"]["matched"]["low"] > -MARGIN
        result["methods"][method] = {"summary": summary, "runs": entries}
    out = ROOT / "results" / "e5" / f"{setting}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for method, block in result["methods"].items():
        s = block["summary"]
        print(f"{setting} {method}: {s['runs']} runs, {s['mean_proposals']:.0f} proposals each")
        for m in METRICS:
            v = s[m]
            print(f"  {m:15s} proposals {v['proposals']:.3f}  " + "  ".join(
                f"{k} {v[k]['mean']:.3f} diff {v[k]['diff']:+.3f} [{v[k]['low']:+.3f}, {v[k]['high']:+.3f}]"
                for k in ("matched", "plain")))
        print(f"  non-inferior at margin {MARGIN} (tau_true vs matched): {s['non_inferior']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
