"""E5 (RQ2): does the referee's ranking hold on the architect's own proposals?

The referee is the E2 model: gradient-boosted trees on the config fields and
the seed-2021 curves up to the chosen k, fitted on the same 300 random configs
as scripts/fidelity.py. It then ranks the distinct configs each LLM search run
proposed (training configs left out). Each run's proposals are compared with
quality-matched random sets: for every proposal, a random pick among the 5
unproposed configs whose multi-seed mean is nearest, so a drop in tau that only
reflects a narrower range of quality does not count against the referee.

Per run: Kendall tau of the referee's prediction against the final seed-2021
value and against the multi-seed mean, and the top-2 regret (best multi-seed
mean among the referee's top 2 minus the best in the set, in pooled seed-SD
units). Runs are the unit: the mean over runs of proposals minus matched
random, with a bootstrap interval over runs.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fidelity import DRAW_SEED, N_EVAL, N_TRAIN  # noqa: E402
from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import Space, Table, _features  # noqa: E402

SEED = 2021
N_MATCHED, NEAREST = 200, 5
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
    by_value = sorted(pool, key=table.true_value)
    rank = {c: i for i, c in enumerate(by_value)}

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
        taken = set(proposed)
        matched = []
        for _ in range(N_MATCHED):
            pick = set()
            for c in proposed:
                i = rank[c]
                near = [x for x in by_value[max(0, i - 3 * NEAREST): i + 3 * NEAREST] if x not in taken and x not in pick]
                near.sort(key=lambda x: abs(rank[x] - i))
                pick.add(near[int(rng.integers(min(NEAREST, len(near))))])
            matched.append(metrics(sorted(pick), predicted, table, sd))
        by_method.setdefault(run["method"], []).append(
            {"seed": run["seed"], "n": len(proposed), "proposals": own,
             "matched": {m: float(np.mean([x[m] for x in matched])) for m in own}})
    for method, entries in sorted(by_method.items()):
        summary = {"runs": len(entries), "mean_proposals": float(np.mean([e["n"] for e in entries]))}
        for m in ("tau_final", "tau_true", "top2_regret_sd"):
            own = np.array([e["proposals"][m] for e in entries])
            ref = np.array([e["matched"][m] for e in entries])
            diff = own - ref
            boots = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(axis=1)
            summary[m] = {"proposals": float(own.mean()), "matched_random": float(ref.mean()), "diff": float(diff.mean()),
                          "diff_low": float(np.quantile(boots, 0.025)), "diff_high": float(np.quantile(boots, 0.975))}
        result["methods"][method] = {"summary": summary, "runs": entries}
    out = ROOT / "results" / "e5" / f"{setting}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for method, block in result["methods"].items():
        s = block["summary"]
        print(f"{setting} {method}: {s['runs']} runs, {s['mean_proposals']:.0f} proposals each")
        for m in ("tau_final", "tau_true", "top2_regret_sd"):
            v = s[m]
            print(f"  {m:15s} proposals {v['proposals']:.3f}  matched {v['matched_random']:.3f}  "
                  f"diff {v['diff']:+.3f} [{v['diff_low']:+.3f}, {v['diff_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
