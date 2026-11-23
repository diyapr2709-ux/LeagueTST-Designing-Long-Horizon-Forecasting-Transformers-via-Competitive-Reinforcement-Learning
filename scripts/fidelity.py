"""E2: choose the screening epoch k and test the referee, from one table.

On 150 random configs (fixed draw) the best validation MSE up to epoch k is
ranked against the final best of the same run, and against the table's
multi-seed mean. k is the value in (1, 2, 3) with the highest Kendall tau
against the final value, ties to the smaller k. The referee (gradient-
boosted trees on the config fields and the curves up to k, the same model
as leaguetst.search.replay.screened_search) is trained on 300 other random
configs and scored on the 150. It is kept only if its tau beats plain
epoch-k ranking (amendment A9). K1 passes when the referee's tau at the
chosen k is at least 0.5.

Example:
    python scripts/fidelity.py --table E1 --dataset ILI --horizon 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import Space, Table, _features  # noqa: E402

KS = (1, 2, 3)
N_EVAL, N_TRAIN, DRAW_SEED = 150, 300, 150


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="E1")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2021, help="table seed whose curves are screened")
    parser.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    args = parser.parse_args()
    from sklearn.ensemble import GradientBoostingRegressor

    rows = [r for r in read_rows(args.ledger) if r["item"] == args.table and r["dataset"] == args.dataset
            and int(r["horizon"]) == args.horizon]
    table = Table(rows)
    space = Space(table.ids)
    ids = [c for c in table.ids if args.seed in table.obs[c]]
    rng = np.random.default_rng(DRAW_SEED)
    order = rng.permutation(len(ids))
    evaluate = [ids[i] for i in order[:N_EVAL]]
    train = [ids[i] for i in order[N_EVAL:N_EVAL + N_TRAIN]]

    def screened(cid: str, k: int) -> float:
        obs = table.obs[cid][args.seed]
        return min(obs.val_curve[: min(k, obs.epochs_run)])

    final = np.array([table.obs[c][args.seed].best for c in evaluate])
    true = np.array([table.true_value(c) for c in evaluate])
    result = {"table": args.table, "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
              "n_eval": len(evaluate), "n_train": len(train), "k": {}}
    for k in KS:
        epoch_k = np.array([screened(c, k) for c in evaluate])

        def feats(cid: str) -> list[float]:
            obs = table.obs[cid][args.seed]
            return _features(space, cid, obs.val_curve[:k], obs.train_curve[:k], k)

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
        model.fit(np.array([feats(c) for c in train]), np.log([table.obs[c][args.seed].best for c in train]))
        predicted = model.predict(np.array([feats(c) for c in evaluate]))
        result["k"][str(k)] = {
            "tau_epoch_k_vs_final": float(kendalltau(epoch_k, final)[0]),
            "tau_epoch_k_vs_true": float(kendalltau(epoch_k, true)[0]),
            "tau_referee_vs_final": float(kendalltau(predicted, final)[0]),
            "tau_referee_vs_true": float(kendalltau(predicted, true)[0]),
        }
    best_k = max(KS, key=lambda k: (round(result["k"][str(k)]["tau_epoch_k_vs_final"], 6), -k))
    chosen = result["k"][str(best_k)]
    result["chosen_k"] = best_k
    result["referee_beats_epoch_k"] = chosen["tau_referee_vs_final"] > chosen["tau_epoch_k_vs_final"]
    result["k1_referee_tau_at_least_0_5"] = chosen["tau_referee_vs_final"] >= 0.5
    # Seed noise ceiling: how well one seed's final value ranks the multi-seed mean.
    multi = [c for c in evaluate if len(table.obs[c]) >= 3]
    if len(multi) >= 10:
        result["tau_one_seed_final_vs_true"] = float(kendalltau([table.obs[c][args.seed].best for c in multi],
                                                                [table.true_value(c) for c in multi])[0])
    out = ROOT / "results" / "fidelity" / f"{args.dataset}_T{args.horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
