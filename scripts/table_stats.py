"""Resolution of the tables and agreement between them.

Per table: the pooled seed SD, how many configs lie within one and two SD
of the best multi-seed mean (a plateau no search can resolve), and, where
configs have several seeds, how one seed's ranking agrees with the others.
Across tables: Spearman between the config rankings and the overlap of
their top-k sets, which says how much a source table's ranking can help on
the target (amendment A13).

Example:
    python scripts/table_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402
from leaguetst.search.replay import Table  # noqa: E402

SETTINGS = (("ETTh1", 336), ("ILI", 24))


def main() -> int:
    rows = list(read_rows(ROOT / "results" / "ledger.jsonl"))
    tables = {}
    result: dict = {"tables": {}, "between": {}}
    for dataset, horizon in SETTINGS:
        selected = [r for r in rows if r["item"] == "E1" and r["dataset"] == dataset and int(r["horizon"]) == horizon]
        if not selected:
            continue
        table = Table(selected)
        tables[dataset] = table
        sd = table.seed_sd()
        multi = [c for c in table.ids if len(table.obs[c]) >= 3]
        entry = {"configs": len(table.ids), "multi_seed_configs": len(multi), "pooled_sd": sd,
                 "best": table.best_true, "best_id": table.best_id,
                 "within_1sd": int(sum(table.true_value(c) <= table.best_true + sd for c in table.ids)),
                 "within_2sd": int(sum(table.true_value(c) <= table.best_true + 2 * sd for c in table.ids))}
        if len(multi) >= 20:
            seed_list = sorted(table.obs[multi[0]])
            agreement = []
            for held in seed_list:
                one = [table.obs[c][held].best for c in multi]
                rest = [np.mean([o.best for s, o in table.obs[c].items() if s != held]) for c in multi]
                agreement.append(float(spearmanr(one, rest)[0]))
                ranked = sorted(multi, key=lambda c: table.obs[c][held].best)
                winner = ranked[0]
                order = sorted(multi, key=lambda c: np.mean([o.best for s, o in table.obs[c].items() if s != held]))
                entry.setdefault("one_seed_winner_rank_on_others", []).append(order.index(winner) + 1)
            entry["spearman_one_seed_vs_rest"] = agreement
        result["tables"][dataset] = entry
    if len(tables) == 2:
        a, b = tables["ETTh1"], tables["ILI"]
        common = sorted(set(a.ids) & set(b.ids))
        va = [a.true_value(c) for c in common]
        vb = [b.true_value(c) for c in common]
        overlap = {}
        for k in (10, 50, 100, 237):
            top_a = set(sorted(common, key=a.true_value)[:k])
            top_b = set(sorted(common, key=b.true_value)[:k])
            overlap[str(k)] = len(top_a & top_b)
        result["between"] = {"configs": len(common), "spearman": float(spearmanr(va, vb)[0]), "top_k_overlap": overlap,
                             "ili_rank_of_etth1_best": sorted(common, key=b.true_value).index(a.best_id) + 1,
                             "etth1_rank_of_ili_best": sorted(common, key=a.true_value).index(b.best_id) + 1}
    out = ROOT / "results" / "table_stats.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
