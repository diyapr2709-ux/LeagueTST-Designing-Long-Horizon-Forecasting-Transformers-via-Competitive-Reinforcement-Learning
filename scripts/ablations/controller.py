"""AB-C: factorized-controller ablations (tested family on ILI, descriptive on ETTh1). CPU.

League rivals, 100 search seeds per arm. Reference: the defaults of
leaguetst.search.replay.factorized_controller (group 8, 4 rivals, learning
rate 0.1, snapshot every 2 updates, entropy 0.01). Arms: 1 rival, 8 rivals,
a snapshot after every update. Endpoint: ILI held-out AUC; permutation tests
against the reference, Holm over 3.

    .venv/bin/python scripts/ablations/controller.py
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, SETTINGS, apply_holm, by_seed, compare, load_table, observation_noise, provenance, replay, \
    setting, summarise, write_json, write_runs  # noqa: E402

from leaguetst.search.replay import factorized_controller  # noqa: E402

SEEDS, FAMILY = 100, 3
ARMS = {"ref": {}, "rivals_1": {"n_rivals": 1}, "rivals_8": {"n_rivals": 8}, "snapshot_1": {"snapshot_every": 1}}


def build(dataset: str) -> dict:
    table, space = load_table(dataset)
    factories = {name: functools.partial(factorized_controller, rivals="league", **kw) for name, kw in ARMS.items()}
    return {"table": table, "space": space, "noise": observation_noise(table), "factories": factories}


def tests(runs: list[dict], table) -> dict:
    ref = list(by_seed(runs, "ref", table).values())
    found = {f"{arm} vs ref": compare(by_seed(runs, arm, table).values(), ref, FAMILY) for arm in ARMS if arm != "ref"}
    return apply_holm(found, FAMILY)


def main(seeds: int = SEEDS) -> int:
    result = {"family": "AB-C", "arms": ARMS, "seeds": seeds, "provenance": provenance(__file__, ["results/ledger.jsonl"]),
              "tables": {}}
    for dataset, _ in SETTINGS:
        runs = replay(functools.partial(build, dataset), list(ARMS), seeds)
        write_runs(OUT / "AB-C" / f"{setting(dataset)}.jsonl", runs)
        table, _space = load_table(dataset)
        entry = {"summary": summarise(runs, table), "registered": dataset == "ILI"}
        if dataset == "ILI":
            entry["tests"] = tests(runs, table)
        result["tables"][setting(dataset)] = entry
    write_json(OUT / "AB-C.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
