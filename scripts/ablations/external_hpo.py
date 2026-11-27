"""AB-X: external HPO baselines on the E3 oracle: SMAC (random forest), Hyperband, DEHB. CPU.

Same tables, budgets, seeds, observation noise and charging rule as E3: every
lookup goes through leaguetst.search.replay.Oracle, each distinct config is
charged its epochs once, and a run stops when the next charge would pass 100
full-run equivalents. 30 seeds per method and table. Tested on ILI held-out
AUC against E3's committed random-search runs, Holm over 3; ETTh1 descriptive.

Definitions (AB3):

* Space: the 8 fields as categorical hyperparameters, as TPE sees them.
* smac: SMAC3's HyperparameterOptimizationFacade (random forest, log-scaled
  cost, EI, its default 8% random interleaving), random initial design of 10
  configs (TPE's n_startup), refitted after every evaluation. Infeasible
  architectures are ConfigSpace forbidden clauses and no config is proposed
  twice. EI is maximised exactly, over every feasible config not evaluated
  yet: SMAC's default local search returns only a few local optima, which in a
  space of 2,368 are soon all evaluated, and it then samples at random (seen on
  a synthetic table, before any run on a real one).
* hyperband: SMAC3's HyperbandFacade (random sampling, eta 3, budgets 30/27 to
  30), same forbidden clauses. A budget is rounded to whole epochs (1, 3, 10,
  30) and answered by Oracle.partial, so a promoted config continues its run.
* dehb: the dehb package's ask/tell interface (its defaults; eta 3, fidelities
  30/27 to 30, rounded the same way). An infeasible proposal, or one already
  evaluated at that fidelity or higher, is replaced by the nearest feasible
  config not looked up in the run, as TPE's proposals are.
* dehb_vector (exploratory, AB3 addendum 2): as dehb, but a repaired proposal's
  stored vector is replaced by the evaluated config's, so DEHB's population
  holds what was trained (TPE discards its raw proposal the same way).
* Every run counts its proposals and its repairs.
* Objective: validation MSE for SMAC (its encoder takes logs); log validation
  MSE for DEHB. Screening values are the best value up to the rung, as ASHA's.
* Incumbent: the best fully trained config, as for every E3 method.

    .venv-abx/bin/python scripts/ablations/external_hpo.py run
    .venv/bin/python scripts/ablations/external_hpo.py analyze
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, SETTINGS, apply_holm, by_seed, compare, load_table, observation_noise, provenance, \
    read_runs, replay, setting, summarise, write_json, write_runs  # noqa: E402

from leaguetst.search.replay import MUTABLE_FIELDS, Oracle, Space, nearest_unseen  # noqa: E402

SEEDS, FAMILY = 30, 3
MIN_FIDELITY, MAX_FIDELITY, ETA = 30 / 27, 30.0, 3
ARCH = ("patch_len", "stride", "n_layers", "d_model", "ff_ratio")
NAMES = ("smac", "hyperband", "dehb")          # registered (AB3)
EXPLORATORY = ("dehb_vector",)                  # AB3 addendum 2


def configspace(space: Space, forbid: bool):
    """The 8 fields as categoricals (option indices as strings); infeasible architectures forbidden on request."""
    import itertools

    from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, ForbiddenAndConjunction, ForbiddenEqualsClause

    cs = ConfigurationSpace()
    hps = {f: CategoricalHyperparameter(f, [str(i) for i in range(len(space.domains[f]))]) for f in MUTABLE_FIELDS}
    cs.add(list(hps.values()))
    if forbid:
        feasible = {tuple(space.domains[f].index(space.choices[c][f]) for f in ARCH) for c in space.ids}
        for combo in itertools.product(*(range(len(space.domains[f])) for f in ARCH)):
            if combo not in feasible:
                cs.add(ForbiddenAndConjunction(*[ForbiddenEqualsClause(hps[f], str(i)) for f, i in zip(ARCH, combo)]))
    return cs


def choice_of(space: Space, config) -> dict:
    values = dict(config)
    return {f: space.domains[f][int(values[f])] for f in MUTABLE_FIELDS}


def rung(budget: float) -> int:
    return max(1, int(round(float(budget))))


@contextlib.contextmanager
def scratch():
    """A throwaway working directory: SMAC and DEHB write logs and state files."""
    with tempfile.TemporaryDirectory() as folder:
        old = os.getcwd()
        os.chdir(folder)
        try:
            yield Path(folder)
        finally:
            os.chdir(old)


def enumerating_maximizer(space: Space, cs):
    """SMAC acquisition maximiser that ranks every feasible config not evaluated yet."""
    from ConfigSpace import Configuration
    from smac.acquisition.maximizer.abstract_acquisition_maximizer import AbstractAcquisitionMaximizer

    everything = [Configuration(cs, values={f: str(space.domains[f].index(space.choices[c][f])) for f in MUTABLE_FIELDS})
                  for c in space.ids]

    class Enumerate(AbstractAcquisitionMaximizer):
        def _maximize(self, previous_configs, n_points):
            seen = set(previous_configs)
            left = [c for c in everything if c not in seen]
            for config in left:
                config.origin = "Acquisition Function Maximizer: enumeration"
            return self._sort_by_acquisition_value(left)

    return Enumerate(cs, challengers=len(everything))


def smac_rf():
    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        from smac import HyperparameterOptimizationFacade as Facade
        from smac import Scenario
        from smac.initial_design import RandomInitialDesign
        from smac.runhistory.dataclasses import TrialValue

        with scratch() as folder:
            cs = configspace(space, True)
            scenario = Scenario(cs, deterministic=True, n_trials=100_000, output_directory=folder,
                                seed=int(rng.integers(2**31 - 1)))
            smac = Facade(scenario, lambda config, seed=0: 0.0, initial_design=RandomInitialDesign(scenario, n_configs=10),
                          acquisition_maximizer=enumerating_maximizer(space, cs),
                          config_selector=Facade.get_config_selector(scenario, retrain_after=1, retries=20_000),
                          logging_level=False, overwrite=True)
            while True:
                info = smac.ask()
                cid = space.to_id(choice_of(space, info.config))
                if cid is None:
                    raise RuntimeError("SMAC proposed an infeasible config")
                oracle.proposals = getattr(oracle, "proposals", 0) + 1
                smac.tell(info, TrialValue(cost=float(oracle.full(cid)), time=0.0))

    return run


def hyperband():
    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        from smac import HyperbandFacade as Facade
        from smac import Scenario
        from smac.runhistory.dataclasses import TrialValue

        with scratch() as folder:
            scenario = Scenario(configspace(space, True), deterministic=True, n_trials=100_000, output_directory=folder,
                                min_budget=MIN_FIDELITY, max_budget=MAX_FIDELITY, seed=int(rng.integers(2**31 - 1)))
            smac = Facade(scenario, lambda config, seed=0, budget=0: 0.0, logging_level=False, overwrite=True,
                          intensifier=Facade.get_intensifier(scenario, eta=ETA))
            while True:
                info = smac.ask()
                cid = space.to_id(choice_of(space, info.config))
                if cid is None:
                    raise RuntimeError("SMAC proposed an infeasible config")
                oracle.proposals = getattr(oracle, "proposals", 0) + 1
                smac.tell(info, TrialValue(cost=float(oracle.partial(cid, rung(info.budget))), time=0.0))

    return run


def dehb(store_evaluated: bool = False):
    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        from dehb import DEHB
        from loguru import logger

        logger.remove()
        with scratch() as folder:
            optimizer = DEHB(f=None, cs=configspace(space, False), dimensions=len(MUTABLE_FIELDS), min_fidelity=MIN_FIDELITY,
                             max_fidelity=MAX_FIDELITY, eta=ETA, n_workers=1, output_path=str(folder), save_freq=None,
                             seed=int(rng.integers(2**31 - 1)), log_level="ERROR")
            reached: dict[str, int] = {}
            while True:
                job = optimizer.ask()
                epochs = rung(job["fidelity"])
                params = choice_of(space, job["config"])
                cid = space.to_id(params)
                oracle.proposals = getattr(oracle, "proposals", 0) + 1
                if cid is None or reached.get(cid, 0) >= epochs:
                    oracle.repairs = getattr(oracle, "repairs", 0) + 1
                    cid = nearest_unseen(params, space, set(oracle.charged), rng)
                    if store_evaluated:
                        stored = configspace_config(space, cid)
                        optimizer.config_repository.configs[job["config_id"]].config = optimizer.de[job["fidelity"]].configspace_to_vector(stored)
                value = oracle.partial(cid, epochs)
                reached[cid] = epochs
                optimizer.tell(job, {"fitness": math.log(value), "cost": float(epochs)})

    return run


def configspace_config(space: Space, cid: str):
    """The ConfigSpace configuration of one feasible config."""
    from ConfigSpace import Configuration

    return Configuration(configspace(space, False), values={f: str(space.domains[f].index(space.choices[cid][f])) for f in MUTABLE_FIELDS})


FACTORIES = {"smac": smac_rf, "hyperband": hyperband, "dehb": dehb, "dehb_vector": lambda: dehb(store_evaluated=True)}


def build(dataset: str) -> dict:
    logging.disable(logging.CRITICAL)
    table, space = load_table(dataset)
    return {"table": table, "space": space, "noise": observation_noise(table), "factories": FACTORIES}


def run(seeds: int) -> None:
    for dataset, _ in SETTINGS:
        runs = replay(functools.partial(build, dataset), [*NAMES, *EXPLORATORY], seeds, workers=10)
        write_runs(OUT / "AB-X" / f"{setting(dataset)}.jsonl", runs)
    write_json(OUT / "AB-X" / "run_provenance.json", provenance(__file__, ["results/ledger.jsonl", "requirements-ablations.txt"]))


def repair_rate(runs: list[dict]) -> float | None:
    total = sum(r.get("proposals") or 0 for r in runs)
    return sum(r.get("repairs") or 0 for r in runs) / total if total else None


def analyze() -> None:
    import json

    result = {"family": "AB-X", "run": json.loads((OUT / "AB-X" / "run_provenance.json").read_text()),
              "provenance": provenance(__file__, ["results/ledger.jsonl", "results/search/E3/ILI_T24/cpu_runs.jsonl",
                                                  "results/search/E3/ETTh1_T336/cpu_runs.jsonl"]), "tables": {}}
    for dataset, _ in SETTINGS:
        table, _space = load_table(dataset)
        runs = read_runs(OUT / "AB-X" / f"{setting(dataset)}.jsonl")
        e3 = read_runs(ROOT / "results" / "search" / "E3" / setting(dataset) / "cpu_runs.jsonl")
        base = [r for r in e3 if r["method"] in ("random", "tpe")]
        entry = {"summary": summarise(runs + base, table), "registered": dataset == "ILI",
                 "repair_rate": {m: repair_rate([r for r in runs if r["method"] == m]) for m in (*NAMES, *EXPLORATORY)}}
        if dataset == "ILI":
            rand = list(by_seed(e3, "random", table).values())
            entry["tests"] = apply_holm({f"{m} vs random": compare(by_seed(runs, m, table).values(), rand, FAMILY) for m in NAMES}, FAMILY)
            entry["exploratory"] = {f"{m} vs random": compare(by_seed(runs, m, table).values(), rand, 1) for m in EXPLORATORY}
        result["tables"][setting(dataset)] = entry
    write_json(OUT / "AB-X.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.seeds) if args.command == "run" else analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
