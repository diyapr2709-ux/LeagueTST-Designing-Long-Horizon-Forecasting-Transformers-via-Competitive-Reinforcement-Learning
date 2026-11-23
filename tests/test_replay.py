"""Replay engine: budget accounting, seed draws, traces and determinism."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from leaguetst.search import configs_by_id
from leaguetst.search.replay import METHODS, Oracle, Space, Table, BudgetExhausted, regret_at, run_method


@pytest.fixture(scope="module")
def table() -> Table:
    rng = np.random.default_rng(0)
    rows = []
    for cid, config in configs_by_id().items():
        quality = 0.4 + 0.05 * (config.d_model == 16) + 0.02 * config.n_layers + 0.03 * (not config.instance_norm)
        quality += rng.normal(0, 0.01)
        for seed in (1, 2, 3):
            n = int(rng.integers(8, 31))
            curve = list(quality * (1 + np.exp(-np.arange(1, n + 1) / 3)) + rng.normal(0, 0.005, n))
            rows.append({"config_id": cid, "seed": seed, "val_curve": curve, "train_curve": [v * 0.8 for v in curve],
                         "best_val_mse": min(curve), "epochs_run": n, "model_config": config.to_dict()})
    return Table(rows)


@pytest.fixture(scope="module")
def space(table) -> Space:
    return Space(table.ids)


def test_oracle_charges_once_and_stops_at_the_budget(table):
    oracle = Oracle(table, budget=60, seed=0)
    cid = table.ids[0]
    first = oracle.full(cid)
    spent = oracle.spent
    assert oracle.full(cid) == first and oracle.spent == spent
    with pytest.raises(BudgetExhausted):
        for other in table.ids[1:]:
            oracle.full(other)
    assert oracle.spent <= 60


def test_partial_then_full_charges_the_run_length(table):
    oracle = Oracle(table, budget=10_000, seed=3)
    cid = table.ids[5]
    oracle.partial(cid, 3)
    obs = table.obs[cid][oracle.obs_seed[cid]]
    assert oracle.charged[cid] == min(3, obs.epochs_run)
    oracle.full(cid)
    assert oracle.charged[cid] == obs.epochs_run


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_respects_the_budget_and_recounts(table, space, method):
    budget = 1500
    oracle = run_method(METHODS[method](), table, space, budget, seed=4)
    assert 0 < oracle.spent <= budget
    totals = defaultdict(int)
    for cid, seed, epochs, kind in oracle.charges:
        totals[(cid, seed)] += epochs
        assert oracle.obs_seed[cid] == seed
    assert sum(totals.values()) == oracle.spent
    for (cid, seed), total in totals.items():
        assert total <= table.obs[cid][seed].epochs_run
    spends = [s for s, _ in oracle.trace]
    assert spends == sorted(spends)
    values = [oracle.full_values[c] for _, c in oracle.trace]
    assert values == sorted(values, reverse=True)
    assert regret_at(oracle, budget, 1.0) >= 0


def test_runs_are_deterministic(table, space):
    a = run_method(METHODS["re"](), table, space, 1200, seed=7)
    b = run_method(METHODS["re"](), table, space, 1200, seed=7)
    assert a.trace == b.trace and a.charges == b.charges


def test_regret_is_scored_against_the_multi_seed_mean(table):
    oracle = Oracle(table, budget=10_000, seed=0)
    oracle.full(table.best_id)
    assert regret_at(oracle, 10_000, 1.0) == pytest.approx(0.0)
