"""The ablation scripts on small synthetic tables: definitions of AB3, statistics, and the Mac runner's sampler."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ablations"))

import common  # noqa: E402
import controller  # noqa: E402
import external_hpo  # noqa: E402
import learning_null  # noqa: E402
import llm_tests  # noqa: E402
import one_seed_bias  # noqa: E402
import plateau  # noqa: E402
import power  # noqa: E402
import referee  # noqa: E402
import robust  # noqa: E402
import rq3  # noqa: E402

from leaguetst.search import configs_by_id  # noqa: E402
from leaguetst.search.replay import METHODS, Space, Table, drawn_seed, factorized_controller, run_method  # noqa: E402


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for cid, config in configs_by_id().items():
        quality = 0.4 + 0.05 * (config.d_model == 16) + 0.02 * config.n_layers + 0.03 * (not config.instance_norm)
        for seed in (2021, 2022, 2023):
            n = int(rng.integers(12, 31))
            curve = list(quality * (1 + np.exp(-np.arange(1, n + 1) / 3)) + rng.normal(0, 0.005, n))
            out.append({"config_id": cid, "seed": seed, "val_curve": curve, "train_curve": [v * 0.8 for v in curve],
                        "best_val_mse": min(curve), "epochs_run": n, "model_config": config.to_dict()})
    return out


@pytest.fixture(scope="module")
def table(rows) -> Table:
    return Table(rows)


@pytest.fixture(scope="module")
def space(table) -> Space:
    return Space(table.ids)


# statistics

def test_holm_keeps_the_registered_family_size():
    tests = {"a": {"p": 0.02}, "b": {"p": 0.9}}
    assert common.apply_holm({k: dict(v) for k, v in tests.items()}, 2)["a"]["reject"]
    assert not common.apply_holm({k: dict(v) for k, v in tests.items()}, 3)["a"]["reject"]
    assert common.apply_holm({"a": {"p": 0.01}}, 3)["a"]["p_holm"] == pytest.approx(0.03)


def test_detectable_difference_matches_the_textbook_value():
    # 30 against 30, SD 1, alpha 0.05, 80% power: (1.95996 + 0.84162) * sqrt(2/30).
    assert common.detectable(1.0, 30, 1.0, 30, 1) == pytest.approx((1.959964 + 0.841621) * math.sqrt(2 / 30), rel=1e-5)
    assert common.detectable(1.0, 30, 1.0, 30, 3) > common.detectable(1.0, 30, 1.0, 30, 1)


def test_compare_reports_the_difference_of_means():
    result = common.compare([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], 3)
    assert result["diff"] == pytest.approx(1.0) and result["low"] < 1.0 < result["high"]


def test_spearman_trend_finds_a_trend_and_no_trend():
    rng = np.random.default_rng(1)
    up = {level: list(rng.normal(i, 0.5, 10)) for i, level in enumerate((0, 30, 60, 120))}
    flat = {level: list(rng.normal(0, 0.5, 10)) for level in (0, 30, 60, 120)}
    assert llm_tests.spearman_trend(up, 2000)["rho"] > 0.7 and llm_tests.spearman_trend(up, 2000)["p"] < 0.01
    assert llm_tests.spearman_trend(flat, 2000)["p"] > 0.01


def test_detectable_rho_shrinks_with_n():
    assert power.detectable_rho(30, 2) == pytest.approx(math.tanh((2.241403 + 0.841621) / math.sqrt(27)), rel=1e-5)
    assert power.detectable_rho(60, 2) < power.detectable_rho(30, 2)


# AB-C

def test_controller_reference_is_the_library_default():
    defaults = {k: p.default for k, p in inspect.signature(factorized_controller).parameters.items()}
    assert controller.ARMS["ref"] == {} and defaults["n_rivals"] == 4 and defaults["snapshot_every"] == 2
    assert {k: v for k, v in controller.ARMS.items() if k != "ref"} == {
        "rivals_1": {"n_rivals": 1}, "rivals_8": {"n_rivals": 8}, "snapshot_1": {"snapshot_every": 1}}
    assert controller.SEEDS == 100 and controller.FAMILY == 3


# AB-Q

def test_noise_free_table_holds_the_seed_means(rows, table):
    free = Table(plateau.noise_free_rows(rows))
    assert free.ids == table.ids and free.best_id == table.best_id
    cid = table.ids[7]
    assert free.true_value(cid) == pytest.approx(table.true_value(cid))
    epochs = [o.epochs_run for o in table.obs[cid].values()]
    assert free.obs[cid][0].epochs_run == math.floor(np.mean(epochs) + 0.5)
    assert free.obs[cid][0].val_curve[0] == pytest.approx(np.mean([o.val_curve[0] for o in table.obs[cid].values()]))
    assert len(free.obs[cid][0].val_curve) == free.obs[cid][0].epochs_run


def test_noise_free_regret_is_zero_for_the_best_config(rows):
    free = Table(plateau.noise_free_rows(rows))
    run = {"trace": [(30, free.best_id)]}
    assert plateau.curve(run, free, 1.0) == pytest.approx(np.zeros(100))


# AB-N

def test_cut_tables(rows, table):
    strict = Table(one_seed_bias.cut_rows(rows, "strict"))
    like = Table(one_seed_bias.cut_rows(rows, "etth1_like"))
    assert all(list(s) == [2021] for s in strict.obs.values())
    multi = [c for c, s in like.obs.items() if len(s) == 3]
    assert len(multi) == 100
    worst_kept = max(like.obs[c][2021].best for c in multi)
    assert sum(o[2021].best <= worst_kept for o in like.obs.values()) == 100
    assert math.isnan(strict.seed_sd())
    assert one_seed_bias.noise_for(strict, table, "strict") == pytest.approx(table.seed_sd())
    assert 0 < one_seed_bias.noise_for(like, table, "etth1_like") < 1


def test_heldout_value_leaves_out_the_seed_the_run_saw(rows, table):
    strict = Table(one_seed_bias.cut_rows(rows, "strict"))
    cid = table.ids[3]
    expected = np.mean([table.obs[cid][s].best for s in (2022, 2023)])
    assert one_seed_bias.heldout_on_full(cid, 5, strict, table) == pytest.approx(expected)
    seen = drawn_seed([2021, 2022, 2023], 5, cid)
    others = np.mean([o.best for s, o in table.obs[cid].items() if s != seen])
    assert one_seed_bias.heldout_on_full(cid, 5, table, table) == pytest.approx(others)


# AB-X

def test_rungs_are_whole_epochs():
    assert [external_hpo.rung(b) for b in (30 / 27, 30 / 9, 10.0, 30.0)] == [1, 3, 10, 30]


def test_forbidden_clauses_leave_exactly_the_feasible_set(space):
    pytest.importorskip("ConfigSpace")
    cs = external_hpo.configspace(space, True)
    cs.seed(0)
    assert all(space.to_id(external_hpo.choice_of(space, c)) is not None for c in cs.sample_configuration(300))
    assert len(list(cs.forbidden_clauses)) == 192 - 74


@pytest.mark.parametrize("name", external_hpo.NAMES)
def test_external_methods_respect_the_budget(name, table, space):
    pytest.importorskip("smac")
    pytest.importorskip("dehb")
    oracle = run_method(external_hpo.FACTORIES[name](), table, space, 450, seed=1)
    again = run_method(external_hpo.FACTORIES[name](), table, space, 450, seed=1)
    assert 0 < oracle.spent <= 450 and oracle.trace == again.trace and oracle.trace
    assert len({c[0] for c in oracle.charges}) == len(oracle.charged)


# AB-R, ST, RQ3

def test_referee_features_partition(table, space):
    cid = table.ids[0]
    both = referee.features(table, space, cid, 2, "both")
    assert referee.features(table, space, cid, 2, "config") + referee.features(table, space, cid, 2, "curves") == both
    assert len(referee.features(table, space, cid, 2, "curves")) == 4


def test_referee_draw_is_e2s(table):
    evaluate, train = referee.draw(table)
    assert len(evaluate) == 150 and len(train) == 1000 and not set(evaluate) & set(train)


def test_mad_sd_recovers_the_seed_sd(rows):
    rng = np.random.default_rng(3)
    noisy = []
    for r in rows:
        noisy.append({**r, "best_val_mse": 0.5 + rng.normal(0, 0.02)})
    assert robust.mad_sd(Table(noisy)) == pytest.approx(0.02, rel=0.08)


def test_probability_of_improvement_counts_ties_as_half():
    x, base = np.zeros((4, 1)), np.zeros((4, 1))
    assert robust.prob_better(x, base)[0] == 0.5
    assert robust.prob_better(x - 1, base)[0] == 1.0


def test_rq3_additive_table_has_no_interactions(table, space):
    class Flat(Table):
        pass

    flat = Flat.__new__(Flat)
    flat.ids = table.ids
    flat.obs = {c: {2021: None} for c in table.ids}
    flat._true = {c: space.choices[c]["n_layers"] + 2.0 * space.choices[c]["instance_norm"] for c in table.ids}
    result = rq3.interactions(flat, space)
    assert result["additive_r2"] == pytest.approx(1.0)
    assert max(result["interaction_increment"].values()) == pytest.approx(0.0, abs=1e-9)
    assert len(result["interaction_increment"]) == 28
    assert result["seed_noise_share_of_config_means"] is None and result["interaction_increment_by_seed"] is None


# AB-L calibration

def test_null_chain_skips_one_step_between_episodes():
    out = learning_null.chain([[1.0, 2.0, 3.0]], np.random.default_rng(0))
    steps = [s for s, _, _ in out]
    assert steps[:6] == [1, 2, 3, 5, 6, 7] and max(steps) <= 60
    assert [p for _, p, _ in out][:6] == [0, 1, 2, 0, 1, 2]


def test_sign_flip_p_is_exact():
    import itertools

    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    assert learning_null.sign_flip_p(np.array([1.0, 1.0, 1.0]), signs) == pytest.approx(2 / 8)


def test_step_rewards_recover_marked_steps(rows, table, space):
    oracle = run_method(METHODS["random"](), table, space, 600, seed=2)
    run = {"seed": 2, "noise_sd": 0.0, "charges": [list(c) for c in oracle.charges], "steps": 2, "step_marks": [0, 4, 9]}
    rewards = learning_null.step_rewards(run, table)
    first = [-math.log(oracle.full_values[c[0]]) for c in oracle.charges[:4]]
    assert len(rewards) == 2 and rewards[0] == pytest.approx(np.mean(first))


# the Mac runner

def test_mac_sampler_draws_from_the_masked_softmax():
    torch = pytest.importorskip("torch")
    import llm_mps

    logits = torch.tensor([0.0, 1.0, 2.0, 50.0])
    generator = torch.Generator().manual_seed(0)
    draws = [llm_mps._draw(logits, [0, 2], generator) for _ in range(4000)]
    assert {t for t, _ in draws} == {0, 2}
    p2 = math.exp(2) / (1 + math.exp(2))
    assert np.mean([t == 2 for t, _ in draws]) == pytest.approx(p2, abs=0.03)
    assert next(lp for t, lp in draws if t == 2) == pytest.approx(math.log(p2), abs=1e-5)


def test_mac_runner_refuses_registered_item_names(monkeypatch):
    import llm_mps

    monkeypatch.setattr(sys, "argv", ["llm_mps.py", "prompted", "--dataset", "ILI", "--item", "ABL_P_ref"])
    with pytest.raises(SystemExit):
        llm_mps.main()


def test_mac_outputs_stay_out_of_the_e4_search_folder():
    import llm_mps

    assert "results/search" not in str(llm_mps.OUT) and llm_mps.CHECKPOINTS == (30, 60, 120)


def test_holdout_build_uses_two_seeds_and_scores_the_third(rows, table):
    search = Table(plateau.noise_free_rows([r for r in rows if int(r["seed"]) != 2021]))
    cid = table.ids[11]
    assert search.true_value(cid) == pytest.approx(np.mean([table.obs[cid][s].best for s in (2022, 2023)]))
    run = {"trace": [(30, search.best_id)], "seed": 0}
    curve = plateau.holdout_curve(run, search, table, 2021)
    assert curve[0] == pytest.approx(table.obs[search.best_id][2021].best - min(o[2021].best for o in table.obs.values()))
    assert len(curve) == 100


def test_heldout_share_is_one_for_three_seed_tables(table):
    assert common.heldout_share([{"trace": [(30, table.best_id)], "seed": 1}], table, 10) == 1.0


def test_incumbent_value_optimism_is_zero_without_bias(rows, table):
    strict = Table(one_seed_bias.cut_rows(rows, "strict"))
    run = {"trace": [(30, strict.best_id)], "seed": 2}
    reported, heldout, gap = one_seed_bias.score(run, strict, table, table.seed_sd())
    assert len(reported) == len(heldout) == len(gap) == 100
    assert heldout[0] == pytest.approx(gap[0] + reported[0] - (table.best_true - strict.best_true) / table.seed_sd())
