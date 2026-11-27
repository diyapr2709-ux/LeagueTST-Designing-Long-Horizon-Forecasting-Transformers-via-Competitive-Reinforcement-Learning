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
import llm_rtx  # noqa: E402
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


def test_revin_parse_matches_every_config():
    for cid, config in configs_by_id().items():
        assert llm_tests.revin_off(cid) == (not config.instance_norm)


def test_policy_level_uses_policies_as_units():
    levels = {0: [1.0, 2.0] * 15}
    for step, shift in ((30, 1.0), (60, 1.1), (120, 1.2)):  # every policy the same shift, rising with steps
        levels[step] = [1.5 + shift + 0.01 * p for p in range(10)]
    out = llm_tests.policy_level(levels)
    assert out["trained_vs_untrained"]["n"] == [10, 30]
    assert out["trained_vs_untrained"]["mean"][0] == pytest.approx(1.5 + 1.1 + 0.045)
    assert out["dose"]["mean_rho"] == pytest.approx(1.0)
    assert out["dose"]["p_sign_flip"] == pytest.approx(2 / 1024)


def test_paired_gap_matches_the_same_seed():
    levels = {0: [1.0] * 30, 30: [2.0] * 10, 60: [2.0] * 10, 120: [2.0] * 10}
    out = llm_tests.paired(levels)
    assert out["policies"] == 10 and out["worse"] == 10
    assert out["gap"]["mean"] == pytest.approx(1.0)


def _levels(n, base, step=0.0):
    return [base + step * i for i in range(n)]


def test_registered_families_full_and_holm_sizes():
    prompted = {"ref": _levels(30, 1.0, 0.01), "anon": _levels(30, 1.2, 0.01), "repair": _levels(30, 0.9, 0.01),
                "0.5B": _levels(30, 1.5, 0.01), "3B": _levels(30, 0.8, 0.01)}
    transfer = {k: _levels(10, 2.0 + j, 0.01) for j, k in enumerate(("steps30", "ref", "steps120", "beta0", "beta01"))}
    out = llm_rtx.registered_families(prompted, transfer, aligned=True)
    assert set(out["AB-P"]["tests"]) == {"anon vs ref_local", "repair vs ref_local", "scale trend"} and not out["AB-P"]["missing"]
    assert set(out["AB-G"]["tests"]) == {"steps trend", "KL trend"} and not out["AB-G"]["missing"]
    scale = out["AB-P"]["tests"]["scale trend"]
    assert list(scale["levels"]) == ["0.5", "1.5", "3.0"] and scale["rho"] < 0  # 0.5B worst, 3B best in this fixture
    assert out["AB-G"]["tests"]["KL trend"]["levels"]["0.04"]["mean"] == pytest.approx(np.mean(transfer["ref"]))
    assert "exploratory_policy_level" in out["AB-G"]


def test_registered_families_missing_arms_keep_family_size():
    prompted = {"ref": _levels(30, 1.0, 0.01), "anon": None, "repair": None, "0.5B": _levels(30, 1.5, 0.01), "3B": _levels(30, 0.8, 0.01)}
    transfer = {"steps30": None, "ref": _levels(10, 2.0, 0.01), "steps120": None, "beta0": _levels(10, 1.0, 0.01), "beta01": _levels(10, 3.0, 0.01)}
    out = llm_rtx.registered_families(prompted, transfer, aligned=False)
    assert out["AB-P"]["missing"] == ["anon vs ref_local", "repair vs ref_local"]
    assert out["AB-G"]["missing"] == ["steps trend"] and "exploratory_policy_level" not in out["AB-G"]
    p = out["AB-P"]["tests"]["scale trend"]["p"]
    assert out["AB-P"]["tests"]["scale trend"]["p_holm"] == pytest.approx(min(1.0, 3 * p))  # alone, it is the smallest of 3
    k = out["AB-G"]["tests"]["KL trend"]
    assert k["p_holm"] == pytest.approx(min(1.0, 2 * k["p"]))


def test_completeness_rules():
    runs = [{"method": "m", "seed": s} for s in range(30)]
    assert llm_rtx.prompted_complete(runs, "m") and not llm_rtx.prompted_complete(runs[:29], "m")
    assert not llm_rtx.prompted_complete(runs, "other")
    from leaguetst.architect.runner import ON_POLICY_STOP
    errors = [{"error": f"x {ON_POLICY_STOP} y", "run_names": ["grpo_league_ETTh1_seed107"]}, {"error": "out of memory", "run_names": ["grpo_league_ETTh1_seed101"]}]
    assert llm_rtx.stopped_seeds(errors) == {7}
    deployed = [{"method": llm_rtx.TRANSFER_METHOD, "seed": s} for s in range(10) if s != 7]
    assert llm_rtx.transfer_complete(deployed, {7}) and not llm_rtx.transfer_complete(deployed, set())
    assert not llm_rtx.transfer_complete(deployed + [{"method": llm_rtx.TRANSFER_METHOD, "seed": 7}], {7})


def test_rtx_source_prefers_the_working_tree(tmp_path, monkeypatch):
    folder = tmp_path / "results/ablations/rtx/ABL_P_ref_local/ILI_T24"
    folder.mkdir(parents=True)
    (folder / "qwen_prompted_1.5b_local_runs.jsonl").write_text('{"method": "m", "seed": 0}\n')
    monkeypatch.setattr(llm_rtx, "ROOT", tmp_path)
    src = llm_rtx.RtxSource(ref="no-such-ref")
    assert src.commit is None
    assert src.runs("ABL_P_ref_local", "ILI_T24") == [{"method": "m", "seed": 0}]
    assert src.files[0]["source"] == "working tree" and len(src.files[0]["sha256"]) == 64
    assert src.runs("ABL_P_3b_local", "ILI_T24") == []


def test_registered_functions_are_unchanged_since_ab4():
    """AB4 registers spearman_trend (with aucs, policy_level, N_PERM) as committed at 5abbd20; the file may grow around them."""
    import subprocess

    try:
        old = subprocess.run(["git", "show", "5abbd20:scripts/ablations/llm_tests.py"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("commit 5abbd20 not available")
    import ast

    old_defs = {n.name: ast.get_source_segment(old, n) for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
    for name in ("spearman_trend", "aucs", "policy_level"):
        assert inspect.getsource(getattr(llm_tests, name)).rstrip() == old_defs[name].rstrip(), name
    assert "N_PERM = 100_000" in old and llm_tests.N_PERM == 100_000


class _FakeSource:
    """Stands in for RtxSource: every run commit descends from the registration and has the registered launcher."""

    def __init__(self, ok=True):
        self.ok, self.files, self.commit, self.ref = ok, [], None, "fake"

    def descends(self, run_hash, registered):
        return self.ok

    def launcher_sha256(self, run_hash):
        return llm_rtx.LAUNCHER_SHA256 if self.ok else "0" * 64

    def frozen_ok(self, run_hash):
        return self.ok

    def run_path_ok(self, run_hash):
        return self.ok

    def _read(self, rel):
        return None


def _env(revision=None, git="a" * 40, backend="local"):
    return {"git_hash": git, "model_revision": revision or llm_rtx.REVISIONS["1.5B"], "backend": backend, "per_gpu": 1, "gpu_name": "RTX"}


def _run(method, seed, revision=None, git="a" * 40, backend="local"):
    return {"method": method, "seed": seed, "env": _env(revision, git, backend)}


def test_check_runs_flags_revision_order_launcher_and_duplicates():
    rev = llm_rtx.REVISIONS["1.5B"]
    check = lambda src, runs, item="ABL_P_ref_local", revision=rev, method="m", frozen=False: llm_rtx.check_runs(src, item, runs, revision, method, frozen)
    runs = [_run("m", s) for s in range(30)]
    assert check(_FakeSource(), runs) == []
    assert any("revisions" in p for p in check(_FakeSource(), runs, "ABL_P_3b_local", llm_rtx.REVISIONS["3B"]))
    assert any("does not descend" in p for p in check(_FakeSource(ok=False), runs))
    assert any("duplicate" in p for p in check(_FakeSource(), runs + runs[:1]))
    assert not llm_rtx.prompted_complete(runs + runs[:1], "m")
    assert any("dirty" in p for p in check(_FakeSource(), [_run("m", s, git="a" * 40 + "-dirty") for s in range(30)]))
    assert any("methods" in p for p in check(_FakeSource(), runs + [_run("wrong", 0)]))
    assert any("backend" in p for p in check(_FakeSource(), [_run("m", s, backend="modal") for s in range(30)]))


def test_check_training_matches_the_item_settings():
    def rec(seed, steps=60, beta=0.04):
        return {"job": {"seed": seed, "steps": steps, "beta": beta, "lr": 1e-4, "rivals": "league", "dataset": "ETTh1",
                        "revision": llm_rtx.REVISIONS["1.5B"], "run_name": f"grpo_league_ETTh1_seed{seed}"}}

    good = [rec(100 + s) for s in range(10)]
    assert llm_rtx.check_training(good, 60, 0.04, set()) == []
    assert llm_rtx.check_training(good, 30, 0.04, set())  # wrong steps for steps30
    assert llm_rtx.check_training(good, 60, 0.1, set())  # wrong beta for beta01
    assert llm_rtx.check_training(good[:9], 60, 0.04, {9}) == []  # a stopped policy has no record
    assert llm_rtx.check_training(good[:9], 60, 0.04, set())


def test_finished_in_time_uses_queue_done_then_commit_time():
    from datetime import datetime

    deadline = datetime.fromisoformat(llm_rtx.DEADLINE)
    assert llm_rtx.finished_in_time("x", [], {"x": "2026-11-28T20:00:00-05:00"}, deadline) is True
    assert llm_rtx.finished_in_time("x", [], {"x": "2026-11-29T00:30:00-05:00"}, deadline) is False
    assert llm_rtx.finished_in_time("x", [{"last_commit_time": "2026-11-27T10:00:00-05:00"}], {}, deadline) is True
    assert llm_rtx.finished_in_time("x", [{"last_commit_time": None}], {}, deadline) is None


def _fake_llm(final: bool) -> dict:
    trend = {"rho": 0.1, "p": 0.4, "n": 90, "p_holm": 1.0, "reject": False,
             "levels": {k: {"mean": 1.0, "low": 0.9, "high": 1.1, "n": 30, "sd": 0.1, "median": 1.0} for k in ("0.5", "1.5", "3.0")}}
    test = {"n": [30, 30], "mean": [1.1, 1.0], "diff": 0.1, "low": -0.1, "high": 0.3, "p": 0.3, "p_holm": 0.9, "reject": False, "detectable_80": 0.5}
    p_tests = {"scale trend": trend, **({"anon vs ref_local": test, "repair vs ref_local": test} if final else {})}
    g_tests = {"KL trend": {**trend, "n": 30}, **({"steps trend": {**trend, "n": 60}} if final else {})}
    status = {"prompted": {"ref": {"item": "ABL_P_ref_local", "searches": 30, "gpu": ["RTX"], "complete": True, "problems": []}},
              "transfer": {"ref": {"item": "ABL_G_ref_local", "deployed": 10, "stopped": [], "gpu": ["RTX"], "complete": True, "problems": [],
                                   "training": {"policies": 10, "onpolicy_mean_max": 0.004, "errors": []}}}}
    reasons = {"anon vs ref_local": "ABL_P_anon_local: pending (AB4 running)", "repair vs ref_local": "ABL_P_repair_local: pending (AB4 running)",
               "steps trend": "ABL_G_steps30_local: pending (AB4 running)"}
    return {"registered": {"source": {"ref": "origin/x", "commit": "c" * 40, "tip": "c" * 40, "tip_time": "t", "fetch_time": "f",
                                      "pinned_is_tip": True, "files": [], "ab2_manifest_matches": {"a": True}},
                           "final": final, "provisional_reasons": [] if final else ["some arms are still running (AB4)"], "status": status, "descriptive": {},
                           "families": {"AB-P": {"tests": p_tests, "missing": [] if final else ["anon vs ref_local", "repair vs ref_local"],
                                                 "missing_reasons": reasons},
                                        "AB-G": {"tests": g_tests, "missing": [] if final else ["steps trend"], "missing_reasons": reasons}}},
            "mps": {"AB-P": {"tests": {}}, "AB-G": {"tests": {}}}}


@pytest.mark.parametrize("final", [False, True])
def test_report_renders_registered_section(monkeypatch, final):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ablations_report", ROOT / "scripts" / "ablations" / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)  # by path: scripts/report.py is also importable as "report"
    monkeypatch.setattr(report, "load", lambda name: _fake_llm(final) if name == "LLM-rtx" else None)
    text = "\n".join(report.registered_llm())
    assert ("Status: final" in text) == final and ("provisional" in text) != final
    assert "scale trend" in text and "KL trend" in text and "positive rho means a higher (worse) AUC" in text
    assert ("pending" in text) != final
    assert "Training (RTX transfer items): ABL_G_ref_local 10 policies" in text
    assert "Not the result of record" in text and "LLM-registered.json" in text


def test_by_level_sorts_numerically():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ablations_report", ROOT / "scripts" / "ablations" / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    test = {"levels": {"0": {"mean": 1.0}, "120": {"mean": 4.0}, "30": {"mean": 2.0}, "60": {"mean": 3.0}}}
    assert [k for k, _ in report.by_level(test)] == ["0", "30", "60", "120"]


def test_step_reading_follows_ab4_rule():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ablations_report", ROOT / "scripts" / "ablations" / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    up, flat = {"rho": 0.3}, {"p": 0.5}
    assert "read as a dose-response" in report.step_reading(up, flat, {"mean_rho": 0.4, "p_sign_flip": 0.01})
    assert "neither" in report.step_reading(up, flat, {"mean_rho": -0.4, "p_sign_flip": 0.01})  # wrong sign
    worse = report.step_reading(up, {"p": 0.01, "diff": 1.5, "low": 1.2, "high": 1.9}, {"mean_rho": 0.4, "p_sign_flip": 0.2})
    assert "differ from the untrained" in worse and "worse by +1.50 SD [+1.20, +1.90]" in worse
    assert "neither" in report.step_reading(None, flat, {"mean_rho": 0.4, "p_sign_flip": 0.01})  # no registered trend



def test_rtx_source_reads_git_and_stops_on_a_differing_copy(tmp_path, monkeypatch):
    import subprocess

    rel = "results/ablations/rtx/ABL_P_ref_local/ILI_T24/qwen_prompted_1.5b_local_runs.jsonl"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text('{"method": "m", "seed": 1}\n')
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "x")
    commit = git("rev-parse", "HEAD")
    (tmp_path / rel).unlink()
    monkeypatch.setattr(llm_rtx, "ROOT", tmp_path)
    src = llm_rtx.RtxSource(ref=commit)
    assert src.commit == commit and src.runs("ABL_P_ref_local", "ILI_T24") == [{"method": "m", "seed": 1}]
    assert src.files[0]["source"] == "git" and src.files[0]["commit"] == commit and src.files[0]["last_commit_time"]
    (tmp_path / rel).write_text('{"method": "m", "seed": 2}\n')
    with pytest.raises(SystemExit):
        llm_rtx.RtxSource(ref=commit).runs("ABL_P_ref_local", "ILI_T24")


class _FakeRtx:
    """A whole RTX branch in memory: AB2 arms complete, AB4 arms absent unless given."""

    def __init__(self, with_ab4=False, manifest_ok=True, bad_method_item=None, tip_ok=True, done_lines=True,
                 commit_time="2026-11-25T03:00:00-05:00", fetch_time="2026-11-29T08:00:00-05:00"):
        import hashlib

        self.files, self.commit, self.ref = [], "c" * 40, "fake"
        self.tip = self.commit if tip_ok else "d" * 40
        self.tip_time, self.fetch_time, self.commit_time = "2026-11-26T01:38:00-05:00", fetch_time, commit_time
        self.arm_runs, self.table_rows, self.data, done = {}, {}, {}, []
        for arm, (item, method) in llm_rtx.PROMPTED.items():
            if llm_rtx.ITEM_AMENDMENT[item] == "AB4" and not with_ab4:
                continue
            m = "wrong" if item == bad_method_item else method
            self.arm_runs[(item, "ILI_T24")] = [{"method": m, "seed": s, "env": _env(llm_rtx.REVISIONS[llm_rtx.PROMPTED_MODEL[arm]])} for s in range(30)]
            done.append(f"2026-11-25T01:00:00-05:00 DONE {item.lower()}_ILI rc=0")
        for arm, item in llm_rtx.TRANSFER.items():
            if llm_rtx.ITEM_AMENDMENT[item] == "AB4" and not with_ab4:
                continue
            steps, beta = llm_rtx.TRANSFER_JOB[arm]
            self.arm_runs[(item, "ILI_T24")] = [{"method": llm_rtx.TRANSFER_METHOD, "seed": s, "env": _env()} for s in range(10)]
            # as on the real branch: AB2 left training logs in results/runs/, AB4's queue copies them to results/ablations/rtx/
            where = llm_rtx.RTX_DIR if llm_rtx.ITEM_AMENDMENT[item] == "AB4" else "results/runs"
            self.table_rows[f"{where}/{item}/ILI_T24/training.jsonl"] = [
                {"job": {"seed": 100 + s, "steps": steps, "beta": beta, "lr": 1e-4, "rivals": "league", "dataset": "ETTh1",
                         "revision": llm_rtx.REVISIONS["1.5B"], "run_name": f"grpo_league_ETTh1_seed{100 + s}"},
                 "env": _env(), "log": [{"onpolicy_mean": 0.001, "unique_rate": 1.0}], "seconds": 60.0} for s in range(10)]
            done.append(f"2026-11-25T02:00:00-05:00 DONE {item.lower()} rc=0")
        self.data["results/logs/l4_queue_status.txt"] = "\n".join(done if done_lines else []).encode()
        paths = [f"results/ablations/rtx/X{i}/f.jsonl" for i in range(9)]
        for path in paths:
            self.data[path] = path.encode()
        lines = [f"{hashlib.sha256(p.encode()).hexdigest() if manifest_ok or i else '0' * 64}  {p}" for i, p in enumerate(paths)]
        self.data[f"{llm_rtx.RTX_DIR}/AB2_move.sha256"] = "\n".join(lines).encode()

    def _read(self, rel):
        return self.data.get(rel)

    def runs(self, item, setting_dir):
        runs = self.arm_runs.get((item, setting_dir), [])
        if runs:
            self.files.append({"path": f"{llm_rtx.RTX_DIR}/{item}/{setting_dir}/x_runs.jsonl", "last_commit_time": self.commit_time})
        return runs

    def rows(self, rel):
        rows = self.table_rows.get(rel, [])
        if rows:
            self.files.append({"path": rel, "last_commit_time": self.commit_time})
        return rows

    def descends(self, run_hash, registered):
        return True

    def launcher_sha256(self, run_hash):
        return llm_rtx.LAUNCHER_SHA256

    def frozen_ok(self, run_hash):
        return True

    def run_path_ok(self, run_hash):
        return True


def test_item_rows_reads_either_layout_and_stops_on_differing_copies():
    item = "ABL_G_steps30_local"
    rtx, runs = f"{llm_rtx.RTX_DIR}/{item}/ILI_T24/training.jsonl", f"results/runs/{item}/ILI_T24/training.jsonl"

    class Src:
        def __init__(self, table):
            self.table = table

        def rows(self, rel):
            return self.table.get(rel, [])

    assert llm_rtx.item_rows(Src({rtx: [{"a": 1}]}), item, "training.jsonl") == [{"a": 1}]
    assert llm_rtx.item_rows(Src({runs: [{"a": 2}]}), item, "training.jsonl") == [{"a": 2}]
    assert llm_rtx.item_rows(Src({rtx: [{"a": 1}], runs: [{"a": 1}]}), item, "training.jsonl") == [{"a": 1}]
    assert llm_rtx.item_rows(Src({}), item, "training.jsonl") == []
    with pytest.raises(SystemExit):
        llm_rtx.item_rows(Src({rtx: [{"a": 1}], runs: [{"a": 2}]}), item, "training.jsonl")


@pytest.fixture
def quiet_registered(monkeypatch):
    monkeypatch.setattr(llm_rtx, "aucs", lambda runs, table, method: [0.01 * int(r["seed"]) + len(method) * 0.1 for r in sorted(runs, key=lambda r: r["seed"])])
    monkeypatch.setattr(llm_rtx, "diagnostics", lambda runs: {})
    monkeypatch.setattr(llm_rtx, "rtx_descriptive", lambda *a: {})
    monkeypatch.setattr(llm_tests, "N_PERM", 200)


def _at(stamp):
    from datetime import datetime

    return datetime.fromisoformat(stamp)


def test_registered_source_before_deadline_is_provisional(quiet_registered):
    out = llm_rtx.registered_source(None, None, now=_at("2026-11-26T12:00:00-05:00"), src=_FakeRtx())
    fam = out["families"]
    assert set(fam["AB-P"]["tests"]) == {"scale trend"} and set(fam["AB-G"]["tests"]) == {"KL trend"}
    assert "pending" in fam["AB-P"]["missing_reasons"]["anon vs ref_local"] and "pending" in fam["AB-G"]["missing_reasons"]["steps trend"]
    assert out["final"] is False and out["queue_done_times"]["abl_g_ref_local"].startswith("2026-11-25")


def test_registered_source_after_deadline_marks_missing_as_not_run(quiet_registered):
    out = llm_rtx.registered_source(None, None, now=_at("2026-11-29T09:00:00-05:00"), src=_FakeRtx())
    assert out["final"] is True
    assert "not run" in out["families"]["AB-P"]["missing_reasons"]["repair vs ref_local"]
    stale = llm_rtx.registered_source(None, None, now=_at("2026-11-29T09:00:00-05:00"), src=_FakeRtx(tip_ok=False))
    assert stale["final"] is False and any("not the local tip" in r for r in stale["provisional_reasons"])
    unfetched = llm_rtx.registered_source(None, None, now=_at("2026-11-29T09:00:00-05:00"), src=_FakeRtx(fetch_time="2026-11-28T10:00:00-05:00"))
    assert unfetched["final"] is False and any("not fetched" in r for r in unfetched["provisional_reasons"])


def test_fetch_time_counts_only_a_fetch_of_the_teammate_branch(tmp_path):
    head = tmp_path / "FETCH_HEAD"
    tip = "a" * 40
    head.write_text(f"{'b' * 40}\tnot-for-merge\tbranch 'main' of https://example.invalid/repo\n")
    assert llm_rtx.fetch_time_of(head, tip) is None  # only main was fetched
    head.write_text(f"{'b' * 40}\tnot-for-merge\tbranch 'main' of https://example.invalid/repo\n"
                    f"{tip}\tnot-for-merge\tbranch 'soumitra-leg2' of https://example.invalid/repo\n")
    assert llm_rtx.fetch_time_of(head, tip) is not None
    assert llm_rtx.fetch_time_of(head, "c" * 40) is None  # listed, but not at the local tip


def test_finished_in_time_falls_back_to_commit_times_by_path(quiet_registered):
    late = _at("2026-11-29T09:00:00-05:00")
    early = llm_rtx.registered_source(None, None, now=late, src=_FakeRtx(done_lines=False))
    assert "scale trend" in early["families"]["AB-P"]["tests"]  # AB2 arms committed on 25 Nov count as finished
    after = llm_rtx.registered_source(None, None, now=late, src=_FakeRtx(done_lines=False, commit_time="2026-11-29T01:00:00-05:00"))
    assert "scale trend" in after["families"]["AB-P"]["missing"]
    assert "not shown to have finished" in after["families"]["AB-P"]["missing_reasons"]["scale trend"]


def test_frozen_check_uses_the_fixed_manifest(tmp_path, monkeypatch):
    import hashlib
    import subprocess

    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=tmp_path,
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    names = [f"f{i}.py" for i in range(10)]
    for n in names:
        (tmp_path / n).write_text(n)
    write_manifest = lambda: (tmp_path / "docs").mkdir(exist_ok=True) or (tmp_path / "docs/e4_rerun.sha256").write_text(
        "".join(f"{hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
    write_manifest()
    git("add", ".")
    git("commit", "-q", "-m", "registered")
    first = git("rev-parse", "HEAD")
    (tmp_path / "f3.py").write_text("changed")
    write_manifest()  # the run commit edits a frozen file and its digest line together
    git("commit", "-q", "-am", "edit")
    second = git("rev-parse", "HEAD")
    monkeypatch.setattr(llm_rtx, "ROOT", tmp_path)
    monkeypatch.setattr(llm_rtx, "E4_MANIFEST_COMMIT", first)
    src = llm_rtx.RtxSource(ref=first)
    assert src.frozen_ok(first) and not src.frozen_ok(second)


def test_registered_source_complete_and_with_a_failing_arm(quiet_registered):
    out = llm_rtx.registered_source(None, None, now=_at("2026-11-27T09:00:00-05:00"), src=_FakeRtx(with_ab4=True))
    assert out["final"] is True and not out["families"]["AB-P"]["missing"] and not out["families"]["AB-G"]["missing"]
    assert "exploratory_policy_level" in out["families"]["AB-G"]
    bad = llm_rtx.registered_source(None, None, now=_at("2026-11-26T12:00:00-05:00"), src=_FakeRtx(bad_method_item="ABL_P_3b_local"))
    assert "scale trend" in bad["families"]["AB-P"]["missing"]
    assert "failed checks" in bad["families"]["AB-P"]["missing_reasons"]["scale trend"]


def test_registered_source_stops_on_a_manifest_mismatch(quiet_registered):
    with pytest.raises(SystemExit):
        llm_rtx.registered_source(None, None, now=_at("2026-11-26T12:00:00-05:00"), src=_FakeRtx(manifest_ok=False))


def test_llm_rtx_is_a_verbatim_move_and_llm_tests_matches_main():
    """26 Nov: the RTX code moved from llm_tests.py (523ae42) to llm_rtx.py unchanged, and llm_tests.py is the
    version Diya's AB4 on main pins for llm_registered.py."""
    import ast
    import hashlib
    import subprocess

    assert hashlib.sha256((ROOT / "scripts/ablations/llm_tests.py").read_bytes()).hexdigest() == \
        "08c72a4c46dded669ebff913a45887ffe215caed24d10005fd27f808b34e9239"
    try:
        old = subprocess.run(["git", "show", "523ae42:scripts/ablations/llm_tests.py"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("523ae42 not in this clone")

    def defs(src):
        out = {}
        for n in ast.parse(src).body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                out[n.name] = ast.dump(n)
            elif isinstance(n, ast.Assign) and all(isinstance(x, ast.Name) for x in n.targets):
                out[",".join(x.id for x in n.targets)] = ast.dump(n)
        return out

    # llm_rtx.py as committed in the move (943d893): later commits add to it (26 Nov, review B1, B6, B19), so the
    # working tree is not the moved file; the move itself is what this test proves verbatim.
    try:
        moved_file = subprocess.run(["git", "show", "943d893:scripts/ablations/llm_rtx.py"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("943d893 not in this clone")
    before, after = defs(old), defs(moved_file)
    moved = [name for name in after if name != "main"]
    assert len(moved) > 30
    assert all(after[name] == before[name] for name in moved), [name for name in moved if after[name] != before.get(name)]


def test_llm_vs_baselines_lists_the_required_baselines_and_orders_seeds():
    import llm_vs_baselines

    assert list(llm_vs_baselines.BASELINES)[:4] == ["random", "regularized evolution", "TPE", "ASHA"]  # RQ1
    assert {"warm-started TPE from ETTh1 (E4)", "portfolio from ETTh1 (E4)"} <= set(llm_vs_baselines.BASELINES)  # E4 controls
    runs = [{"method": "m", "seed": s} for s in (2, 0, 1)]
    original = llm_vs_baselines.by_seed
    try:
        llm_vs_baselines.by_seed = lambda rs, method, table: {int(r["seed"]): float(r["seed"]) for r in rs if r["method"] == method}
        assert llm_vs_baselines.values(runs, "m", None) == [0.0, 1.0, 2.0]
    finally:
        llm_vs_baselines.by_seed = original


# Added 26 Nov, after the AB2 and AB4 results were seen (area-chair review B1, B7, B19): checks and descriptive blocks

def test_run_path_check_compares_with_ab2(tmp_path, monkeypatch):
    import subprocess

    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=tmp_path,
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "leaguetst").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "leaguetst/a.py").write_text("a")
    (tmp_path / "scripts/llm_local.py").write_text("b")
    (tmp_path / "requirements-llm.txt").write_text("c")
    (tmp_path / "notes.md").write_text("x")
    git("add", ".")
    git("commit", "-q", "-m", "ab2")
    first = git("rev-parse", "HEAD")
    (tmp_path / "notes.md").write_text("y")
    git("commit", "-q", "-am", "docs only")
    docs_only = git("rev-parse", "HEAD")
    (tmp_path / "leaguetst/a.py").write_text("changed")
    git("commit", "-q", "-am", "code")
    code = git("rev-parse", "HEAD")
    monkeypatch.setattr(llm_rtx, "ROOT", tmp_path)
    monkeypatch.setitem(llm_rtx.REGISTERED_AT, "AB2", first)
    src = llm_rtx.RtxSource(ref=first)
    assert src.run_path_ok(first) and src.run_path_ok(docs_only) and not src.run_path_ok(code)
    assert not src.run_path_ok("0" * 40)  # unknown commit


def test_check_runs_flags_a_differing_run_path():
    class NoRunPath(_FakeSource):
        def run_path_ok(self, run_hash):
            return False

    runs = [_run("m", s) for s in range(30)]
    problems = llm_rtx.check_runs(NoRunPath(), "ABL_P_ref_local", runs, llm_rtx.REVISIONS["1.5B"], "m", False)
    assert any("run-path" in p and "c2dda8d" in p for p in problems)
    assert llm_rtx.check_runs(_FakeSource(), "ABL_P_ref_local", runs, llm_rtx.REVISIONS["1.5B"], "m", False) == []


def test_queue_events_keep_every_line_but_start():
    lines = ["2026-11-26T01:32:52-05:00 QUEUE_START git abc", "2026-11-26T01:32:52-05:00 START x log=y",
             "2026-11-26T02:09:35-05:00 DONE abl_p_anon_local_ILI rc=0",
             "2026-11-26T05:21:14-05:00 INTERRUPTED abl_p_anon_local_ETTh1: laptop shut down", "2026-11-26T18:25:15-05:00 QUEUE_END"]

    class Src:
        def _read(self, rel):
            return "\n".join(lines).encode() if rel.endswith("ab4_queue_status.txt") else None

    events = llm_rtx.queue_events(Src())
    assert [e["event"] for e in events] == ["QUEUE_START", "DONE", "INTERRUPTED", "QUEUE_END"]
    assert events[0]["detail"] == "git abc" and events[1]["name"] == "abl_p_anon_local_ILI" and events[1]["detail"] == "rc=0"
    assert events[2]["name"] == "abl_p_anon_local_ETTh1" and events[2]["detail"] == "laptop shut down"


def test_disclosure_is_quoted_verbatim():
    text = 'x\n* Report disclosure, fixed now: "The RTX arms were registered\n  on 26 Nov (AB4), after one author\n  had analysed runs."\n* next'
    assert llm_rtx.disclosure(text) == "The RTX arms were registered on 26 Nov (AB4), after one author had analysed runs."
    assert llm_rtx.disclosure("no such sentence") is None and llm_rtx.disclosure(None) is None
    import subprocess

    try:
        real = subprocess.run(["git", "show", f"{llm_rtx.MAIN_AB4}:docs/prereg_ablations.md"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Diya's AB4 commit not in this clone")
    quoted = llm_rtx.disclosure(real)
    assert quoted.startswith("The RTX AB-P and AB-G arms were registered on 26 Nov (AB4), after one author had analysed")
    assert quoted.endswith("The Mac runs are exploratory and never pooled.")


def test_registered_source_records_done_times_and_commits(quiet_registered):
    out = llm_rtx.registered_source(None, None, now=_at("2026-11-27T09:00:00-05:00"), src=_FakeRtx(with_ab4=True))
    ref = out["status"]["prompted"]["ref"]
    assert ref["done_time"] == "2026-11-25T01:00:00-05:00" and ref["amendment"] == "AB2" and ref["commits"] == ["a" * 40]
    st = out["status"]["transfer"]["steps30"]
    assert st["done_time"] == "2026-11-25T02:00:00-05:00" and st["amendment"] == "AB4" and st["training_commits"] == ["a" * 40]


def test_t_based_detectable_is_larger_than_the_registered_normal_one():
    from scipy.stats import t as student_t

    normal = common.detectable(1.0, 10, 1.0, 10, 6)
    t_based = power.detectable_t(1.0, 10, 1.0, 10, 6)
    assert t_based == pytest.approx((student_t.ppf(1 - 0.05 / 12, 18) + student_t.ppf(0.8, 18)) * math.sqrt(0.2), rel=1e-9)
    assert t_based > normal
    r = power.row("x", [0.0, 1.0, 2.0, 3.0, 4.0] * 2, [1.0, 2.0, 3.0] * 10, 6, "u")
    assert r["detectable_80"] == pytest.approx(common.detectable(np.std([0.0, 1.0, 2.0, 3.0, 4.0] * 2, ddof=1), 10, np.std([1.0, 2.0, 3.0] * 10, ddof=1), 30, 6))
    assert r["t_ratio"] == pytest.approx(r["detectable_80_t"] / r["detectable_80"]) and r["t_ratio"] > 1
    assert power.detectable_t(0.0, 10, 0.0, 10, 1) is None


def test_compute_sums_records_and_queue_wall_clock():
    import compute

    runs = [{"method": "m", "spent": 3000, "seconds": 10.0, "env": {"gpu_name": "A"}}, {"method": "m", "spent": 2990, "seconds": 20.0, "gpu": "B"},
            {"method": "cpu", "spent": 3000}]
    rows = compute.search_rows("E3 x", runs, None, "ILI_T24")
    m = next(r for r in rows if r["arm"] == "E3 x: m")
    assert m["machine"] == "Modal" and m["target_epochs"] == 5990 and m["seconds"] == 30.0 and m["cards"] == {"A": 1, "B": 1}
    cpu = next(r for r in rows if r["arm"] == "E3 x: cpu")
    assert cpu["machine"] == "CPU replay" and cpu["seconds"] is None and cpu["cards"] == {compute.NOT_RECORDED: 1}
    training = compute.group([{"training_epochs": 100, "seconds": 5.0, "gpu": "A"}], "training")
    assert training["source_epochs"] == 100 and "target_epochs" not in training
    assert compute.totals(rows + [{"machine": "Modal", "kind": "training", **training}]) == [
        {"machine": "Modal", "card": "A", "kind": "search", "seconds": 10.0, "hours": 10.0 / 3600},
        {"machine": "Modal", "card": "A", "kind": "training", "seconds": 5.0, "hours": 5.0 / 3600},
        {"machine": "Modal", "card": "B", "kind": "search", "seconds": 20.0, "hours": 20.0 / 3600}]
    lines = ["2026-11-26T04:01:31-05:00 START a log=x", "2026-11-26T05:21:14-05:00 INTERRUPTED a: shut down",
             "2026-11-26T05:21:14-05:00 START a log=y", "2026-11-26T05:36:50-05:00 DONE a rc=0"]
    q = compute.queue_wall_clock(lines)
    assert [x["outcome"] for x in q] == ["INTERRUPTED", "DONE rc=0"] and q[0]["seconds"] == 4783.0 and q[1]["seconds"] == 936.0
