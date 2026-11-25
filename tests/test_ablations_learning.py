"""Checks of scripts/ablations/learning.py on hand-computed cases."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "learning", Path(__file__).resolve().parents[1] / "scripts/ablations/learning.py")
learning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(learning)


def test_sign_flip_all_positive():
    # 4 equal positive values: only all-plus and all-minus reach |mean|, p = 2 / 16
    r = learning.sign_flip_test([1.0, 1.0, 1.0, 1.0])
    assert r["patterns"] == 16 and r["p"] == pytest.approx(2 / 16)


def test_sign_flip_symmetric_is_one():
    assert learning.sign_flip_test([1.0, -1.0])["p"] == pytest.approx(1.0)


def test_sign_flip_hand_case():
    # values 3, 1: sign means are 2, 1, -1, -2; |mean| >= 2 in 2 of 4
    assert learning.sign_flip_test([3.0, 1.0])["p"] == pytest.approx(0.5)


def test_spearman_slope_skips_nulls():
    log = [{"step": 1, "reward_mean": 0.1}, {"step": 2, "reward_mean": None},
           {"step": 3, "reward_mean": 0.2}, {"step": 4, "reward_mean": 0.3}]
    slope, n = learning.spearman_slope(log)
    assert n == 3 and slope == pytest.approx(1.0)


def test_spearman_slope_needs_three():
    with pytest.raises(ValueError):
        learning.spearman_slope([{"step": 1, "reward_mean": 0.1}, {"step": 2, "reward_mean": None}])


def test_final_episode_spend_and_check():
    rec = {"job": {"run_name": "x"},
           "log": [{"step": 1, "episode": 1, "training_epochs": 200, "best_in_history": 1.0},
                   {"step": 2, "episode": 2, "training_epochs": 3150, "best_in_history": 0.9},
                   {"step": 3, "episode": 2, "training_epochs": 3400, "best_in_history": 0.8}]}
    fin = learning.final_episode(rec)
    assert fin == {"episode": 2, "spend": 400, "best_in_history": 0.8}
    rec["log"][1]["training_epochs"] = 4000      # episode 2 cannot start 1,000 epochs past its budget
    with pytest.raises(ValueError):
        learning.final_episode(rec)


def test_prompted_best_at_spend():
    table = {"A": {2021: 1.0}, "B": {2022: 0.5}, "C": {2021: 0.1}}
    recs = [{"charges": [["A", 2021, 10, "full", 10], ["B", 2022, 20, "full", 20], ["C", 2021, 30, "full", 30]]}]
    assert learning.prompted_best_at(recs, table, 9) == float("inf")
    assert learning.prompted_best_at(recs, table, 30) == pytest.approx(0.5)
    assert learning.prompted_best_at(recs, table, 60) == pytest.approx(0.1)


def test_holm_two():
    h = learning.holm({"a": 0.02, "b": 0.04})
    assert h["a"]["reject"] and h["b"]["reject"]
    h = learning.holm({"a": 0.03, "b": 0.04})
    assert not h["a"]["reject"] and not h["b"]["reject"]
