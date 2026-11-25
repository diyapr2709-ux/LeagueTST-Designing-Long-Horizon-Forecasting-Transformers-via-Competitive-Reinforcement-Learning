"""Checks of the statistics in scripts/independent/e4_analysis.py on hand-computed cases."""

import importlib.util
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "e4_analysis", Path(__file__).resolve().parents[1] / "scripts/independent/e4_analysis.py")
e4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e4)


def test_exact_permutation_separated_groups():
    # 3 vs 3, fully separated: only the observed split and its mirror reach |diff|, so p = 2 / C(6,3)
    r = e4.permutation_p([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert r["method"] == "exact" and r["splits"] == 20
    assert r["p"] == pytest.approx(2 / 20)
    assert r["diff"] == pytest.approx(-9.0)


def test_exact_permutation_identical_groups():
    assert e4.permutation_p([1.0, 1.0], [1.0, 1.0])["p"] == pytest.approx(1.0)


def test_random_permutation_branch(monkeypatch):
    monkeypatch.setattr(e4, "EXACT_LIMIT", 1)
    monkeypatch.setattr(e4, "N_RANDOM", 2000)
    r = e4.permutation_p([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert r["method"] == "random"
    assert r["p"] == pytest.approx(0.1, abs=0.03)


def test_holm_step_down():
    out = e4.holm({"a": 0.01, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert out["a"]["reject"] and out["b"]["reject"] and out["c"]["reject"]
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["b"]["p_holm"] == pytest.approx(0.04)
    out = e4.holm({"a": 0.01, "b": 0.03, "c": 0.001, "d": 1.0, "e": 1.0, "f": 1.0}, alpha=0.05)
    assert out["c"]["reject"] and out["a"]["reject"] is True and out["b"]["reject"] is False
    assert not out["d"]["reject"]


def test_regret_curve_and_trace_check():
    table = {"A": {2021: 1.0, 2022: 1.2, 2023: 1.4},
             "B": {2021: 0.5, 2022: 0.9, 2023: 0.7}}
    record = {"method": "m", "seed": 0, "spent": 50,
              "charges": [["A", 2021, 20, "full", 20], ["B", 2022, 30, "full", 30]],
              "trace": [[20, "A"], [50, "A"]]}   # wrong: B seen at 2022 (0.9) beats A seen at 2021 (1.0)
    with pytest.raises(ValueError):
        e4.regret_curve(record, table, best_mean=0.7, sd=0.1)
    record["trace"] = [[20, "A"], [50, "B"]]
    curve = e4.regret_curve(record, table, best_mean=0.7, sd=0.1)
    assert len(curve) == 100
    # b = 1 (30 epochs): incumbent A seen at 2021, held out 2022 and 2023 -> 1.3
    assert curve[0] == pytest.approx((1.3 - 0.7) / 0.1)
    # b >= 2: incumbent B seen at 2022, held out 2021 and 2023 -> 0.6
    assert curve[1] == pytest.approx((0.6 - 0.7) / 0.1)


def test_non_full_charge_refused():
    table = {"A": {2021: 1.0, 2022: 1.0}}
    record = {"method": "m", "seed": 0, "spent": 3,
              "charges": [["A", 2021, 3, "k", 3]], "trace": [[3, "A"]]}
    with pytest.raises(ValueError):
        e4.regret_curve(record, table, best_mean=1.0, sd=1.0)


def test_pooled_sd_leave_out():
    table = {"A": {2021: 0.0, 2022: 1.0, 2023: 3.0}, "B": {2021: 5.0}}
    sd, n = e4.pooled_sd(table, leave_out=(2021,))
    assert n == 1 and sd == pytest.approx(math.sqrt(2.0))


def test_worst_case_substitution():
    test_vals = {0: {"auc": 1.0}, 1: {"auc": 2.0}}
    comp_vals = {0: {"auc": 3.0}, 1: {"auc": 4.0}}
    r = e4.worst_case(test_vals, comp_vals, [1], [], "auc")
    # seed 1 of the test arm takes the worst value (4.0): means 2.5 vs 3.5
    assert r["diff"] == pytest.approx(-1.0)
