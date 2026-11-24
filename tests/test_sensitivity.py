"""A26 worst-case imputation for failed runs."""

import pytest

from leaguetst.sensitivity import worst_case, worst_case_test


def test_failures_are_imputed_against_the_arm_under_test():
    a, b = worst_case({0: 1.0, 1: None, 2: 0.5}, {0: 2.0, 1: None}, lower_is_better=True)
    assert list(a) == [1.0, 2.0, 0.5] and list(b) == [2.0, 0.5]
    a, b = worst_case({0: 1.0, 1: None}, {0: 2.0, 1: None}, lower_is_better=False)
    assert list(a) == [1.0, 1.0] and list(b) == [2.0, 2.0]


def test_without_failures_both_tests_agree():
    test = {s: 1.0 + 0.1 * s for s in range(5)}
    comparator = {s: 2.0 + 0.1 * s for s in range(5)}
    result = worst_case_test(test, comparator)
    assert result["failed"] == {"test": 0, "comparator": 0}
    assert result["completed"]["p"] == result["worst_case"]["p"]
    assert result["completed"]["diff"] == pytest.approx(-1.0)


def test_a_win_that_depends_on_failures_is_exposed():
    test = {0: 1.0, 1: 1.1, 2: None, 3: None, 4: None}
    comparator = {s: 1.5 + 0.01 * s for s in range(5)}
    result = worst_case_test(test, comparator)
    assert result["failed"]["test"] == 3
    assert result["completed"]["diff"] < 0
    assert result["worst_case"]["diff"] > result["completed"]["diff"]
    assert result["worst_case"]["p"] > result["completed"]["p"]


def test_no_completed_seed_is_refused():
    with pytest.raises(ValueError):
        worst_case({0: None}, {0: None})
