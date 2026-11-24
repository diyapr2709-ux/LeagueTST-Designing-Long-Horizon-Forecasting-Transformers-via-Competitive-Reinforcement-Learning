"""Worst-case sensitivity for failed runs (docs/prereg_E4.md, A26).

Tests use the seeds that completed. Beside each test, every failed seed of
the arm under test takes the worst value observed in either arm, and every
failed seed of the comparator takes the best, and the test is run again. A
conclusion that survives this imputation does not depend on which runs
failed.
"""

from __future__ import annotations

import numpy as np

from leaguetst.eval.stats import permutation_test

Values = dict[int, "float | None"]


def worst_case(test: Values, comparator: Values, lower_is_better: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed values with failures (None) imputed against the arm under test."""
    observed = [v for v in [*test.values(), *comparator.values()] if v is not None]
    if not observed:
        raise ValueError("no completed seed in either arm")
    worst, best = (max(observed), min(observed)) if lower_is_better else (min(observed), max(observed))
    a = np.array([worst if v is None else v for v in test.values()], dtype=float)
    b = np.array([best if v is None else v for v in comparator.values()], dtype=float)
    return a, b


def worst_case_test(test: Values, comparator: Values, lower_is_better: bool = True) -> dict:
    """The test on completed seeds and on the worst-case imputation, with the failure counts."""
    done_a = np.array([v for v in test.values() if v is not None], dtype=float)
    done_b = np.array([v for v in comparator.values() if v is not None], dtype=float)
    a, b = worst_case(test, comparator, lower_is_better)
    return {
        "failed": {"test": sum(v is None for v in test.values()), "comparator": sum(v is None for v in comparator.values())},
        "completed": {"n": [len(done_a), len(done_b)], "diff": float(done_a.mean() - done_b.mean()),
                      "p": permutation_test(done_a, done_b)["p"]},
        "worst_case": {"n": [len(a), len(b)], "diff": float(a.mean() - b.mean()), "p": permutation_test(a, b)["p"]},
    }
