"""Shared pieces of the ablation scripts (docs/prereg_ablations.md): tables, runs, tests, records.

Nothing here decides a search or a score: lookups, charges, incumbents and
held-out regret come from leaguetst.search.replay, tests and Holm from
leaguetst.eval.stats.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leaguetst.architect import runner  # noqa: E402
from leaguetst.eval.stats import holm, permutation_test  # noqa: E402
from leaguetst.search.replay import (  # noqa: E402
    FULL_RUN_EPOCHS, Oracle, Space, Table, no_incumbent_value, regret_from_trace, run_method, run_metric,
)

OUT = ROOT / "results" / "ablations"
SETTINGS = (("ILI", 24), ("ETTh1", 336))
BUDGETS = (10, 25, 50, 100)
MAX_EPOCHS = 100 * FULL_RUN_EPOCHS
ALPHA, POWER = 0.05, 0.80


def setting(dataset: str) -> str:
    return f"{dataset}_T{runner.HORIZON[dataset]}"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(script: str | Path, inputs: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Git hash, script digest, package versions and input digests behind a result file."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "scikit-learn", "optuna", "smac", "dehb", "ConfigSpace", "torch", "transformers"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    script = Path(script)
    return {"git_hash": runner.git_hash(), "script": str(script.resolve().relative_to(ROOT)), "script_sha256": sha256(script),
            "versions": versions, "inputs_sha256": {str(Path(p)): sha256(ROOT / p) for p in inputs}}


def load_table(dataset: str) -> tuple[Table, Space]:
    rows = runner.table_rows(dataset)
    table = Table(rows)
    return table, Space(table.ids)


def observation_noise(table: Table) -> float:
    from replay import observation_noise as noise

    return noise(table)


@functools.lru_cache(maxsize=None)
def code_hash() -> str:
    return runner.git_hash()


def record(method: str, seed: int, oracle: Oracle, noise: float) -> dict[str, Any]:
    """The run dict that ``run_metric`` and the summaries read (as scripts/replay.py writes it)."""
    return {"method": method, "seed": seed, "git_hash": code_hash(), "noise_sd": noise, "spent": oracle.spent, "lookups": len(oracle.charged),
            "trace": [list(t) for t in oracle.trace], "charges": [list(c) for c in oracle.charges]}


_STATE: dict[str, Any] = {}


def _init(build: Callable[[], dict[str, Any]]) -> None:
    _STATE.update(build())


def _work(task: tuple[str, int]) -> dict[str, Any]:
    name, seed = task
    oracle = run_method(_STATE["factories"][name](), _STATE["table"], _STATE["space"], MAX_EPOCHS, seed, _STATE["noise"])
    return record(name, seed, oracle, _STATE["noise"])


def replay(build: Callable[[], dict[str, Any]], names: list[str], seeds: int, workers: int = 8) -> list[dict[str, Any]]:
    """Run every (method, seed) of ``build()``'s factories on its table; order is fixed by (method, seed)."""
    tasks = [(name, seed) for name in names for seed in range(seeds)]
    if workers <= 1:
        _init(build)
        return [_work(t) for t in tasks]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(build,)) as pool:
        return pool.map(_work, tasks, chunksize=4)


def write_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in runs))


def read_runs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True, default=_plain) + "\n")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


# ---------------------------------------------------------------- statistics

def auc(run: dict[str, Any], table: Table) -> float:
    """Held-out AUC regret over budgets 1 to 100 in pooled seed-SD units (E3's endpoint)."""
    return run_metric(run, table, "auc", no_incumbent_value(table), table.seed_sd())


def by_seed(runs: list[dict[str, Any]], method: str, table: Table, metric: str | int = "auc") -> dict[int, float]:
    fallback, sigma = no_incumbent_value(table), table.seed_sd()
    return {int(r["seed"]): run_metric(r, table, metric, fallback, sigma) for r in runs if r["method"] == method}


def mean_ci(values: Iterable[float]) -> dict[str, float]:
    """Mean with a t interval."""
    from scipy.stats import t as student_t

    x = np.asarray(list(values), dtype=float)
    half = float(student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
    return {"n": len(x), "mean": float(x.mean()), "low": float(x.mean() - half), "high": float(x.mean() + half),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"), "median": float(np.median(x))}


def detectable(sd_a: float, n_a: int, sd_b: float, n_b: int, family: int) -> float:
    """Smallest mean difference a two-sided test at 0.05/family finds with 80% power (normal approximation)."""
    from scipy.stats import norm

    return float((norm.ppf(1 - ALPHA / family / 2) + norm.ppf(POWER)) * math.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))


def compare(a: Iterable[float], b: Iterable[float], family: int) -> dict[str, Any]:
    """Two-sample seed-level comparison: permutation p, Welch t interval, smallest detectable difference."""
    from scipy.stats import t as student_t

    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
    diff = float(a.mean() - b.mean())
    half = float(student_t.ppf(0.975, df) * se)
    return {"n": [len(a), len(b)], "mean": [float(a.mean()), float(b.mean())], "diff": diff, "low": diff - half,
            "high": diff + half, "p": permutation_test(a, b)["p"],
            "detectable_80": detectable(a.std(ddof=1), len(a), b.std(ddof=1), len(b), family)}


def apply_holm(tests: dict[str, dict[str, Any]], family: int) -> dict[str, dict[str, Any]]:
    """Holm over a registered family; a test that was not run counts as p = 1 and keeps the family size."""
    p = {name: test["p"] for name, test in tests.items()}
    for i in range(family - len(p)):
        p[f"not run {i + 1}"] = 1.0
    if len(p) != family:
        raise ValueError(f"{len(p)} tests in a family of {family}")
    rejected = holm(p)
    running = 0.0
    for rank, name in enumerate(sorted(p, key=p.get)):
        running = max(running, min(1.0, (family - rank) * p[name]))
        if name in tests:
            tests[name].update({"p_holm": running, "reject": bool(rejected[name])})
    return tests


def heldout_share(runs: list[dict[str, Any]], table: Table, budget: int) -> float:
    """Share of incumbents at ``budget`` scored on seeds the search did not see (1.0 on ILI; small on ETTh1)."""
    fallback = no_incumbent_value(table)
    flags = []
    for run in runs:
        _, cid = regret_from_trace(run["trace"], table, budget * FULL_RUN_EPOCHS, fallback)
        flags.append(bool(cid) and table.heldout_value(cid, int(run["seed"]))[1])
    return float(np.mean(flags))


def summarise(runs: list[dict[str, Any]], table: Table) -> dict[str, Any]:
    """Per method: AUC and regret at each budget (pooled seed-SD units), and the same regret in raw MSE."""
    sigma = table.seed_sd()
    out: dict[str, Any] = {}
    for method in sorted({r["method"] for r in runs}):
        aucs = list(by_seed(runs, method, table).values())
        entry: dict[str, Any] = {"auc": mean_ci(aucs), "auc_raw_mse": mean_ci([v * sigma for v in aucs])}
        chosen = [r for r in runs if r["method"] == method]
        entry["heldout_share"] = {str(b): heldout_share(chosen, table, b) for b in BUDGETS}
        for b in BUDGETS:
            values = list(by_seed(runs, method, table, b).values())
            entry[str(b)] = mean_ci(values)
            entry[f"{b}_raw_mse"] = mean_ci([v * sigma for v in values])
        out[method] = entry
    return out
