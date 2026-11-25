"""AB-L calibration (exploratory, post hoc): what the AB-L slopes look like for a model that never learns.

Not registered: written after AB-L's results were pushed (3cb92a1). A training
run resets its search history every episode, and even the untrained model finds
better configs as its in-prompt history grows, so the Spearman slope of reward
over 60 steps need not be centred on zero without learning. The null here is
built from untrained prompted-Qwen searches, whose budget (3,000 epochs) equals
one training episode's: random episodes are chained until 60 steps, and the raw
slope and the position-adjusted slope of scripts/ablations/learning.py are
computed on the chain. Two sources: E3's ten A10 searches per table (no step
boundaries recorded, so charges are split evenly over the recorded steps) and
the Mac reference ABL_P_ref_mps (exact step boundaries), when it exists.

    .venv/bin/python scripts/ablations/learning_null.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, SETTINGS, Oracle, load_table, provenance, read_runs, setting, write_json  # noqa: E402

N_STEPS, N_CHAINS, N_TESTS, POLICIES = 60, 4000, 2000, 10


def step_rewards(run: dict, table) -> list[float]:
    """Mean reward (-log validation MSE, as seen by the search) of the new configs of each completed step."""
    oracle = Oracle(table, 10**9, int(run["seed"]), float(run["noise_sd"]))
    rewards = np.array([-math.log(oracle._observation(c[0]).best) for c in run["charges"]])
    if "step_marks" in run:
        marks = run["step_marks"][: run["steps"]] + [len(rewards) if len(run["step_marks"]) == run["steps"] else run["step_marks"][run["steps"]]]
        groups = [rewards[a:b] for a, b in zip(marks, marks[1:])]
    else:
        groups = np.array_split(rewards, run["steps"])
    return [float(g.mean()) for g in groups if len(g)]


def chain(episodes: list[list[float]], rng: np.random.Generator, gap: bool = True) -> list[tuple[int, int, float]]:
    """(step, position in episode, reward) of a 60-step run made of untrained episodes; one unlogged step between episodes."""
    out, step = [], 1
    while step <= N_STEPS:
        for pos, reward in enumerate(episodes[int(rng.integers(len(episodes)))]):
            if step > N_STEPS:
                break
            out.append((step, pos, reward))
            step += 1
        step += int(gap)
    return out


def slopes(chains: list[list[tuple[int, int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    """Raw and position-adjusted Spearman slopes (position means pooled over the arm, as learning.py does)."""
    from scipy.stats import spearmanr

    sums: dict[int, list[float]] = {}
    for c in chains:
        for _, pos, r in c:
            sums.setdefault(pos, []).append(r)
    mean = {p: float(np.mean(v)) for p, v in sums.items()}
    raw = [spearmanr([s for s, _, _ in c], [r for _, _, r in c])[0] for c in chains]
    adjusted = [spearmanr([s for s, _, _ in c], [r - mean[p] for _, p, r in c])[0] for c in chains]
    return np.array(raw), np.array(adjusted)


def sign_flip_p(x: np.ndarray, signs: np.ndarray) -> float:
    return float(np.mean(np.abs(signs @ x / len(x)) >= abs(x.mean()) - 1e-12))


def null(episodes: list[list[float]], observed: dict) -> dict:
    rng = np.random.default_rng(0)
    chains = [chain(episodes, rng) for _ in range(N_CHAINS)]
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=POLICIES)))
    out = {"episodes": len(episodes), "steps_per_episode": float(np.mean([len(e) for e in episodes]))}
    raw, _ = slopes(chains)
    means = np.array([raw[rng.choice(len(raw), POLICIES, replace=False)].mean() for _ in range(N_TESTS)])
    out["raw"] = {"mean": float(raw.mean()), "sd": float(raw.std(ddof=1)),
                  "false_positive_rate": {str(a): float(np.mean([sign_flip_p(rng.choice(raw, POLICIES, replace=False), signs) <= a
                                                                  for _ in range(N_TESTS)])) for a in (0.05, 0.025)},
                  "observed_league_mean": observed["raw"], "p_one_sided_against_null": float((1 + (means >= observed["raw"]).sum()) / (N_TESTS + 1))}
    # The adjusted slope pools position means over the arm, so each null arm is adjusted on its own ten chains.
    adjusted = []
    for _ in range(N_TESTS):
        picked = [chains[i] for i in rng.choice(len(chains), POLICIES, replace=False)]
        adjusted.append(slopes(picked)[1].mean())
    adjusted = np.array(adjusted)
    out["position_adjusted"] = {"mean_of_arm_means": float(adjusted.mean()), "sd_of_arm_means": float(adjusted.std(ddof=1)),
                                "observed_league_mean": observed["adjusted"],
                                "p_one_sided_against_null": float((1 + (adjusted >= observed["adjusted"]).sum()) / (N_TESTS + 1))}
    return out


def observed_slopes(dataset: str) -> dict:
    """The league arm's mean slopes in results/ablations/AB-L.json (Soumitra)."""
    arm = json.loads((OUT / "AB-L.json").read_text())["by_source"][dataset]["arms"]["league"]
    return {"raw": float(arm["mean_slope"]), "adjusted": float(arm["mean_position_adjusted_slope"])}


def sign_flip_observed(dataset: str) -> dict:
    """Exact two-sided sign flip of the ten observed position-adjusted slopes against zero (no null model)."""
    import itertools

    data = json.loads((OUT / "AB-L.json").read_text())
    slopes = np.array([v["position_adjusted_slope"] for v in data["by_source"][dataset]["per_policy"].values() if v["arm"] == "league"])
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(slopes))))
    return {"n": len(slopes), "mean": float(slopes.mean()), "sd": float(slopes.std(ddof=1)), "p_sign_flip": sign_flip_p(slopes, signs)}


def main() -> int:
    result = {"analysis": "AB-L calibration", "exploratory": "post hoc: written after AB-L's results were seen",
              "n_chains": N_CHAINS, "n_tests": N_TESTS, "tables": {},
              "provenance": provenance(__file__, ["results/ledger.jsonl", "results/ablations/AB-L.json", "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl", "results/search/E3/ETTh1_T336/qwen_prompted_runs.jsonl"])}
    for dataset, _ in SETTINGS:
        table, _space = load_table(dataset)
        observed = observed_slopes(dataset)
        sources = {"E3_A10": read_runs(ROOT / "results/search/E3" / setting(dataset) / "qwen_prompted_runs.jsonl"),
                   "ABL_P_ref_mps": read_runs(OUT / "mps/ABL_P_ref_mps" / setting(dataset) / "qwen_prompted_mps_runs.jsonl")}
        result.setdefault("observed_adjusted_slopes", {})[dataset] = sign_flip_observed(dataset)
        result["tables"][dataset] = {name: null([step_rewards(r, table) for r in runs], observed) for name, runs in sources.items() if runs}
    write_json(OUT / "AB-L-null.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
