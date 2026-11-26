"""AB-P and AB-G on the local RTX arms (amendments AB2 and AB4): the registered tests, computed by the items' owner.

scripts/ablations/llm_tests.py (Anirudh, 5abbd20) leaves registered items to their owner ("their families are
computed by the owner of the items"). This file computes them for the RTX arms under results/ablations/rtx/ with
the same functions, imported unchanged from byte-identical copies of llm_tests.py and common.py at 5abbd20 (sha256
checked at import): ILI held-out AUC regret over budgets 1 to 100 in pooled seed-SD units (common.by_seed),
compare (seed-level permutation p, Welch interval, detectable difference), spearman_trend (N_PERM relabellings,
seed 0), apply_holm (a missing test counts as p = 1 and keeps the family size), policy_level (AB3 addendum 3).

Registered (AB2, AB4), ILI:
  AB-P, Holm over 3: anon vs ABL_P_ref_local, repair vs ABL_P_ref_local, scale trend (0.5B, 1.5B = ABL_P_ref_local,
      3B). AB-P names a trend without its own statistic; it is computed with spearman_trend, AB-G's trend statistic.
  AB-G, Holm over 2: steps trend (0 = ABL_P_ref_local, 30, 60 = ABL_G_ref_local, 120) and KL trend (beta 0,
      0.04 = ABL_G_ref_local, 0.1). Step levels share training and deployment seeds, so they are dependent (AB4).
Exploratory or descriptive, labelled so: AB3 addendum 3's training-at-all and dose analyses on the RTX levels (AB4),
the AB1 hardware check (ABL_G_ref_local vs the J2 league arm, same seeds), ETTh1 summaries (in-sample, A18), GPU
names per arm (AB1). Nothing else (AB4: no analysis added after results are seen).

    .venv/bin/python scripts/ablations/rtx_tests.py      # writes results/ablations/rtx/RTX_tests.json
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED = {"common.py": "46aef3c196864a1f835505e58e2b1bef22940ca03219bb60887514039efa310b",
          "llm_tests.py": "01b70835dd128001c78eceb040590b7e79ace2a17f260a3b98022cdbcd0ee0a6"}
for _name, _digest in PINNED.items():
    if hashlib.sha256((HERE / _name).read_bytes()).hexdigest() != _digest:
        raise SystemExit(f"{_name} differs from 5abbd20 (AB4 pins it); refusing to run")
sys.path.insert(0, str(HERE))

from common import OUT, ROOT, apply_holm, by_seed, compare, load_table, mean_ci, provenance, read_runs, write_json  # noqa: E402
from llm_tests import STEP_LEVELS, policy_level, spearman_trend  # noqa: E402

RTX = OUT / "rtx"
J2 = ROOT / "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl"
G = "grpo_league_from_ETTh1"
P_ARMS = {"ref": ("ABL_P_ref_local", "qwen_prompted_1.5b_local"), "anon": ("ABL_P_anon_local", "qwen_prompted_anon_1.5b_local"),
          "repair": ("ABL_P_repair_local", "qwen_prompted_repair_1.5b_local"),
          "0.5b": ("ABL_P_05b_local", "qwen_prompted_0.5b_local"), "3b": ("ABL_P_3b_local", "qwen_prompted_3b_local")}
G_STEPS = {30: "ABL_G_steps30_local", 60: "ABL_G_ref_local", 120: "ABL_G_steps120_local"}
G_BETA = {0.0: "ABL_G_beta0_local", 0.04: "ABL_G_ref_local", 0.1: "ABL_G_beta01_local"}
N_SEARCH, N_POLICY = 30, 10  # every GRPO arm here deployed all 10 policies (no on-policy stop occurred, AB2/AB4 commits);
# an arm with an on-policy stop would have 9 deployments, be reported as not run by complete(), and need its own handling


def runs(item: str, setting: str) -> list[dict]:
    folder = RTX / item / setting
    return [r for path in sorted(folder.glob("*_runs.jsonl")) for r in read_runs(path)]


def values(item: str, method: str, table, setting: str = "ILI_T24") -> list[float]:
    """AUC per seed in seed order; refuses duplicates (a seed run twice) and other methods' rows."""
    rows = runs(item, setting)
    seeds = [int(r["seed"]) for r in rows]
    if len(seeds) != len(set(seeds)) or any(r["method"] != method for r in rows):
        raise SystemExit(f"{item}/{setting}: duplicate seeds or foreign methods")
    if sorted(seeds) != list(range(len(seeds))):
        raise SystemExit(f"{item}/{setting}: seeds are not 0..{len(seeds) - 1}")
    return [v for _, v in sorted(by_seed(rows, method, table).items())]


def complete(vals: list[float], n: int) -> bool:
    """values() has already refused any seed set other than 0..len-1, so a length of n means seeds 0..n-1."""
    return len(vals) == n


def detectable_rho(n: int, family: int) -> float:
    """Approximate smallest |rho| a two-sided test at 0.05/family finds with 80% power (Fisher z; ignores dependence)."""
    from scipy.stats import norm

    return float(math.tanh((norm.ppf(1 - 0.05 / family / 2) + norm.ppf(0.8)) / math.sqrt(n - 3)))


def trend(levels: dict, family: int, caveat: str) -> dict:
    out = spearman_trend(levels)
    out["caveat"] = caveat
    out["detectable_rho_80_approx"] = detectable_rho(sum(len(v) for v in levels.values()), family)
    return out


SCALE_CAVEAT = "the three model sizes share search seeds 0-29, so levels are dependent; the permutation p treats them as exchangeable"
STEPS_CAVEAT = ("levels 30, 60, 120 share training seeds 100-109 and deployment seeds 0-9, and level 0 (prompted, 30 seeds) "
                "mixes 'trained at all' with 'trained longer'; levels are dependent and the permutation p treats them as exchangeable")
KL_CAVEAT = "the three KL levels share training seeds 100-109 and deployment seeds 0-9, so levels are dependent"


def ab_p(table) -> dict:
    v = {arm: values(item, method, table) for arm, (item, method) in P_ARMS.items()}
    tests, not_run = {}, []
    for arm in ("anon", "repair"):
        if complete(v[arm], N_SEARCH) and complete(v["ref"], N_SEARCH):
            tests[f"{arm} vs ref"] = compare(v[arm], v["ref"], 3)
        else:
            not_run.append(f"{arm} vs ref")
    levels = {0.5: v["0.5b"], 1.5: v["ref"], 3.0: v["3b"]}
    if all(complete(x, N_SEARCH) for x in levels.values()):
        tests["scale trend"] = trend(levels, 3, SCALE_CAVEAT)
    else:
        not_run.append("scale trend")
    return {"complete": {arm: len(x) for arm, x in v.items()}, "tests": apply_holm(tests, 3), "not_run": not_run,
            "statistic_note": "scale trend: spearman_trend on level ranks 0.5B < 1.5B < 3B (AB4 addendum 2)"}


def ab_g(table) -> dict:
    steps = {0: values("ABL_P_ref_local", "qwen_prompted_1.5b_local", table)}
    steps.update({s: values(item, G, table) for s, item in G_STEPS.items()})
    beta = {b: values(item, G, table) for b, item in G_BETA.items()}
    tests, not_run, out = {}, [], {}
    steps_ok = complete(steps[0], N_SEARCH) and all(complete(steps[s], N_POLICY) for s in STEP_LEVELS[1:])
    if steps_ok:
        tests["steps trend"] = trend(steps, 2, STEPS_CAVEAT)
        out["exploratory_addendum3"] = {"label": "exploratory (AB4: AB3 addendum 3's analyses on the RTX levels)",
                                        **policy_level(steps)}
    else:
        not_run.append("steps trend")
    if all(complete(x, N_POLICY) for x in beta.values()):
        tests["KL trend"] = trend(beta, 2, KL_CAVEAT)
    else:
        not_run.append("KL trend")
    out.update({"complete": {"steps": {str(k): len(x) for k, x in steps.items()}, "beta": {str(k): len(x) for k, x in beta.items()}},
                "tests": apply_holm(tests, 2), "not_run": not_run})
    return out


def hardware_check(table) -> dict:
    """AB1, descriptive: ABL_G_ref_local against the J2 league arm on the same deployment seeds. ABL_P_ref is not run."""
    local = by_seed(runs("ABL_G_ref_local", "ILI_T24"), G, table)
    j2 = by_seed(read_runs(J2), G, table)
    both = sorted(set(local) & set(j2))
    return {"label": "descriptive, no test (AB1)", "seeds": both, "rtx_local": mean_ci([local[s] for s in both]),
            "j2_modal": mean_ci([j2[s] for s in both]), "diff_rtx_minus_j2": mean_ci([local[s] - j2[s] for s in both]),
            "ABL_P_ref_check": "not available: ABL_P_ref was not run (AB4)"}


def etth1_descriptive(etth1) -> dict:
    out = {"label": "descriptive, in-sample (ETTh1 has one seed per config, A18)"}
    for arm, (item, method) in P_ARMS.items():
        out[arm] = mean_ci(values(item, method, etth1, "ETTh1_T336"))
    return out


def gpus() -> dict:
    names = {}
    for folder in sorted(RTX.glob("ABL_*_local/*")):
        rows = [r for path in sorted(folder.glob("*.jsonl")) for r in read_runs(path)]
        names[f"{folder.parent.name}/{folder.name}"] = sorted({r.get("gpu") or r.get("env", {}).get("gpu_name", "unknown") for r in rows})
    return names


def main() -> int:
    if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "leaguetst", "scripts"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("uncommitted changes under leaguetst/ or scripts/: commit first so the recorded git hash covers this code")
    table, _ = load_table("ILI")
    etth1, _ = load_table("ETTh1")
    inputs = ["results/ledger.jsonl", str(J2.relative_to(ROOT)), "scripts/ablations/common.py", "scripts/ablations/llm_tests.py",
              *(str(p.relative_to(ROOT)) for p in sorted(RTX.glob("ABL_*_local/*/*.jsonl"))),
              *(str(p.relative_to(ROOT)) for p in sorted((ROOT / "results/runs").glob("ABL_G_*_local/ILI_T24/training.jsonl")))]
    result = {"registered": "AB-P (Holm 3) and AB-G (Holm 2) on ILI held-out AUC, RTX arms with local references (AB2, AB4)",
              "provenance": provenance(__file__, inputs), "AB-P": ab_p(table), "AB-G": ab_g(table),
              "hardware_check": hardware_check(table), "ETTh1": etth1_descriptive(etth1), "gpu_per_arm": gpus(),
              "lower_is_better": "AUC is held-out regret: a negative difference or rho means the arm or the higher level is better",
              "scope": ("variants of the same architect compared with each other; no external baseline appears here, so none "
                        "of this answers RQ1 (E4 does). Secondary tests, not claims under CLAUDE.md rule 6: permutation p and "
                        "Welch intervals, no paired block-bootstrap interval; the addendum-3 analyses are exploratory")}
    write_json(RTX / "RTX_tests.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
