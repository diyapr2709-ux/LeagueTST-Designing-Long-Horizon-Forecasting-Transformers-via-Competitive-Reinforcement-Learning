"""AB-P and AB-G: the registered LLM ablation families, run on the local RTX (AB2, AB4). CPU.

Inputs: results/ablations/rtx/<item>/<table>/ only, plus the E3 and E4 runs for the descriptive hardware
check. AB4 moved the AB2 items there from results/search/ and results/runs/, which scripts/picks.py and
scripts/e5_referee.py read in full. Every arm and both references ran on one card with the same code, so
local references serve local arms (AB1).

AB-P, ILI held-out AUC, Holm over 3: anon against ref, repair against ref, and the scale trend
(0.5B, 1.5B = ref, 3B). AB-G, ILI held-out AUC, Holm over 2: the step trend (0 = ref, the untrained
prompted model; 30; 60 = ABL_G_ref_local; 120) and the KL trend (beta 0, 0.04 = ABL_G_ref_local, 0.1).
A trend is the seed-level Spearman correlation between level rank and AUC with a two-sided permutation p
(llm_tests.spearman_trend, the function of the Mac analysis). A pair is common.compare (permutation p,
exact up to its limit and Monte Carlo beyond, Welch interval, detectable difference). A test with any arm
incomplete counts as not run, keeps its family size and is not rejected (AB1); no test is run on fewer
arms, levels or seeds. Policies stopped by the on-policy check have no value and are left out; retried
policies stay in the tests as they finished (AB1, AB2, A29). ETTh1 is descriptive.

    .venv/bin/python scripts/ablations/llm_registered.py --check   # completeness and settings, no values
    .venv/bin/python scripts/ablations/llm_registered.py           # once, when all 15 arms are complete
    .venv/bin/python scripts/ablations/llm_registered.py --final   # once, only after 28 Nov 18:00 ET (AB4)
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, apply_holm, by_seed, compare, load_table, mean_ci, provenance, sha256, \
    summarise, write_json  # noqa: E402
from llm_tests import spearman_trend  # noqa: E402

from leaguetst.architect import runner  # noqa: E402

RTX = OUT / "rtx"
RESULT = OUT / "LLM-registered.json"
AB2_MANIFEST = RTX / "AB2_records.sha256"
TABLES = {"ILI": "ILI_T24", "ETTh1": "ETTh1_T336"}
SEARCH_SEEDS = 30
POLICIES = 10
CARD = "NVIDIA GeForce RTX 4080 Laptop GPU"  # the AB2 card
AB2_COMMIT = "c2dda8dfa49c6f090816f516285792a00585a79d"  # the commit that registered AB2
# AB4 arms: no job may run after this (the queue enforces it); --final may run only after it.
DEADLINE_UTC = datetime(2026, 11, 28, 23, 0, tzinfo=timezone.utc)  # 28 Nov 18:00 ET (EST)
# Shared code this script imports, as registered in AB4. A change makes the full run refuse.
DIGESTS = {"scripts/ablations/common.py": "46aef3c196864a1f835505e58e2b1bef22940ca03219bb60887514039efa310b",
           "scripts/ablations/llm_tests.py": "08c72a4c46dded669ebff913a45887ffe215caed24d10005fd27f808b34e9239"}

# item: (method name in the runs file, pinned revision, AB4 arm?). Pins from AB2; the 1.5B pin is A25's.
PROMPTED = {
    "ABL_P_ref_local": ("qwen_prompted_1.5b_local", runner.REVISION, False),
    "ABL_P_05b_local": ("qwen_prompted_0.5b_local", "7ae557604adf67be50417f59c2c2f167def9a775", False),
    "ABL_P_3b_local": ("qwen_prompted_3b_local", "aa8e72537993ba99e69dfaafa59ed015b17504d1", False),
    "ABL_P_anon_local": ("qwen_prompted_anon_1.5b_local", runner.REVISION, True),
    "ABL_P_repair_local": ("qwen_prompted_repair_1.5b_local", runner.REVISION, True),
}
# item: (steps, beta, AB4 arm?) as registered; everything else is the E4 recipe (lr 1e-4, league, ETTh1 source).
GRPO = {
    "ABL_G_ref_local": (60, 0.04, False),
    "ABL_G_steps30_local": (30, 0.04, True),
    "ABL_G_steps120_local": (120, 0.04, True),
    "ABL_G_beta0_local": (60, 0.0, False),
    "ABL_G_beta01_local": (60, 0.1, False),
}
GRPO_METHOD = "grpo_league_from_ETTh1"
POLICY_NAMES = {f"grpo_league_ETTh1_seed{100 + s}": s for s in range(POLICIES)}
LOG_KEYS = ("episode", "reward_mean", "loss", "kl", "best_in_history")  # training-log fields compared across levels


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


@functools.lru_cache(maxsize=None)
def commit_problem(git_hash: str, ab4: bool) -> str | None:
    """A record's commit must exist, be clean, and descend from AB2 (AB2 arms) or contain AB4 (AB4 arms)."""
    if git_hash.endswith("-dirty"):
        return f"dirty git hash {git_hash}"
    if _git("cat-file", "-e", f"{git_hash}^{{commit}}").returncode != 0:
        return f"unknown commit {git_hash[:12]}"
    if _git("merge-base", "--is-ancestor", AB2_COMMIT, git_hash).returncode != 0:
        return f"commit {git_hash[:12]} does not descend from AB2"
    if ab4 and "## Amendment AB4" not in _git("show", f"{git_hash}:docs/prereg_ablations.md").stdout:
        return f"commit {git_hash[:12]} predates AB4"
    return None


def read_records(path: Path) -> tuple[list[dict], list[str]]:
    """Rows of a jsonl file and a problem for each line that does not parse."""
    if not path.exists():
        return [], []
    rows, problems = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append(f"{path.name} line {i} does not parse")
    return rows, problems


def _record_problems(records: list[dict], revision: str, ab4: bool) -> list[str]:
    problems = []
    for r in records:
        env = r.get("env") or {}
        tag = r.get("seed", (r.get("job") or {}).get("run_name"))
        if env.get("backend") != "local" or env.get("per_gpu") != 1:
            problems.append(f"{tag}: backend {env.get('backend')} per_gpu {env.get('per_gpu')}")
        if env.get("model_revision") != revision:
            problems.append(f"{tag}: revision {env.get('model_revision')}")
        if env.get("gpu_name") != CARD:
            problems.append(f"{tag}: card {env.get('gpu_name')}")
        problem = commit_problem(str(env.get("git_hash", "")), ab4)
        if problem:
            problems.append(f"{tag}: {problem}")
    return sorted(set(problems))


def prompted_arm(item: str, dataset: str) -> dict:
    """One prompted table (AB2): 30 distinct seeds 0 to 29, one method, the card, the pin, a registered commit."""
    method, revision, ab4 = PROMPTED[item]
    path = RTX / item / TABLES[dataset] / f"{method}_runs.jsonl"
    runs, problems = read_records(path)
    seeds = [int(r["seed"]) for r in runs]
    problems += _record_problems(runs, revision, ab4)
    if {r.get("method") for r in runs} - {method}:
        problems.append(f"other methods {sorted({str(r.get('method')) for r in runs} - {method})}")
    if len(seeds) != len(set(seeds)):
        problems.append("duplicate seeds")
    complete = set(seeds) == set(range(SEARCH_SEEDS)) and not problems
    return {"item": item, "table": TABLES[dataset], "method": method, "seeds": len(set(seeds)), "complete": complete,
            "problems": problems, "runs": runs, "files": [path]}


def grpo_arm(item: str) -> dict:
    """One transfer item (AB2): complete when every policy has an outcome (a training record, or an on-policy
    stop) and every non-stopped policy is deployed. An errors.jsonl row is a problem only if no stop and no later
    finish resolves it: the runner adds a "no adapter to deploy" row for every stopped policy, and a crash row
    stays after a relaunch finishes the job."""
    steps, beta, ab4 = GRPO[item]
    folder = RTX / item / TABLES["ILI"]
    training, p1 = read_records(folder / "training.jsonl")
    errors, p2 = read_records(folder / "errors.jsonl")
    deployed, p3 = read_records(folder / f"{GRPO_METHOD}_runs.jsonl")
    problems = p1 + p2 + p3
    retried_file = folder / "retried.txt"
    retried = sorted(set(retried_file.read_text().split())) if retried_file.exists() else []
    problems += [f"retried.txt names {n}, not a policy of this item" for n in retried if n not in POLICY_NAMES]
    trained = {(r.get("job") or {}).get("run_name") for r in training}
    stopped = {n for e in errors for n in e.get("run_names", []) if runner.ON_POLICY_STOP in e.get("error", "")}
    seeds = {int(r["seed"]) for r in deployed}
    deploy_name = {f"{GRPO_METHOD}:{s}": n for n, s in POLICY_NAMES.items()}

    def resolved(run_name: str) -> bool:
        name = deploy_name.get(run_name, run_name)
        return name in stopped or (name in trained and POLICY_NAMES.get(name) in seeds)

    other = [e for e in errors if runner.ON_POLICY_STOP not in e.get("error", "")]
    unresolved = sorted({n for e in other for n in e.get("run_names", []) if not resolved(n)})
    if unresolved or any(not e.get("run_names") for e in other):
        problems.append(f"errors.jsonl rows not resolved by a stop or a later finish: {unresolved}")
    problems += _record_problems(deployed + training, runner.REVISION, ab4)
    for r in training:
        job = r.get("job") or {}
        try:
            got = (job.get("steps"), float(job.get("beta")), float(job.get("lr")), job.get("rivals"), job.get("dataset"))
        except (TypeError, ValueError):
            got = None
        if got != (steps, float(beta), 1e-4, "league", "ETTh1"):
            problems.append(f"{job.get('run_name')}: steps, beta, lr, rivals, dataset = {got}")
    if {r.get("method") for r in deployed} - {GRPO_METHOD}:
        problems.append("other methods in the deployment file")
    if len(deployed) != len(seeds):
        problems.append("duplicate deployment seeds")
    if trained & stopped:
        problems.append(f"trained and stopped: {sorted(trained & stopped)}")
    missing_outcome = sorted(set(POLICY_NAMES) - trained - stopped)
    undeployed = sorted(n for n, s in POLICY_NAMES.items() if n not in stopped and s not in seeds)
    complete = not (missing_outcome or undeployed or problems)
    files = [folder / "training.jsonl", folder / f"{GRPO_METHOD}_runs.jsonl"]
    files += [p for p in (folder / "errors.jsonl", retried_file) if p.exists()]
    return {"item": item, "steps": steps, "beta": beta, "trained": len(trained), "deployed": len(seeds),
            "stopped": sorted(stopped), "retried": retried, "missing_outcome": missing_outcome,
            "undeployed": undeployed, "complete": complete, "problems": problems, "runs": deployed,
            "training": training, "files": files}


def manifest_problems() -> list[str]:
    """The nine AB2 items' record files must equal the digests frozen in AB4."""
    if not AB2_MANIFEST.exists():
        return [f"{AB2_MANIFEST.name} missing"]
    problems = []
    for line in AB2_MANIFEST.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if not (ROOT / path).exists() or sha256(ROOT / path) != digest:
            problems.append(f"{path} differs from {AB2_MANIFEST.name}")
    return problems


def arms() -> dict:
    out = {f"{item} {d}": prompted_arm(item, d) for item in PROMPTED for d in TABLES}
    out.update({item: grpo_arm(item) for item in GRPO})
    frozen = manifest_problems()
    for arm in out.values():
        ab4 = PROMPTED[arm["item"]][2] if arm["item"] in PROMPTED else GRPO[arm["item"]][2]
        if frozen and not ab4:  # an AB2 arm whose records differ from the frozen manifest
            arm["problems"] += frozen
            arm["complete"] = False
    return out


def check_lines(all_arms: dict) -> list[str]:
    """Counts and settings only; never an AUC or a test."""
    lines = []
    for name, arm in all_arms.items():
        if "seeds" in arm:
            count = f"{arm['seeds']}/{SEARCH_SEEDS} seeds"
        else:
            count = (f"{arm['trained']} trained, {arm['deployed']} deployed, stopped {arm['stopped'] or 'none'}, "
                     f"retried {arm['retried'] or 'none'}")
        state = "complete" if arm["complete"] else "INCOMPLETE"
        lines.append(f"{name:28s} {state:10s} {count}" + (f"  problems: {arm['problems'][:3]}" if arm["problems"] else ""))
    return lines


def values(arm: dict, table) -> dict[int, float]:
    return by_seed(arm["runs"], arm.get("method", GRPO_METHOD), table)


def policy_values(arm: dict, table) -> tuple[dict[int, float], set[int]]:
    """Deployment seed -> AUC for every deployed policy (retried ones included), and the seeds that are stopped
    or retried (the fills below)."""
    names = set(arm["stopped"]) | set(arm["retried"])
    return values(arm, table), {POLICY_NAMES[n] for n in names if n in POLICY_NAMES}


def trend(levels: dict, failed: dict | None = None) -> dict:
    """The registered trend. Each level needs at least two values, or the trend is not run (never on fewer
    levels). Beside it, with stopped or retried policies, the same trend with each of them set to the lowest
    and, separately, the highest value observed in the trend: illustrative fills, not bounds."""
    done = {k: list(v.values()) for k, v in levels.items()}
    short = [k for k, v in done.items() if len(v) < 2]
    if short:
        return {"not_run": f"levels {short} have fewer than two values"}
    out = spearman_trend(done)
    failed = {k: s for k, s in (failed or {}).items() if s}
    if failed:
        observed = [x for v in done.values() for x in v]
        for label, fill in (("fill_lowest", min(observed)), ("fill_highest", max(observed))):
            filled = {}
            for k, v in levels.items():
                seeds = set(v) | failed.get(k, set())
                filled[k] = [fill if s in failed.get(k, set()) else v[s] for s in sorted(seeds)]
            out[label] = {key: spearman_trend(filled)[key] for key in ("rho", "p")}
        out["filled_policies"] = {str(k): sorted(s) for k, s in failed.items()}
    return out


def sign_flip(values_: list[float]) -> float:
    """Exact two-sided sign-flip p for the mean of per-policy statistics (2^n patterns, n <= 10)."""
    x = np.asarray(values_, dtype=float)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(x))))
    return float(np.mean(np.abs(signs @ x) >= abs(x.sum()) - 1e-12))


def step_secondary(untrained: dict[int, float], trained: dict[int, dict[int, float]], all_arms: dict) -> dict:
    """AB4, registered secondary analyses of the step trend, policy as unit, no Holm.
    (i) each policy's mean AUC over the trained levels it completed, against the 30 untrained searches;
    (ii) per policy with all three trained levels, the Spearman correlation of AUC with 30, 60 and 120 steps,
    exact two-sided sign flip over policies; (iii) diagnostic: how far the 30- and 120-step training logs repeat
    the 60-step log of the same training seed over the first 30 steps (the levels are nested, AB4)."""
    from scipy.stats import spearmanr

    policies = sorted(set().union(*[set(v) for v in trained.values()]))
    means = [float(np.mean([trained[k][s] for k in trained if s in trained[k]])) for s in policies]
    full = [s for s in policies if all(s in trained[k] for k in trained)]
    rhos = []
    for s in full:
        rho = spearmanr(sorted(trained), [trained[k][s] for k in sorted(trained)]).statistic
        rhos.append(0.0 if np.isnan(rho) else float(rho))
    out = {"i_trained_vs_untrained": compare(means, list(untrained.values()), 1) if len(means) > 1 else None,
           "ii_per_policy_rho": {"policies": len(full), "mean_rho": float(np.mean(rhos)) if rhos else None,
                                 "rhos": rhos, "p_sign_flip": sign_flip(rhos) if rhos else None}}
    ref = {(r.get("job") or {}).get("run_name"): r.get("log", []) for r in all_arms["ABL_G_ref_local"]["training"]}
    diagnostic = {}
    for item in ("ABL_G_steps30_local", "ABL_G_steps120_local"):
        shares = []
        for r in all_arms[item]["training"]:
            name = (r.get("job") or {}).get("run_name")
            mine = {row["step"]: tuple(row.get(k) for k in LOG_KEYS) for row in r.get("log", []) if row["step"] <= 30}
            theirs = {row["step"]: tuple(row.get(k) for k in LOG_KEYS) for row in ref.get(name, []) if row["step"] <= 30}
            common_steps = sorted(set(mine) & set(theirs))
            if common_steps:
                shares.append(float(np.mean([mine[t] == theirs[t] for t in common_steps])))
        diagnostic[item] = {"policies": len(shares), "share_of_rows_equal_to_60_step_log": mean_ci(shares) if len(shares) > 1 else shares}
    out["iii_nesting_diagnostic"] = diagnostic
    return out


def families(all_arms: dict, table) -> dict:
    ref = all_arms["ABL_P_ref_local ILI"]
    p_tests, g_tests, not_run, secondary = {}, {}, [], {}
    for arm_name in ("anon", "repair"):
        arm = all_arms[f"ABL_P_{arm_name}_local ILI"]
        if arm["complete"] and ref["complete"]:
            a, b = values(arm, table), values(ref, table)
            p_tests[f"{arm_name} vs ref"] = compare([a[s] for s in sorted(a)], [b[s] for s in sorted(b)], 3)
        else:
            not_run.append(f"AB-P {arm_name} vs ref")
    scale = {i: all_arms[f"{item} ILI"] for i, item in enumerate(("ABL_P_05b_local", "ABL_P_ref_local", "ABL_P_3b_local"))}
    if all(arm["complete"] for arm in scale.values()):
        result = trend({i: values(arm, table) for i, arm in scale.items()})
        if "p" in result:
            p_tests["scale trend"] = {**result, "levels_named": ["0.5B", "1.5B (ref)", "3B"]}
        else:
            not_run.append(f"AB-P scale trend ({result['not_run']})")
    else:
        not_run.append("AB-P scale trend")

    step_items = {30: "ABL_G_steps30_local", 60: "ABL_G_ref_local", 120: "ABL_G_steps120_local"}
    if ref["complete"] and all(all_arms[i]["complete"] for i in step_items.values()):
        levels, failed = {0: values(ref, table)}, {}
        for k, item in step_items.items():
            levels[k], failed[k] = policy_values(all_arms[item], table)
        result = trend(levels, failed)
        if "p" in result:
            g_tests["step trend"] = result
            secondary["step trend"] = step_secondary(levels[0], {k: levels[k] for k in step_items}, all_arms)
        else:
            not_run.append(f"AB-G step trend ({result['not_run']})")
    else:
        not_run.append("AB-G step trend")
    kl_items = {0.0: "ABL_G_beta0_local", 0.04: "ABL_G_ref_local", 0.1: "ABL_G_beta01_local"}
    if all(all_arms[i]["complete"] for i in kl_items.values()):
        levels, failed = {}, {}
        for k, item in kl_items.items():
            levels[k], failed[k] = policy_values(all_arms[item], table)
        result = trend(levels, failed)
        if "p" in result:
            g_tests["KL trend"] = result
        else:
            not_run.append(f"AB-G KL trend ({result['not_run']})")
    else:
        not_run.append("AB-G KL trend")
    return {"AB-P": {"family_size": 3, "tests": apply_holm(p_tests, 3)},
            "AB-G": {"family_size": 2, "tests": apply_holm(g_tests, 2)},
            "not_run": not_run, "secondary_no_holm": secondary}


def hardware_check(all_arms: dict, table) -> dict:
    """Descriptive, no test (AB1). The registered prompted check (ABL_P_ref_local against Modal ABL_P_ref) has
    no Modal side, because ABL_P_ref was not launched (AB3); the E3 A10 runs are shown instead, labelled."""
    out = {"prompted_vs_modal_ABL_P_ref": "not run: ABL_P_ref was never launched (AB3, AB4)"}
    ref = all_arms["ABL_P_ref_local ILI"]
    if ref["complete"]:
        e3_runs, _ = read_records(ROOT / "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl")
        e3, local = by_seed(e3_runs, "qwen_prompted", table), values(ref, table)
        both = sorted(set(e3) & set(local))
        out["prompted_vs_E3_A10 (not the registered check; E3 predates the A25 pins)"] = {
            "seeds": len(both), "rtx": mean_ci([local[s] for s in both]), "a10_E3": mean_ci([e3[s] for s in both]),
            "diff": mean_ci([local[s] - e3[s] for s in both])}
    g = all_arms["ABL_G_ref_local"]
    if g["complete"]:
        j2_runs, _ = read_records(ROOT / "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl")
        j2, local = by_seed(j2_runs, GRPO_METHOD, table), values(g, table)
        both = sorted(set(j2) & set(local))
        out["grpo_ref_vs_J2_league"] = {"seeds": len(both), "rtx": mean_ci([local[s] for s in both]),
                                        "a10_and_a10g_J2": mean_ci([j2[s] for s in both]),
                                        "diff": mean_ci([local[s] - j2[s] for s in both])}
    return out


def etth1_descriptive(all_arms: dict, etth1) -> dict:
    return {item: summarise(all_arms[f"{item} ETTh1"]["runs"], etth1) if all_arms[f"{item} ETTh1"]["complete"]
            else "not run" for item in PROMPTED}


def preflight(final: bool) -> None:
    """AB4: computed once, from committed records and the registered code, and --final only after the deadline."""
    if final and datetime.now(timezone.utc) <= DEADLINE_UTC:
        raise SystemExit("--final only after 28 Nov 18:00 ET (AB4)")
    if RESULT.exists():
        raise SystemExit(f"results/ablations/{RESULT.name} exists: the families are computed once (AB4)")
    dirty = _git("status", "--porcelain", "--", "results/ablations/rtx", "scripts/ablations", "scripts/ablate_llm.py",
                 "leaguetst").stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted records or code (AB4):\n{dirty}")
    for path, digest in DIGESTS.items():
        if sha256(ROOT / path) != digest:
            raise SystemExit(f"{path} differs from the digest registered in AB4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="completeness and settings only, no values")
    parser.add_argument("--final", action="store_true", help="after 28 Nov 18:00 ET: incomplete arms count as "
                                                                "not run (AB1, AB4)")
    args = parser.parse_args(argv)
    all_arms = arms()
    print("\n".join(check_lines(all_arms)))
    if args.check:
        return 0
    incomplete = [name for name, arm in all_arms.items() if not arm["complete"]]
    if incomplete and not args.final:
        raise SystemExit(f"{len(incomplete)} arms incomplete ({', '.join(incomplete)}); AB4: the families are computed "
                         "once, when every arm is complete or, with --final, after 28 Nov 18:00 ET")
    preflight(args.final)
    table, _ = load_table("ILI")
    etth1, _ = load_table("ETTh1")
    inputs = ["results/ledger.jsonl", "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl",
              "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl",
              *sorted({str(p.relative_to(ROOT)) for arm in all_arms.values() for p in arm["files"] if p.exists()})]
    result = {"registered": "AB-P and AB-G on the local RTX (AB2, AB4); ETTh1 descriptive",
              "computed_at_utc": datetime.now(timezone.utc).isoformat(), "final_mode": bool(args.final),
              "incomplete_arms": incomplete,
              "arms": {name: {k: v for k, v in arm.items() if k not in ("runs", "files", "training")}
                       for name, arm in all_arms.items()},
              "families": families(all_arms, table), "hardware_check": hardware_check(all_arms, table),
              "ETTh1_descriptive": etth1_descriptive(all_arms, etth1),
              "provenance": provenance(__file__, inputs)}
    write_json(RESULT, result)
    print("wrote", RESULT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
