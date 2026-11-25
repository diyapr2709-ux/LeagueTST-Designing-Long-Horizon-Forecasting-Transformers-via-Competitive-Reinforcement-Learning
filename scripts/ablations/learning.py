"""AB-L: did GRPO learn? (docs/prereg_ablations.md AB-L and AB1; docs/LEG2_soumitra.md L3). CPU only.

Reads the E4 training logs and the E3 prompted-Qwen runs, and E1 rows of results/ledger.jsonl for table
values; never writes the ledger. Writes results/ablations/AB-L.json and results/ablations/AB-L.md.

Registered (AB-L): for every E4 policy, the Spearman slope of the group's mean reward over its training
steps; per source table, the league policies' slopes tested against zero with an exact sign-flip
permutation test over policies; Holm over the 2 source tables (ILI source = J1, ETTh1 source = J2).

Definitions fixed here:
* Spearman slope = Spearman's rank correlation between step and reward_mean over the steps with a reward
  (null rewards skipped: one exists, grpo_none_ETTh1_seed108 step 30).
* Sign-flip test: statistic = mean slope over policies; two-sided p = share of the 2^n sign patterns whose
  |mean| is at least the observed |mean| (exact).
* Every policy with a training record enters. The A29 retried run with a record
  (grpo_league_ILI_seed109) is kept, as A29 keeps retried runs in the tests; a sensitivity line leaves it
  out. grpo_none_ILI_seed108 and seed109 have no training record (lost in J1 attempt 1) and cannot enter.
* Descriptive (AB1): best_in_history resets every episode and an episode's budget (3,000 epochs) equals a
  prompted search's, so each policy's final episode is compared with prompted Qwen at the same spend on
  the source table. Final-episode spend = the last row's cumulative training_epochs minus 3,000 times the
  completed episodes. A completed episode spends at most 3,000 (the oracle refuses the charge that would
  pass the budget), so this undercounts the final episode's spend by up to about one full run (<= 30
  epochs here, at most ~90 over three completed episodes) and compares prompted Qwen at a slightly smaller
  spend: a bias toward GRPO. The only check is loose: every episode's first row lies within 3,000 (k-1) and
  3,000 (k-1) + 600.
* Steps: the registration says "over the 60 steps"; each policy logs 57 or 58 updates, because a step whose
  group exhausts the episode budget makes no update and writes no row (for example steps 18, 35 and 52). Prompted Qwen's best at that spend = the lowest table
  value, at the seed it saw, among its lookups with cumulative epochs at most that spend, averaged over
  its 10 seeds. On the ETTh1 source GRPO's best_in_history includes the registered observation-noise
  offset (A18), which the records do not hold; prompted Qwen's value is the noise-free table entry, so
  the ETTh1 comparison favours GRPO by the minimum's noise. On ILI there is no noise.
* Exploratory (not registered, from the critic): a position-adjusted slope, the Spearman slope of each
  step's reward minus the mean reward of its arm and source at the same position within an episode,
  to separate learning from in-context search position.
* Exploratory: reward at each episode's first step (empty history, the same prompt every time), last
  episode against first; the same slopes for the none and uniform arms; on-policy gaps; invalid, repeat
  and unique rates.

Usage:  python scripts/ablations/learning.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
EPISODE_BUDGET = 3000
ALPHA = 0.05
TRAINING = {"ILI": ROOT / "results/runs/E4/soumitra_ETTh1_T336/training.jsonl",    # J1: trained on ILI
            "ETTh1": ROOT / "results/runs/E4/soumitra_ILI_T24/training.jsonl"}     # J2: trained on ETTh1
PROMPTED = {"ILI": ROOT / "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl",
            "ETTh1": ROOT / "results/search/E3/ETTh1_T336/qwen_prompted_runs.jsonl"}
RETRIED_WITH_RECORD = {"grpo_league_ILI_seed109"}
NO_RECORD = ["grpo_none_ILI_seed108", "grpo_none_ILI_seed109"]


def spearman_slope(log: list[dict]) -> tuple[float, int]:
    """Spearman correlation of reward_mean with step, null rewards skipped."""
    pts = [(r["step"], r["reward_mean"]) for r in log if r.get("reward_mean") is not None]
    if len(pts) < 3:
        raise ValueError("fewer than 3 steps with a reward")
    rho = spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic
    return float(rho), len(pts)


def sign_flip_test(values: list[float]) -> dict:
    """Exact two-sided sign-flip test of mean = 0."""
    x = np.asarray(values, dtype=float)
    observed = float(x.mean())
    tol = 1e-12 * max(1.0, abs(observed))
    hits = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(x)):
        total += 1
        hits += abs(float(np.dot(signs, x)) / len(x)) >= abs(observed) - tol
    return {"n": len(x), "mean": observed, "p": hits / total, "patterns": total}


def holm(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    order = sorted(p_values, key=lambda k: p_values[k])
    m, running, stop, out = len(order), 0.0, False, {}
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        reject = bool((not stop) and p_values[name] <= alpha / (m - rank))
        stop = stop or not reject
        out[name] = {"p": float(p_values[name]), "p_holm": float(running), "reject": reject}
    return out


def episodes(log: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in log:
        out.setdefault(int(r["episode"]), []).append(r)
    return out


def position_means(records: list[dict]) -> dict[int, float]:
    """Mean reward_mean at each within-episode position over a set of policies."""
    sums: dict[int, list[float]] = {}
    for rec in records:
        for rows in episodes(rec["log"]).values():
            for i, r in enumerate(rows):
                if r.get("reward_mean") is not None:
                    sums.setdefault(i, []).append(r["reward_mean"])
    return {i: float(np.mean(v)) for i, v in sums.items()}


def position_adjusted_slope(record: dict, pos_mean: dict[int, float]) -> float:
    pts = []
    for rows in episodes(record["log"]).values():
        for i, r in enumerate(rows):
            if r.get("reward_mean") is not None:
                pts.append((r["step"], r["reward_mean"] - pos_mean[i]))
    return float(spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic)


def final_episode(record: dict) -> dict:
    eps = episodes(record["log"])
    for k, rows in eps.items():
        start = rows[0]["training_epochs"]
        lo = EPISODE_BUDGET * (k - 1)
        if not lo <= start <= lo + 600:
            raise ValueError(f"{record['job']['run_name']}: episode {k} starts at {start}, not near {lo}")
    k = max(eps)
    last = eps[k][-1]
    return {"episode": k, "spend": int(last["training_epochs"] - EPISODE_BUDGET * (k - 1)),
            "best_in_history": float(last["best_in_history"])}


def load_table(dataset: str) -> dict:
    table: dict = {}
    with open(ROOT / "results/ledger.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("item") == "E1" and row["dataset"] == dataset:
                table.setdefault(row["config_id"], {})[row["seed"]] = float(row["best_val_mse"])
    return table


def prompted_best_at(records: list[dict], table: dict, spend: int) -> float:
    bests = []
    for rec in records:
        spent, best = 0, np.inf
        for charge in rec["charges"]:
            spent += int(charge[2])
            if spent > spend:
                break
            best = min(best, table[charge[0]][int(charge[1])])
        bests.append(best)
    return float(np.mean(bests))


def analyse() -> dict:
    out: dict = {"by_source": {}, "tests": {}}
    p_values = {}
    for source, path in TRAINING.items():
        records = [json.loads(line) for line in open(path)]
        table = load_table(source)
        prompted = [json.loads(line) for line in open(PROMPTED[source])]
        per_policy = {}
        pos = {arm: position_means([r for r in records if r["job"]["rivals"] == arm])
               for arm in {r["job"]["rivals"] for r in records}}
        for rec in records:
            name = rec["job"]["run_name"]
            slope, n_steps = spearman_slope(rec["log"])
            fin = final_episode(rec)
            eps = episodes(rec["log"])
            first_rewards = [rows[0]["reward_mean"] for _, rows in sorted(eps.items())]
            log = rec["log"]
            per_policy[name] = {
                "arm": rec["job"]["rivals"], "seed": rec["job"]["seed"], "spearman_slope": slope,
                "steps_with_reward": n_steps, "final_episode": fin,
                "position_adjusted_slope": position_adjusted_slope(rec, pos[rec["job"]["rivals"]]),
                "prompted_best_at_same_spend": prompted_best_at(prompted, table, fin["spend"]),
                "episode_first_step_rewards": first_rewards,
                "first_step_reward_last_minus_first": (None if None in (first_rewards[0], first_rewards[-1])
                                                       else first_rewards[-1] - first_rewards[0]),
                "max_onpolicy_mean": max(r["onpolicy_mean"] for r in log),
                "max_onpolicy_gap": max(r["onpolicy_gap"] for r in log),
                "mean_invalid_rate": float(np.mean([r["invalid_rate"] for r in log])),
                "mean_repeat_rate": float(np.mean([r["repeat_rate"] for r in log])),
                "mean_unique_rate": float(np.mean([r["unique_rate"] for r in log])),
                "retried_A29": name in RETRIED_WITH_RECORD,
                "gpu": rec.get("gpu"),
            }
        arms = {}
        for arm in sorted({v["arm"] for v in per_policy.values()}):
            pols = {k: v for k, v in per_policy.items() if v["arm"] == arm}
            slopes = [v["spearman_slope"] for v in pols.values()]
            fe = [v["final_episode"]["best_in_history"] for v in pols.values()]
            pq = [v["prompted_best_at_same_spend"] for v in pols.values()]
            d1 = [v["first_step_reward_last_minus_first"] for v in pols.values()
                  if v["first_step_reward_last_minus_first"] is not None]
            arms[arm] = {"policies": len(pols), "mean_slope": float(np.mean(slopes)),
                         "median_slope": float(np.median(slopes)), "positive_slopes": int(sum(s > 0 for s in slopes)),
                         "mean_position_adjusted_slope": float(np.mean([v["position_adjusted_slope"]
                                                                        for v in pols.values()])),
                         "positive_position_adjusted": int(sum(v["position_adjusted_slope"] > 0
                                                               for v in pols.values())),
                         "logged_updates": sorted({v["steps_with_reward"] for v in pols.values()}),
                         "final_episode_best_mean": float(np.mean(fe)),
                         "prompted_best_same_spend_mean": float(np.mean(pq)),
                         "final_minus_prompted_mean": float(np.mean(np.subtract(fe, pq))),
                         "first_step_reward_last_minus_first_mean": float(np.mean(d1)) if d1 else None,
                         "mean_unique_rate": float(np.mean([v["mean_unique_rate"] for v in pols.values()])),
                         "max_onpolicy_mean": float(max(v["max_onpolicy_mean"] for v in pols.values()))}
        league = {k: v for k, v in per_policy.items() if v["arm"] == "league"}
        name = f"league slopes vs 0, {source} source"
        test = sign_flip_test([v["spearman_slope"] for v in league.values()])
        p_values[name] = test["p"]
        out["tests"][name] = test
        sens = {k: v for k, v in league.items() if not v["retried_A29"]}
        if len(sens) != len(league):
            out["tests"][name]["sensitivity_without_retried"] = sign_flip_test([v["spearman_slope"]
                                                                                for v in sens.values()])
        out["by_source"][source] = {"training_file": str(path.relative_to(ROOT)), "policies": len(per_policy),
                                    "arms": arms, "per_policy": per_policy}
    out["holm"] = holm(p_values)
    out["missing_training_records"] = NO_RECORD
    out["noise_note"] = ("ETTh1 source: GRPO best_in_history includes the A18 observation-noise offset, "
                         "prompted Qwen's best is the noise-free table value; the comparison favours GRPO.")
    out["spend_note"] = ("Final-episode spend is undercounted by up to about one full run per completed episode "
                         "(at most ~90 epochs), so prompted Qwen is compared at a slightly smaller spend: "
                         "a bias toward GRPO. Some final episodes are one or two steps long (147 to 794 epochs) and pull the "
                         "arm means; where the budget-crossing step 60 is unlogged, the final episode is the last "
                         "logged one. The GRPO-vs-prompted comparison is descriptive and untested.")
    out["steps_note"] = ("Registered 'over the 60 steps': each policy logs 57 or 58 updates (the three steps that "
                         "exhaust an episode budget make no update), one null reward skipped "
                         "(grpo_none_ETTh1_seed108 step 30).")
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_md(res: dict, path: Path) -> None:
    lines = ["# AB-L: did GRPO learn?", "",
             "Generated by scripts/ablations/learning.py from results/ablations/AB-L.json. "
             "Secondary to the registered primary endpoints (docs/prereg_ablations.md).", "",
             "## Registered test: league slopes against zero (exact sign flip, Holm over 2)", "",
             "| Source | Policies | Mean Spearman slope | p | p (Holm) | Reject |", "|---|---|---|---|---|---|"]
    for name, t in res["tests"].items():
        h = res["holm"][name]
        lines.append(f"| {name.split(', ')[1]} | {t['n']} | {t['mean']:+.3f} | {t['p']:.4f} | {h['p_holm']:.4f} | "
                     f"{'yes' if h['reject'] else 'no'} |")
        if "sensitivity_without_retried" in t:
            s = t["sensitivity_without_retried"]
            lines.append(f"| {name.split(', ')[1]}, without retried league109 | {s['n']} | {s['mean']:+.3f} | "
                         f"{s['p']:.4f} | (not adjusted) | |")
    lines += ["", res["steps_note"], "", "## Descriptive and exploratory, per arm", "",
              "Position-adjusted slope: exploratory, not registered (reward minus the arm's mean reward at the "
              "same within-episode position).", "",
              "| Source | Arm | n | Mean slope | Slopes > 0 | Position-adjusted slope (> 0) | Final-episode best MSE |"
              " Prompted best, same spend | First-step reward, last minus first episode | Mean unique rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for source, s in res["by_source"].items():
        for arm, a in s["arms"].items():
            d = a["first_step_reward_last_minus_first_mean"]
            lines.append(f"| {source} | {arm} | {a['policies']} | {a['mean_slope']:+.3f} | {a['positive_slopes']} | "
                         f"{a['mean_position_adjusted_slope']:+.3f} ({a['positive_position_adjusted']}) | "
                         f"{a['final_episode_best_mean']:.4f} | {a['prompted_best_same_spend_mean']:.4f} | "
                         f"{'' if d is None else f'{d:+.4f}'} | {a['mean_unique_rate']:.3f} |")
    lines += ["", f"Missing training records (J1 attempt 1, A29): {', '.join(res['missing_training_records'])}.",
              "", res["noise_note"], "", res["spend_note"], ""]
    path.write_text("\n".join(lines))


def main() -> int:
    res = analyse()
    res["script"] = "scripts/ablations/learning.py"
    res["script_sha256"] = sha256(Path(__file__))
    res["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip()
    res["inputs_sha256"] = {str(p.relative_to(ROOT)): sha256(p)
                            for p in [*TRAINING.values(), *PROMPTED.values(), ROOT / "results/ledger.jsonl"]}
    out_dir = ROOT / "results/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AB-L.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    write_md(res, out_dir / "AB-L.md")
    for name, t in res["tests"].items():
        print(name, f"mean {t['mean']:+.4f} p {t['p']:.4f} holm {res['holm'][name]['p_holm']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
