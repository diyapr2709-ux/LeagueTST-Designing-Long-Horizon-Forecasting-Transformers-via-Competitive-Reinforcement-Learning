"""L1 step 3 (docs/LEG2_soumitra.md): diff the independent E4 analysis against scripts/replay.py.

Run `scripts/replay.py --item E4 --table E1 --dataset ILI --horizon 24 --summarize-only` in a separate git
worktree (never committed), then:

    python scripts/independent/diff_replay.py <worktree>

It recomputes the independent per-seed scores with scripts/independent/e4_analysis.py (imported, unchanged),
reads the worktree's results/search/E4/ILI_T24/summary.json and docs/claims/E4.json, and writes
results/independent/E4_replay_diff.json: every statistic both produce, the absolute differences, every
difference above 1e-9, and the decisions of both. Monte Carlo p-values (10 vs 30 seeds) use different
random streams and split counts in the two analyses (200,000 here, 100,000 in leaguetst/eval/stats.py), so
their differences are expected and reported with the combined Monte Carlo standard error. The two analyses
also use different Holm families (replay: over the claims in its file, 2 in family 1 and 1 in family 2;
independent: 6 per family with pending tests as not rejected); this is recorded as a procedure difference.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-9
REPLAY_SPLITS = 100_000   # leaguetst/eval/stats.py permutation_test default, used by scripts/replay.py


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def load_independent():
    spec = importlib.util.spec_from_file_location("e4_analysis", ROOT / "scripts/independent/e4_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: diff_replay.py <worktree with replay --summarize-only outputs>")
    wt = Path(argv[0])
    e4 = load_independent()
    ili = e4.load_tables(ROOT / "results/ledger.jsonl")["ILI"]
    sd, _ = e4.pooled_sd(ili)
    means = {c: float(np.mean(list(s.values()))) for c, s in ili.items()}
    best_config = min(means, key=means.get)
    scores: dict = {}
    for name in ("grpo_league_from_ETTh1_runs.jsonl", "grpo_none_from_ETTh1_runs.jsonl",
                 "transfer_from_ETTh1_runs.jsonl"):
        scores.update(e4.score_runs(ROOT / "results/search/E4/ILI_T24" / name, ili, means[best_config], sd))
    mine = json.loads((ROOT / "results/independent/E4_soumitra.json").read_text())
    summary = json.loads((wt / "results/search/E4/ILI_T24/summary.json").read_text())
    claims = json.loads((wt / "docs/claims/E4.json").read_text())["claims"]

    rows = []

    def add(stat: str, ours: float, theirs: float, note: str = "") -> None:
        rows.append({"statistic": stat, "independent": ours, "replay": theirs, "abs_diff": abs(ours - theirs),
                     "note": note})

    add("ILI pooled seed SD", sd, summary["table"]["seed_sd_pooled"])
    add("ILI best 3-seed mean", means[best_config], summary["table"]["best_true"])
    best_match = best_config == summary["table"]["best_id"]
    for method, per in summary["methods"].items():
        if method not in scores:
            continue
        seeds = sorted(scores[method])
        add(f"{method} seeds", float(len(seeds)), float(per["seeds"]))
        add(f"{method} AUC mean", float(np.mean([scores[method][s]["auc"] for s in seeds])), per["auc_sd"])
        for b in ("25", "50"):
            vals = [scores[method][s][f"regret_{b}"] for s in seeds]
            add(f"{method} regret b{b} mean", float(np.mean(vals)), per[b]["mean_sd"])
            add(f"{method} regret b{b} median", float(np.median(vals)), per[b]["median_sd"])

    tests = {"grpo_none_from_ETTh1": ("family_2", "F2 ILI AUC league vs none"),
             "portfolio_from_ETTh1": ("family_1", "F1 ILI AUC league vs portfolio"),
             "tpe_warm_from_ETTh1": ("family_1", "F1 ILI AUC league vs tpe_warm")}
    decisions = []
    for claim in claims:
        src = claim["source"]
        if src["metric"] != "auc" or src["baseline"] not in tests:
            continue
        a, b = src["method"], src["baseline"]
        ours_a = [scores[a][s]["auc"] for s in sorted(scores[a])]
        ours_b = [scores[b][s]["auc"] for s in sorted(scores[b])]
        if len(ours_a) != len(claim["a_values"]) or len(ours_b) != len(claim["b_values"]):
            raise ValueError(f"{a} vs {b}: seed counts differ ({len(ours_a)}/{len(claim['a_values'])}, "
                             f"{len(ours_b)}/{len(claim['b_values'])})")
        add(f"{a} vs {b}: max per-seed AUC diff, {a}",
            max(abs(x - y) for x, y in zip(ours_a, claim["a_values"])), 0.0, "per-seed")
        add(f"{a} vs {b}: max per-seed AUC diff, {b}",
            max(abs(x - y) for x, y in zip(ours_b, claim["b_values"])), 0.0, "per-seed")
        fam, key = tests[b]
        t = mine[fam]["tests"][key]
        mc = t["method"] == "random"
        pv = max(t["p"] * (1 - t["p"]), 1e-12)
        se = math.sqrt(pv * (1 / t["splits"] + 1 / REPLAY_SPLITS)) if mc else 0.0
        add(f"{a} vs {b}: p", t["p"], claim["p"],
            f"Monte Carlo in both ({t['splits']} vs {REPLAY_SPLITS} random splits, different streams; combined "
            f"SE of the difference ~{se:.1e})" if mc else "exact in the independent analysis")
        reject = mine[fam]["holm_with_pending_as_not_rejected"][key]["reject"]
        independent_win = t["lower_regret"] == a and reject
        decisions.append({"test": key, "independent_lower_regret": t["lower_regret"],
                          "independent_holm_reject": reject, "independent_league_win": independent_win,
                          "replay_claims_win": bool(claim["claims_win"]),
                          "agree": bool(claim["claims_win"]) == independent_win})
    over = [r for r in rows if r["abs_diff"] > TOL]
    out = {"worktree_head": git("-C", str(wt), "rev-parse", "HEAD"),
           "worktree_status": git("-C", str(wt), "status", "--porcelain").splitlines(),
           "repo_head": git("-C", str(ROOT), "rev-parse", "HEAD"),
           "holm_families": {"replay": "over the claims in docs/claims/E4.json: 2 in family 1, 1 in family 2",
                             "independent": "6 per family, pending tests as not rejected (A25, A30)"},
           "best_config_match": best_match, "tolerance": TOL, "rows": rows, "above_tolerance": over,
           "decisions": decisions,
           "decisions_agree": all(d["agree"] for d in decisions),
           "j3_decision_independent": mine["j3_decision_A30"]}
    path = ROOT / "results/independent/E4_replay_diff.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} statistics, {len(over)} above {TOL}: " + "; ".join(
        f"{r['statistic']} ({r['independent']:.6g} vs {r['replay']:.6g})" for r in over))
    print("best config match:", best_match, "| decisions agree:", out["decisions_agree"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
