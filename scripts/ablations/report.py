"""Write docs/ablations.md from results/ablations/*.json. Every number comes from a results file.

    .venv/bin/python scripts/ablations/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGETS, OUT, ROOT, SETTINGS, setting  # noqa: E402

TARGET = ROOT / "docs" / "ablations.md"
MISSING = "Not run."

# Added 26 Nov, after every result was seen (area-chair review B1 to B22, results review R1, R2, R6, R7, R13 to R15):
# wording, scope and descriptive tables only; every number still comes from a results file.
WELCH = "Welch t 95%, unadjusted"
NO_CLAIM = "in-sample, no claim"
SCOPE = ("These numbers describe the replay table (every lookup returns a recorded training run), not live training: "
         "E3b's K7 check of the table against live training failed on ILI, so they hold for the table only.")
SCOPE_SHORT = "Replay table only (E3b's K7 failed on ILI; see the header)."
RULE6 = ("These are secondary tests, not claims under CLAUDE.md rule 6: the intervals are Welch t and the p values unpaired "
         "permutation p, with no paired block-bootstrap interval, and no gate applies (G4 covers E3 and E4 only).")
TRANSFER_SCOPE = ("The GRPO evidence is transfer between two tables (ETTh1 to ILI); within-table GRPO was dropped (docs/PLAN.md A12, A18). "
                  "The CPU baselines search ILI directly, so they have a different information set; only the E4 portfolio and warm-started TPE "
                  "are the registered transfer controls.")


def load(name: str) -> dict | None:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def ci(e: dict, digits: int = 2) -> str:
    return f"{e['mean']:.{digits}f} [{e['low']:.{digits}f}, {e['high']:.{digits}f}]"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def family(tests: dict, size: int, missing: str = "registered test(s) not run") -> list[str]:
    rows = [[name, f"{t['n'][0]} / {t['n'][1]}", f"{t['diff']:+.3f} [{t['low']:+.3f}, {t['high']:+.3f}]", f"{t['p']:.4f}",
             f"{t['p_holm']:.4f}", "yes" if t["reject"] else "no", f"{t['detectable_80']:.2f}"] for name, t in tests.items()]
    n_missing = size - len(tests)
    if n_missing:
        rows.append([f"{n_missing} {missing}", "", "", "", "", "no (counted)", ""])
    return table(["test", "seeds", f"difference (SD units; {WELCH})", "p", "p, Holm", "rejected", "detectable at 80%"], rows)


def provenance_line(data: dict) -> str:
    p = data["provenance"]
    return f"Source: {p['script']} at {p['git_hash'][:12]} (script sha256 {p['script_sha256'][:12]})."


def controller() -> list[str]:
    d = load("AB-C")
    lines = ["## AB-C: factorized-controller ablations (tested, CPU)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += [f"League rivals, {d['seeds']} search seeds per arm; held-out AUC regret on ILI in pooled seed-SD units; lower is better. {SCOPE_SHORT} "
              + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        lines += [f"{dataset}{'' if t['registered'] else f' ({NO_CLAIM}; descriptive, one seed per config)'}:", ""]
        lines += table(["arm", "AUC (t 95%)", "AUC, raw validation MSE (t 95%)", *(f"@{b}" for b in BUDGETS)],
                       [[a, ci(v["auc"]), ci(v["auc_raw_mse"], 5), *(ci(v[str(b)]) for b in BUDGETS)] for a, v in sorted(t["summary"].items(), key=lambda kv: kv[1]["auc"]["mean"])])
        if "tests" in t:
            lines += family(t["tests"], 3)
    return lines


def external() -> list[str]:
    d = load("AB-X")
    lines = ["## AB-X: external HPO baselines (tested, CPU)", ""]
    if not d:
        return lines + [MISSING, ""]
    v = d["run"]["versions"]
    lines += [f"SMAC3 {v.get('smac')}, DEHB {v.get('dehb')}, ConfigSpace {v.get('ConfigSpace')} through the E3 oracle; 30 seeds; "
              f"tested on ILI against E3's random search. {SCOPE_SHORT} " + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        lines += [f"{dataset}{'' if t['registered'] else f' ({NO_CLAIM}; descriptive)'}:", ""]
        lines += table(["method", "AUC (t 95%)", "AUC, raw validation MSE (t 95%)", *(f"@{b}" for b in BUDGETS)],
                       [[a, ci(x["auc"]), ci(x["auc_raw_mse"], 5), *(ci(x[str(b)]) for b in BUDGETS)] for a, x in sorted(t["summary"].items(), key=lambda kv: kv[1]["auc"]["mean"])])
        if "tests" in t:
            lines += family(t["tests"], 3)
        rates = t["repair_rate"]
        lines += ["Share of proposals replaced by the repair rule: " + ", ".join(f"{m} {v:.0%}" if v is not None else f"{m} none (always feasible)" for m, v in rates.items()) + ".", ""]
        if t.get("exploratory"):
            lines += ["Exploratory (AB3 addendum 2): DEHB with the evaluated config's vector stored in its population, against random: "
                      + "; ".join(f"{k} {v['diff']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}], p {v['p']:.3f}" for k, v in t["exploratory"].items()) + ".", ""]
    lines += ["Choices a SMAC or DEHB user would question: SMAC starts from 10 random configs (its default is a Sobol design), refits after every trial (default 8), and maximises EI "
              "over all unevaluated configs; every field is an unordered categorical (the numeric fields are ordinal); SMAC sees a deterministic objective, the one seed value the search run draws, "
              "which is the same noise E3's methods see. Hyperband and DEHB screen at one epoch, whose ranking agrees with the final value only weakly (E2: tau at k = 1 is in AB-R). "
              "SMAC and Hyperband never repair; DEHB does.", ""]
    return lines


def noise_free() -> list[str]:
    d = load("AB-Q")
    lines = ["## AB-Q: noise-free lookups (descriptive, CPU)", ""]
    if not d:
        return lines + [MISSING, ""]
    p = d["plateau"]
    lines += [f"Every lookup returns the 3-seed mean, and regret is scored on the same means. {p['within_1sd']} of {p['configs']} configs lie "
              f"within one seed SD of the best mean and {p['within_2sd']} within two. The last two columns are the share of seed pairs in "
              f"which the method beats random search: with seed noise (E3, held out) and without it. {SCOPE_SHORT} " + provenance_line(d), ""]
    rows = []
    for m, v in sorted(d["noise_free"].items(), key=lambda kv: kv[1]["auc"]["mean"]):
        s = d["separation"].get(m)
        rows.append([m, ci(v["auc"]), f"{d['e3_noisy_heldout_auc'][m]['mean']:.2f}", *(f"{v[str(b)]['mean']:.2f}" for b in BUDGETS),
                     f"{s['e3_prob_better_than_random']:.2f}" if s else "0.50 (itself)", f"{s['noise_free_prob_better_than_random']:.2f}" if s else "0.50 (itself)"])
    lines += table(["method", "noise-free AUC", "E3 AUC, noisy", *(f"noise-free @{b}" for b in BUDGETS), "P(beats random), noisy", "P(beats random), noise-free"], rows)
    lines += ["The noise-free table is searched and scored on the same 3-seed means, so a method that fits the remaining noise of those "
              "means is rewarded for it; a gap here is not evidence of a gap on fresh seeds (the pick reseeds are). "
              f"Left out: {d['left_out']}.", ""]
    h = d["holdout"]
    lines += ["Hold-out check (exploratory, AB3 addendum): lookups are the mean of two seeds and the search's incumbent is scored on the third, each seed held out in turn; "
              f"{h['unit']}. Held seeds {', '.join(str(x) for x in h['held_seeds'])}.", ""]
    lines += [f"Held-out AUC is measured from the best config at the held seed, so it is not comparable with the noise-free and E3 columns above.", ""]
    lines += table(["method", "held-out AUC", "difference from random", "t interval over the 3 hold-outs (2 df)", "difference per held seed", "in-sample difference of the same runs", "distinct configs in 30 final picks (per held seed)", "P(beats random), run level"],
                   [[m, ci(v["auc"]), f"{v['diff_vs_random']['mean']:+.2f} [{v['diff_vs_random']['low']:+.2f}, {v['diff_vs_random']['high']:+.2f}]" if m != "random" else "0 (itself)",
                     f"{v['diff_t_over_holdouts']['low']:+.2f} to {v['diff_t_over_holdouts']['high']:+.2f}" if m != "random" else "",
                     " / ".join(f"{x:+.2f}" for x in v["diff_by_held_seed"].values()) if m != "random" else "",
                     f"{v['in_sample_diff_vs_random']['mean']:+.2f}" if m != "random" else "0 (itself)",
                     " / ".join(str(x) for x in v["distinct_final_configs_of_30_runs"].values()),
                     f"{v['prob_better_than_random']:.2f}" if m != "random" else "0.50 (itself)"] for m, v in sorted(h["methods"].items(), key=lambda kv: kv[1]["auc"]["mean"])])
    lines += ["The hold-out scores one seed and cuts lookups to two, so its per-seed differences are noisy; its interval resamples search seeds only, while the runs share the same three seeds of every config.", ""]
    q = load("LLM")
    if q and q["mps"].get("AB-Q_prompted"):
        e = q["mps"]["AB-Q_prompted"]
        lines += [f"Prompted Qwen on the same noise-free table (Mac, exploratory, {e['n']} seeds): AUC {ci(e['auc'])}.", ""]
    return lines


def one_seed() -> list[str]:
    d = load("AB-N")
    lines = ["## AB-N: one seed per config (descriptive, CPU)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += ["The ILI table is cut to look like ETTh1's (variant etth1_like: ETTh1's observation-noise rule) or to seed 2021 only (variant strict: noise SD is the full ILI table's pooled SD), and every run is scored twice: as the cut "
              f"table would report it, and held out on the full 3-seed table. Optimism is held-out AUC minus reported AUC. {SCOPE_SHORT} " + provenance_line(d), ""]
    for name, v in d["variants"].items():
        r = v["method_ranking"]
        lines += [f"Variant {name}: {v['multi_seed_configs']} configs keep three seeds; noise SD {v['noise_sd']:.4f}; best value reported "
                  f"{v['best_reported']:.4f}, best 3-seed mean {v['best_full']:.4f}. Kendall tau between the method rankings: reported against held-out "
                  f"{r['reported_vs_heldout_kendall']:.2f}, reported against E3's 3-seed replay {r['reported_vs_e3_kendall']:.2f}, held-out against E3 "
                  f"{r['heldout_vs_e3_kendall']:.2f}.", ""]
        lines += [f"Exploratory split (AB3 addendum): the cut table's best value sits {v['reference_shift_sd']:+.2f} SD from the full table's best; optimism = the incumbents' value optimism minus that shift.", ""]
        lines += table(["method", "reported AUC", "held-out AUC", "optimism of regret", "optimism of the incumbent's value", "E3 AUC (3 seeds)"],
                       [[m, ci(x["reported_auc"]), ci(x["heldout_auc"]), ci(x["optimism_auc"]), ci(x["incumbent_value_optimism_auc"]), f"{d['e3_three_seed_auc'][m]['mean']:.2f}"]
                        for m, x in sorted(v["methods"].items(), key=lambda kv: kv[1]["reported_auc"]["mean"])])
    return lines


def referee() -> list[str]:
    d = load("AB-R")
    lines = ["## AB-R: referee variants (descriptive, CPU)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += [f"Kendall tau against the multi-seed mean on the 150 scored configs of E2, 95% bootstrap interval. {SCOPE_SHORT} " + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        e = t["e2_referee"]
        ceiling = t["one_seed_ceiling"]
        lines += [f"{dataset}{f' ({NO_CLAIM})' if dataset == 'ETTh1' else ''}: E2's referee (both feature sets, k = {e['k']}, 300 configs, boosted trees) {e['tau']:.2f} [{e['low']:.2f}, {e['high']:.2f}]. "
                  + (f"One full seed against the mean: {ceiling['tau']:.2f} [{ceiling['low']:.2f}, {ceiling['high']:.2f}]. " if ceiling else
                     f"Only {t['eval_configs_with_3_seeds']} scored configs have three seeds, so the target is mostly the seed-2021 value. ")
                  + "Plain epoch-k ranking: " + ", ".join(f"k = {k} {v['tau']:.2f}" for k, v in t["epoch_k_ranking"].items()) + ".", ""]
        best = {}
        for x in t["variants"]:
            best.setdefault((x["features"], x["size"]), []).append(x)
        rows = []
        for features in ("config", "curves", "both"):
            for size in (100, 300, 1000):
                cells = {}
                for x in best[(features, size)]:
                    cells.setdefault(x["model"], []).append(x)
                rows.append([features, str(size), *(" / ".join(f"{x['tau']:.2f}" for x in sorted(cells[m], key=lambda x: x["k"] or 0)) for m in ("gbt", "ridge", "forest"))])
        lines += table(["features", "training configs", "boosted trees, k = 1 / 2 / 3", "ridge, k = 1 / 2 / 3", "random forest, k = 1 / 2 / 3"], rows)
        ind = [x for x in t["variants"] if x["size"] == 300 and x["model"] == "gbt"]
        if ind[0].get("tau_independent_seeds"):
            lines += [f"Against the mean of the seeds the referee never saw (2022 and 2023, {t['independent_target_configs']} configs; exploratory), boosted trees at 300: "
                      + "; ".join(f"{x['features']}{'' if x['k'] is None else ' k=' + str(x['k'])} {x['tau_independent_seeds']['tau']:.2f}" for x in ind) + ".", ""]
    lines += ["Config-only rows have one value because they do not depend on k. Intervals of the 300-config boosted-trees variants:", ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        rows = [[f"{x['features']}{'' if x['k'] is None else ', k=' + str(x['k'])}", f"{x['tau']:.2f} [{x['low']:.2f}, {x['high']:.2f}]",
                 f"{x['tau_vs_seed2021_final']:.2f}", f"{x['tau_independent_seeds']['tau']:.2f} [{x['tau_independent_seeds']['low']:.2f}, {x['tau_independent_seeds']['high']:.2f}]" if x.get("tau_independent_seeds") else "not computed"]
                for x in t["variants"] if x["size"] == 300 and x["model"] == "gbt"]
        lines += [f"{dataset}{f' ({NO_CLAIM})' if dataset == 'ETTh1' else ''}:", ""] + table(["boosted trees, 300 configs", "tau against the multi-seed mean", "tau against the seed-2021 value", "tau against seeds 2022 and 2023 (exploratory)"], rows)
    return lines


def learned() -> list[str]:
    d, n = load("AB-L"), load("AB-L-null")
    lines = ["## AB-L: did GRPO learn? (tested by Soumitra; calibration here is exploratory and post hoc)", ""]
    if not d:
        return lines + [MISSING, ""]
    for dataset in ("ETTh1", "ILI"):
        t = d["tests"][f"league slopes vs 0, {dataset} source"]
        h = d["holm"][f"league slopes vs 0, {dataset} source"]
        lines.append(f"* {dataset} source{f' ({NO_CLAIM}: the reward is the ETTh1 table the policies train on)' if dataset == 'ETTh1' else ''}: mean Spearman slope {t['mean']:+.3f} over {t['n']} league policies, p {t['p']:.3f}, Holm {h['p_holm']:.3f}, "
                     f"{'rejected' if h['reject'] else 'not rejected'} (results/ablations/AB-L.json).")
    lines += ["", SCOPE_SHORT, ""]
    if n:
        lines += ["The registered test compares the slope with zero, but the search history resets every episode and the untrained model "
                  "already improves within one, so the slope of a model that never learns is not centred on zero. Chaining untrained "
                  "prompted-Qwen searches into 60-step runs gives: " + provenance_line(n), ""]
        rows = []
        for dataset in ("ETTh1", "ILI"):
            for source, v in n["tables"][dataset].items():
                r, a = v["raw"], v["position_adjusted"]
                rows.append([dataset, source, str(v["episodes"]), f"{r['mean']:+.3f}", f"{r['false_positive_rate']['0.025']:.2f} / {r['false_positive_rate']['0.05']:.2f}",
                             f"{r['observed_league_mean']:+.3f}", f"{r['p_one_sided_against_null']:.3f}", f"{a['mean_of_arm_means']:+.3f}",
                             f"{a['observed_league_mean']:+.3f}", f"{a['p_one_sided_against_null']:.4f}"])
        lines += table(["source table", "untrained runs", "episodes", "null slope, raw", "registered test's false-positive rate at 0.025 / 0.05",
                        "observed raw", "p against null, raw", "null slope, position-adjusted", "observed adjusted", "p against null, adjusted"], rows)
        lines += ["The position-adjusted slope (Soumitra's exploratory field) is centred on zero without learning, so it is the safer reading. "
                  "The E3 source has ten searches per table and no recorded step boundaries, so the null is approximate. The Mac source has thirty searches and recorded steps but another machine and code, so the two sources differ for several reasons; both say the registered test is too liberal.", ""]
    return lines


LEVEL_UNITS = {"scale trend": lambda k: {"0.5": "0.5B", "1.5": "1.5B", "3.0": "3B"}.get(k, k),
               "KL trend": lambda k: f"beta {float(k):g}",
               "steps trend": lambda k: f"{k} steps"}


def welch(a: dict, b: dict) -> dict:
    """The unpaired Welch t 95% interval of a minus b from two mean_ci summaries (n, mean, sd), as common.compare computes it."""
    import math

    from scipy.stats import t as student_t

    va, vb = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (a["n"] - 1) + vb**2 / (b["n"] - 1)) if se > 0 else a["n"] + b["n"] - 2
    diff = a["mean"] - b["mean"]
    half = float(student_t.ppf(0.975, df) * se)
    return {"diff": diff, "mean": diff, "low": diff - half, "high": diff + half}


def ci_signed(e: dict, digits: int = 2) -> str:
    return f"{e['mean']:+.{digits}f} [{e['low']:+.{digits}f}, {e['high']:+.{digits}f}]"


def p3(p: float) -> str:
    """p to 3 decimals; below 0.001 as an inequality, never as 0.000."""
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def trend_words(name: str, test: dict) -> str:
    """A trend's levels in words, with units, lowest level first."""
    unit = LEVEL_UNITS.get(name, str)
    return ", ".join(f"{unit(k)} {v['mean']:.2f}" for k, v in by_level(test))


def by_level(test: dict) -> list[tuple[str, dict]]:
    """A trend's levels in numeric order (JSON keys are strings, so 120 would sort before 30)."""
    return sorted(test["levels"].items(), key=lambda kv: float(kv[0]))


def pfmt(p: float) -> str:
    """p to 4 decimals; the Monte Carlo floor is shown as an inequality."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


AMENDMENT_NAMES = {"AB2": "AB2 (c2dda8d)", "AB4": "Soumitra's AB4 (15db163)"}
# The verifier's recomputation (docs/verdicts/ablations_rtx_reviews.md) was of the families at this pin; the digest is of
# json.dumps(registered.families, sort_keys=True) in the LLM-rtx.json it checked (committed in 40804c7).
VERIFIED_PIN = "e6926d8664b8493442f4e372a4cf0349a34d06e0"
VERIFIED_FAMILIES_SHA256 = "aee7655f598b6a46ce1b2ff903126a828b2645d7f52773317bfd8f2e5330f279"


def et(stamp: str | None) -> str:
    """An ISO time as 'D Mon HH:MM ET' (ET is UTC-5 in late November)."""
    from datetime import datetime, timedelta, timezone

    if not stamp:
        return "not recorded"
    dt = datetime.fromisoformat(stamp).astimezone(timezone(timedelta(hours=-5)))
    return f"{dt.day} {dt:%b} {dt:%H:%M} ET"


def when(stamp: str):
    from datetime import datetime

    return datetime.fromisoformat(stamp)


def families_digest(families: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(families, sort_keys=True).encode()).hexdigest()


def verified(g: dict) -> bool:
    return g["source"].get("commit") == VERIFIED_PIN and families_digest(g["families"]) == VERIFIED_FAMILIES_SHA256


def verification_text(g: dict) -> str:
    if verified(g):
        return ("Verification: a verifier recomputed the AB2 and AB4 values of these families independently, and they agree with this computation "
                "and with Soumitra's scripts/ablations/rtx_tests.py (reports in docs/verdicts/ablations_rtx_reviews.md). The raw-MSE column, the "
                "per-policy table and the compute totals were added after that check.")
    return (f"Not verified: the verifier's recomputation (docs/verdicts/ablations_rtx_reviews.md) was of the families at {VERIFIED_PIN[:12]}, "
            "and the families here differ from those.")


def commit_time(g: dict, prefix: str) -> str | None:
    return next((c["time"] for c in (g.get("timeline") or {}).get("commits", []) if c["commit"].startswith(prefix)), None)


def arm_done_times(v: dict) -> list[str]:
    return [t for t in (v.get("done_time"), v.get("done_time_ETTh1")) if t]


def deadline_text(g: dict) -> str:
    """Both deadlines, each with its source, and whether every arm's queue DONE time is before them."""
    main_dl = g.get("deadline_main_AB4")
    late, unknown = [], []
    for kind in ("prompted", "transfer"):
        for v in g["status"][kind].values():
            times = arm_done_times(v)
            if not v.get("done_time"):
                unknown.append(v["item"])
            limits = [x for x in (g.get("deadline"), main_dl if v.get("amendment") == "AB4" else None) if x]
            if any(when(t) > when(d) for t in times for d in limits):
                late.append(v["item"])
    text = (f"Deadlines: {et(g.get('deadline'))} for every arm (AB1, kept by Soumitra's AB4, 15db163), which this computation applies"
            + (f"; {et(main_dl)} for the four AB4 items under Diya's AB4 (2ab2996)" if main_dl else "") + ". ")
    if late:
        return text + f"Finished after a deadline by the queue's DONE time: {', '.join(late)}."
    return text + ("Every arm's queue DONE time is before both (timeline below)." if not unknown else
                   f"No queue DONE time for {', '.join(unknown)}; every recorded DONE time is before both.")


def status_text(g: dict) -> str:
    st = [v for kind in ("prompted", "transfer") for v in g["status"][kind].values()]
    done = sum(v["complete"] for v in st)
    missing = [n for fam in ("AB-P", "AB-G") for n in g["families"][fam]["missing"]]
    etth1 = len(g.get("ETTh1_descriptive") or {})
    if not g["final"]:
        return ("Status: provisional. Reasons: " + "; ".join(g.get("provisional_reasons") or ["not final"]) + ". "
                "The pending tests count as not rejected for now (Holm treats them as p = 1). A rejection shown here holds whatever they show; "
                "only the pending tests, and the decisions of tests not rejected, are still open.")
    head = (f"Status: complete, {done} of {len(st)} tested ILI arms" if not missing else
            f"Status: closed by AB1's deadline, {done} of {len(st)} tested ILI arms; not run, counted as not rejected: {', '.join(missing)}")
    return (head + f" (and {etth1} ETTh1 tables, descriptive); Anirudh's independent computation, not the result of record (below). "
            f"AB5 calls it final because the pinned commit is the local tip of origin/soumitra-leg2 as this clone last fetched it "
            f"({et(g['source'].get('fetch_time'))}); that does not stop Soumitra from pushing before the deadline, and a new tip would reopen it.")


def llm_tests_edits(g: dict) -> str:
    """Review B8, round 2: which llm_tests.py commits came before and after docs/LEG3_soumitra.md, from git times."""
    hist = g.get("llm_tests_history") or []
    leg3_commit, fetched = commit_time(g, "2ab2996"), g.get("main_AB4_fetched")
    if not hist or not leg3_commit:
        return ("llm_tests.py, which llm_registered.py pins, was changed on this branch after AB5 and restored byte-identical to main's 08c72a4c in 943d893.")
    restore = hist[-1] if hist[-1]["commit"].startswith("943d893") else None
    edits = [h for h in hist if h is not restore]
    before = [h for h in edits if when(h["time"]) < when(leg3_commit)]
    after = [h for h in edits if fetched and when(h["time"]) > when(fetched)]
    between = [h for h in edits if h not in before and h not in after]
    fmt = lambda hs: ", ".join(f"{h['commit'][:7]} ({et(h['time'])})" for h in hs)
    parts = []
    if before:
        parts.append(f"{fmt(before)} before docs/LEG3_soumitra.md existed (committed on main with Diya's AB4, {et(leg3_commit)})")
    if between:
        parts.append(f"{fmt(between)} after it was committed but before this clone fetched it")
    if after:
        parts.append(f"{fmt(after)} after this clone fetched it ({et(fetched)}), against its request")
    return ("llm_tests.py, which llm_registered.py pins, was changed on this branch in " + "; in ".join(parts)
            + (f"; {restore['commit'][:7]} ({et(restore['time'])}) restored it byte-identical to main's 08c72a4c" if restore else "") + ".")


def record_note(g: dict) -> str:
    """Not of record: where LLM-registered.json is looked for, and the history of llm_tests.py on this branch (B2, B8)."""
    rf = g.get("record_file") or {}
    at = rf.get("at") or {}
    exists = rf.get("working_tree") or any(v["exists"] for v in at.values())
    checked = [f"{name} at {str(v['commit'])[:7]} ({'present' if v['exists'] else 'absent'})" for name, v in at.items()]
    checked.append(f"this working tree ({'present' if rf.get('working_tree') else 'absent'})")
    found = (("That file exists; where it differs from these tables, it stands and the difference is a finding to report. " if exists else
              "That file is not yet computed, so the registered families are not yet computed of record. ")
             + "Places checked for it: " + "; ".join(checked) + ".")
    sha = g.get("llm_tests_sha256")
    return ("Not the result of record. Diya's AB4 (2ab2996, origin/main) names scripts/ablations/llm_registered.py, run once by Soumitra, as the computation of the "
            f"registered AB-P and AB-G families, writing {rf.get('path', 'results/ablations/LLM-registered.json')}. {found} Soumitra's branch also carries its own "
            "scripts/ablations/rtx_tests.py (e6926d8). AB5 (510633d) was committed before Diya's AB4 but reached GitHub after it. The tables below are an independent "
            "computation of the same registered tests from the same run files. docs/LEG3_soumitra.md asks Anirudh not to read results/ablations/rtx/ before "
            "LLM-registered.json is committed, and not to edit common.py or llm_tests.py until then: the AB2 values here were computed before docs/LEG3_soumitra.md "
            "reached this clone, and the AB4 values after it, on Anirudh's decision, as a check. " + llm_tests_edits(g)
            + (f" Its sha256 when this computation ran: {sha[:8]}." if sha else ""))


def governance_text() -> str:
    return ("Two amendments are both headed AB4, and the report names each by its commit. Soumitra's AB4 (15db163, origin/soumitra-leg2), in force when the "
            "queue started, registers the anon, repair and step arms, their local references, the tests and the 28 Nov 23:59 ET deadline (llm_rtx.py: REGISTERED_AT, "
            "DEADLINE), and runs the policy-level analyses as exploratory. Diya's AB4 (2ab2996, origin/main) names llm_registered.py as the computation of record, makes "
            "analyses (i) to (iii) registered secondary, fixes the reading rule for the step result and an 18:00 ET deadline for the four AB4 items. Which one governs "
            "each element is for Diya to settle in a dated amendment; this report takes the arms, tests and deadline from 15db163 and the reading rule from 2ab2996.")


def timeline_table(g: dict) -> list[str]:
    """Commit times from git and the queue lines of the status files at the pinned commit, in time order (B1)."""
    tl = g.get("timeline") or {}
    events = [(c["time"], f"{c['label']}; commit {c['commit'][:7]}") for c in tl.get("commits", []) if c.get("time")]
    for q in tl.get("queue", []):
        source = Path(q["file"]).name
        text = {"QUEUE_START": f"queue started at {q['detail'].split()[-1][:7] if q['detail'] else '?'}",
                "DONE": f"DONE {q['name']} ({q['detail']})", "INTERRUPTED": f"INTERRUPTED {q['name']}: {q['detail']}",
                "STALE_LOCK": f"stale lock: {q['detail']}", "QUEUE_END": "queue ended"}.get(q["event"], q["event"])
        events.append((q["time"], f"{text} [{source}]"))
    if g["source"].get("fetch_time"):
        events.append((g["source"]["fetch_time"], "this clone last fetched origin/soumitra-leg2"))
    if g.get("computed_at"):
        events.append((g["computed_at"], "this computation"))
    if not events:
        return []
    rows = [[et(t), text] for t, text in sorted(events, key=lambda e: when(e[0]))]
    return ["Timeline (commit times from git; queue lines from results/logs/l4_queue_status.txt and ab4_queue_status.txt at the pinned commit; "
            "START lines left out):", ""] + table(["time", "event"], rows)


def disclosure_text(g: dict) -> list[str]:
    """Diya's AB4 report disclosure, verbatim, with the two registration times beside it (B7)."""
    d = (g.get("disclosure_main_AB4") or {}).get("text")
    out = []
    if d:
        out.append(f"Disclosure fixed by Diya's AB4 (2ab2996), quoted verbatim: \"{d}\" By the timeline, the arms' registration is Soumitra's AB4, committed "
                   f"{et(commit_time(g, '15db163'))}, and Diya's AB4 was committed {et(commit_time(g, '2ab2996'))}.")
    main_time = commit_time(g, "2ab2996")
    if main_time:
        before = [v["item"] for kind in ("prompted", "transfer") for v in g["status"][kind].values()
                  if v.get("amendment") == "AB4" and v.get("done_time") and when(v["done_time"]) < when(main_time)]
        if before:
            out.append(f"By the queue lines, {', '.join(before)} had finished on ILI before Diya's AB4 was committed, under the queue whose status file Soumitra's "
                       "AB4 names (ab4_queue_status.txt); Diya's 'What existed' paragraph, which says no run of them exists in git, is hers to correct and is not edited here.")
    return [" ".join(out), ""] if out else []


def rtx_checks_table(g: dict) -> list[str]:
    """Arms with their registration, card and run commits (results review R6), DONE time, completeness and checks."""
    short = lambda xs: ", ".join(x[:7] for x in xs) or "-"
    rows = []
    for kind in ("prompted", "transfer"):
        for v in g["status"][kind].values():
            n = v.get("searches", v.get("deployed", 0))
            cards = ", ".join(v.get("gpu", [])) or "-"
            commits = short(v.get("commits", []))
            if kind == "transfer":
                cards = f"deployment: {cards}; training: {', '.join(v.get('training_gpu', [])) or '-'}"
                commits = f"deployment: {commits}; training: {short(v.get('training_commits', []))}"
            rows.append([v["item"] + (" ILI" if kind == "prompted" else ""), AMENDMENT_NAMES.get(v.get("amendment"), str(v.get("amendment"))), str(n),
                         ", ".join(map(str, v.get("stopped", []))) or "-", cards, commits, et(v.get("done_time")), "yes" if v["complete"] else "no",
                         ("; ".join(v["problems"]) or "all pass") if n else "not checked (no runs)"])
    return table(["item", "registered in", "searches or deployed policies", "on-policy stops", "card", "run commits", "queue DONE", "complete",
                  "checks (method, backend, revision, registration order, launcher, run-path code, frozen files, settings)"], rows)


def family_rows(fam: str, f: dict) -> list[list[str]]:
    rows = []
    for name, test in f["tests"].items():
        if "rho" in test:
            unit = LEVEL_UNITS.get(name, str)
            levels = "; ".join(f"{unit(k)}: {ci(v)}" for k, v in by_level(test))
            det = next((r["detectable_80"] for r in (load("ST-power") or {}).get("rows", []) if r["comparison"].startswith(f"{fam} {name} on the RTX")), None)
            stat = f"rho {test['rho']:+.2f} over {test['n']} values ({levels})" + (f"; detectable rho at 80% about {det:.2f} (Fisher z, approximate)" if det else "")
        else:
            stat = f"{test['diff']:+.3f} [{test['low']:+.3f}, {test['high']:+.3f}] ({test['n'][0]} / {test['n'][1]} seeds; detectable {test['detectable_80']:.2f})"
        rows.append([name, stat, pfmt(test["p"]), pfmt(test["p_holm"]), "yes" if test["reject"] else "no"])
    return rows + [[name, f["missing_reasons"][name], "", "", "no (counted)"] for name in f["missing"]]


FAMILY_HEADER = ["test", f"result (pairs: difference in SD units, {WELCH}; trends: level means, t 95%)", "p", "p, Holm", "rejected"]


def arm_label(key: str) -> str:
    kind, arm = key.split(" ", 1)
    return {"ref": "1.5B ref" if kind == "prompted" else "60 steps, beta 0.04 (ref)", "steps30": "30 steps", "steps120": "120 steps",
            "beta0": "beta 0", "beta01": "beta 0.1", "anon": "1.5B anon", "repair": "1.5B repair"}.get(arm, arm) + f" ({kind})"


REPAIR_DEFINITION = ("The repair arm is the 1.5B model without constrained decoding: a proposal over the parameter cap is replaced by the nearest feasible "
                     "config not yet in the search's history (leaguetst/architect/episode.py), as TPE's proposals are.")


def load_period(g: dict) -> dict | None:
    """Results re-review N2: the RTX reboot from the INTERRUPTED queue line (its boot time if logged); c7e4bc9's message records the load after it."""
    import re

    for q in (g.get("timeline") or {}).get("queue", []):
        if q["event"] == "INTERRUPTED":
            m = re.search(r"boot (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", q["detail"])
            start = f"{m.group(1)}T{m.group(2)}{q['time'][-6:]}" if m else q["time"]
            return {"start": start, "logged": q["time"]}
    return None


def ran_after(times: list[str], period: dict | None) -> bool:
    return bool(period) and any(t and when(t) > when(period["start"]) for t in times)


LOAD_TEXT = ("c7e4bc9's commit message records that from that reboot on the card also carried other Windows GPU and CPU load and that per-step time rose; "
             "no end of that period is recorded. These timings are therefore not comparable with the AB2 arms', which ran before it; no statistic depends on them.")


def load_note(g: dict, keys: list[str]) -> list[str]:
    """A footnote for the arms whose queue job finished after the reboot (per-arm ILI table)."""
    period = load_period(g)
    flagged = [arm_label(k) for k in keys if ran_after([g["status"]["transfer" if k.startswith("transfer ") else "prompted"].get(k.split(" ", 1)[1], {}).get("done_time")], period)]
    if not flagged:
        return []
    return [f"† Seconds marked † come from queue jobs that finished after the RTX laptop's reboot at {et(period['start'])} (queue line at {et(period['logged'])}): "
            f"{', '.join(flagged)}. {LOAD_TEXT}"]


def repeat_note(diag: dict, summary: dict, prefix: str = "prompted ") -> str:
    """The prompted arm with the lowest AUC and its repeat rate (a repeated proposal is not looked up and costs no budget)."""
    arms = [k for k in diag if k.startswith(prefix) and summary.get(k)]
    if not arms:
        return ""
    best = min(arms, key=lambda k: summary[k]["auc"]["mean"])
    most = max(arms, key=lambda k: diag[k]["repeat_rate"])
    return (f"The prompted arm with the lowest ILI AUC, {arm_label(best)}, repeats {diag[best]['repeat_rate']:.0%} of its proposals"
            + (", the most of any prompted arm" if most == best else "") + " (a repeated proposal is not looked up and costs no budget).")


def rtx_arm_table(g: dict) -> list[str]:
    """ILI per arm: AUC in SD units and raw MSE (common.summarise), search diagnostics and RevIN-off share (B6, B16)."""
    desc = g["descriptive"]
    summary, diag, share = desc.get("summary_ILI") or {}, g.get("diagnostics_ILI") or {}, desc.get("revin_off_share_ILI") or {}
    keys = [k for k in diag if k in summary] or list(diag)
    if not keys:
        return []
    rows = []
    for k in keys:
        s, x = summary.get(k), diag[k]
        st = g["status"]["transfer" if k.startswith("transfer ") else "prompted"].get(k.split(" ", 1)[1], {})
        mark = " †" if ran_after([st.get("done_time")], load_period(g)) else ""
        rows.append([arm_label(k), str(x["searches"]), ci(s["auc"]) if s else "-", ci(s["auc_raw_mse"], 5) if s else "not shown",
                     f"{x['lookups']['mean']:.0f}", f"{x['repeat_rate']:.2f}", f"{x['repaired_rate']:.2f}", f"{x['invalid_rate']:.3f}",
                     f"{share[k]:.0%}" if k in share else "-", f"{x['seconds']['mean']:.0f}{mark}"])
    out = ["ILI per arm (descriptive; AB5's diagnostics). AUC with a t 95% interval over searches; raw MSE is common.summarise's AUC times the table's pooled seed SD; "
           "repeat, repaired and invalid rates are shares of proposals; the RevIN-off share is of distinct lookups and is shown beside the AUC with no inference drawn from its order. "
           , ""]
    out += table(["arm", "searches", "AUC (t 95%)", "AUC, raw validation MSE (t 95%)", "lookups per search", "repeat rate", "repaired rate", "invalid rate",
                  "RevIN-off share of lookups", "seconds per search"], rows)
    rep, ref = diag.get("prompted repair"), diag.get("prompted ref")
    sentences = []
    if rep and ref:
        sentences.append(f"{REPAIR_DEFINITION} {rep['repaired_rate']:.0%} of its ILI proposals went through that replacement, against {ref['repaired_rate']:.0%} for ref, "
                         "so its comparison with ref measures decoding without the constraint together with the replacement rule, not unconstrained proposals alone.")
    sentences.append(repeat_note(diag, summary))
    sentences += load_note(g, [k for k in keys if k.startswith("transfer ")])
    return out + ([" ".join(x for x in sentences if x), ""] if any(sentences) else [])


def per_policy_table(g: dict) -> list[str]:
    """Each policy's ILI AUC at every trained level (results review R7)."""
    per = g["descriptive"].get("per_seed_ILI") or {}
    cols = [(a, label) for a, label in (("steps30", "30 steps"), ("ref", "60 steps = beta 0.04 (ABL_G_ref_local)"), ("steps120", "120 steps"),
                                         ("beta0", "beta 0"), ("beta01", "beta 0.1")) if a in per]
    if not cols:
        return []
    rhos = ((g["families"]["AB-G"].get("exploratory_policy_level") or {}).get("dose") or {}).get("rhos")
    seeds = sorted({s for a, _ in cols for s in per[a]}, key=int)
    rows = [[f"{s} (training seed {100 + int(s)})", *(f"{per[a][s]:.2f}" if s in per[a] else "-" for a, _ in cols),
             f"{rhos[i]:+.2f}" if rhos and len(rhos) == len(seeds) else "-"] for i, s in enumerate(seeds)]
    return ["Each policy's ILI held-out AUC (common.by_seed, SD units), by deployment seed. The levels share training seeds 100 to 109 and deployment seeds 0 to 9, "
            "so a row is one seed's policies, not independent replications, and the step question has 10 policies, not 30 values; ABL_G_ref_local is at the same "
            "time the 60-step level, the beta 0.04 KL level and the J2 hardware twin. Seeds are compared only within the RTX. ", ""] + \
        table(["deployment seed", *(label for _, label in cols), "per-policy rho of AUC with 30, 60, 120 steps"], rows)


def step_paragraph(g: dict) -> list[str]:
    pl = g["families"]["AB-G"].get("exploratory_policy_level")
    if not pl:
        return []
    u, dose = pl["trained_vs_untrained"], pl["dose"]
    steps = g["families"]["AB-G"]["tests"].get("steps trend")
    return [f"Policy-level step analyses, policy as unit, no Holm. They are registered secondary analyses in Diya's AB4 (2ab2996) and exploratory in Soumitra's AB4 "
            f"(15db163), under which the arms ran. (ii) Dose: the per-policy Spearman correlation of AUC with 30, 60 and 120 steps averages {dose['mean_rho']:+.2f} over "
            f"{dose['policies']} policies, exact sign flip p {dose['p_sign_flip']:.3f}. (i) Trained at all: the ten policies' AUC averaged over their three steps levels, "
            f"against the 30 untrained searches, {u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}] ({WELCH}; permutation p {pfmt(u['p'])}; the test could detect "
            f"{u['detectable_80']:.2f} SD; positive is worse). Diya's reading rule: a dose-response only if (ii) has the registered trend's sign and p at most 0.05; "
            "otherwise, if (i) shows a difference, trained policies differ from the untrained model. 'Shows a difference' is read here as p at most 0.05, which is "
            "Anirudh's operationalisation, not the amendment's. " + step_reading(steps, u, dose), ""]


def step_reading(steps: dict | None, u: dict, dose: dict) -> str:
    """Diya's AB4 reading rule for the step result, fixed before any AB4 value was seen."""
    if steps and dose["p_sign_flip"] <= 0.05 and (dose["mean_rho"] > 0) == (steps["rho"] > 0) and dose["mean_rho"] != 0:
        return "By the reading rule of Diya's AB4 (2ab2996) the step result is read as a dose-response: (ii) has the registered trend's sign and p at most 0.05."
    if u["p"] <= 0.05:
        return ("By the reading rule of Diya's AB4 (2ab2996) the step result is not read as a dose-response; trained policies differ from the untrained model: "
                f"{'worse' if u['diff'] > 0 else 'better'} by {u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}], permutation p {pfmt(u['p'])} ((i)).")
    return "By the reading rule of Diya's AB4 (2ab2996) the step result is read neither as a dose-response nor as a difference between trained and untrained policies."


def etth1_auc(g: dict, arm: str) -> dict | None:
    x = (g.get("ETTh1_descriptive") or {}).get(arm)
    return next(iter(x["summary"].values()))["auc"] if x else None


def scale_scope(g: dict) -> list[str]:
    sc = g["families"]["AB-P"]["tests"].get("scale trend")
    if not sc:
        return []
    lv = [sc["levels"][k]["mean"] for k in ("0.5", "1.5", "3.0")]
    names = {"0.5B": "0.5B", "ref": "1.5B", "3B": "3B"}
    e = {names[k]: etth1_auc(g, k) for k in names if etth1_auc(g, k)}
    ili_order = [n for n, _ in sorted(zip(("0.5B", "1.5B", "3B"), lv), key=lambda kv: kv[1])]
    etth1_order = [k for k, _ in sorted(e.items(), key=lambda kv: kv[1]["mean"])]
    order = ", ".join(f"{k} {e[k]['mean']:.2f}" for k in etth1_order)
    overlap = bool(e) and max(v["low"] for v in e.values()) <= min(v["high"] for v in e.values())
    relation = "the same as on ILI" if etth1_order == ili_order else f"not ILI's order ({', '.join(ili_order)}, best first)"
    return ["Scope of the scale trend: on ILI, for these three Qwen2.5-Instruct checkpoints (0.5B, 1.5B, 3B) and this prompt, under one fixed protocol "
            "(the same prompt, temperature 1.0, constrained decoding, one card); the checkpoints differ in more than size, so the test does not isolate size, "
            f"and three points allow no extrapolation. The change is {lv[1] - lv[0]:+.2f} SD from 0.5B to 1.5B and {lv[2] - lv[1]:+.2f} SD from 1.5B to 3B; no pair was tested. "
            "The levels share search seeds 0 to 29; the unpaired permutation test ignores that pairing. "
            + (f"On ETTh1 ({NO_CLAIM}; descriptive, no test) the order by AUC is {order}, {relation}" + (", and every pair of intervals overlaps" if overlap else "") + ". " if order else "")
            + "The 3B checkpoint is under the Qwen Research License (non-commercial; docs/neurips_checklist.md, item 12); releasing its generated proposals may need that licence's notice.", ""]


ANON_DEFINITION = ("The anon arm replaces the field names by neutral letters (field_a, field_b, ...) and drops the field descriptions from the prompt "
                   "(leaguetst/architect/prompt.py, ANONYMOUS and DESCRIPTIONS), and the constrained-decoding trie is built on the same renamed JSON keys "
                   "(leaguetst/architect/constrained.py, canonical_text), so its token paths change too; the task sentence and the allowed values stay. A difference "
                   "therefore mixes prior knowledge of the names with prompt content and tokenisation.")


def anon_scope(g: dict) -> list[str]:
    a = g["families"]["AB-P"]["tests"].get("anon vs ref_local")
    if not a:
        return []
    ea, er = etth1_auc(g, "anon"), etth1_auc(g, "ref")
    ph = programme_row("AB-P", "anon vs ref_local")
    return [f"Scope of the anon result: on ILI, for this prompt and the 1.5B checkpoint. {ANON_DEFINITION}"
            + (f" On ETTh1 ({NO_CLAIM}; descriptive, no test) anon has AUC {ea['mean']:.2f} and ref {er['mean']:.2f}." if ea and er else "")
            + (f" Under an exploratory, post hoc Holm over all {ph['size']} registered ablation tests, its p is {pfmt(ph['programme_p_holm'])}, "
               f"{'rejected' if ph['programme_reject'] else 'not rejected'} (section below); the registered decision is the one within AB-P." if ph else ""), ""]


def programme_row(fam: str, name: str) -> dict | None:
    d = load("programme-holm")
    if not d:
        return None
    r = next((r for r in d["rows"] if r["family"] == fam and r["test"] == name), None)
    return {**r, "size": d["size"]} if r else None


def hardware_table(g: dict) -> list[str]:
    """AB1's hardware check and AB5's comparisons: arm means on the shared seeds, never paired across cards (results review R6)."""
    desc = g["descriptive"]
    rows = []
    for key, label, cards in (("hardware_G_ref_local_vs_J2", "ABL_G_ref_local against J2 league, AB1 hardware check", ("RTX", "A10/A10G")),
                              ("prompted_ref_local_vs_mac", "ABL_P_ref_local against ABL_P_ref_mps, AB5", ("RTX", "Mac")),
                              ("prompted_ref_local_vs_E3_A10", "ABL_P_ref_local against E3 prompted, AB5", ("RTX", "A10"))):
        v = desc.get(key)
        if v:
            rows.append([label, str(v["seeds"]), f"{ci(v['a'])} ({cards[0]})", f"{ci(v['b'])} ({cards[1]})", ci_signed(welch(v["a"], v["b"]))])
    if not rows:
        return []
    return ["AB1 hardware check and AB5 comparisons (descriptive, no test): each arm's mean AUC (t 95%) over the seeds both arms ran. Seeds are not paired across "
            "cards (an equal seed does not reproduce a search on another card), so the difference is given with the unpaired Welch interval of the two sets of "
            "searches. These intervals do not show that the RTX matches the other machines; the samplers differ, and the E3 searches predate the A25 environment pins.", ""] + \
        table(["comparison", "seeds", "first (card)", "second (card)", f"difference ({WELCH}, unpaired)"], rows)


def registered_llm() -> list[str]:
    """AB2 and AB4: the registered AB-P and AB-G families on Soumitra's RTX items (results/ablations/LLM-rtx.json)."""
    d = load("LLM-rtx")
    lines = ["## AB-P and AB-G on Soumitra's RTX (AB2 and Soumitra's AB4): Anirudh's independent computation", ""]
    if not d or "families" not in d.get("registered", {}):
        return lines + [MISSING, ""]
    g = d["registered"]
    src = g["source"]
    lines += [f"{status_text(g)} {deadline_text(g)} "
              f"Registered in AB2 (c2dda8d) and Soumitra's AB4 (15db163), both on origin/soumitra-leg2, under which every arm ran; the scale trend's statistic (Spearman of model-size rank with AUC, two-sided permutation p) "
              "was fixed in AB5 (510633d, this branch), after the AB2 runs existed and before Anirudh had seen any AB2 value. "
              f"Run files read from {src['ref']} at {str(src['commit'])[:12]} "
              f"({len(src['files'])} files, each with its source, commit and sha256 in results/ablations/LLM-rtx.json; AB2's move manifest matches for "
              f"{sum(src['ab2_manifest_matches'].values())} of {len(src['ab2_manifest_matches'])} files). Every reference is local to the same card (AB1; Soumitra's AB4, 15db163). "
              "ILI held-out AUC, pooled seed-SD units, lower is better. For a trend, positive rho means a higher (worse) AUC at the higher level: more steps, a larger KL weight, a larger model. "
              f"{RULE6} {SCOPE_SHORT} " + provenance_line(d), ""]
    lines += [verification_text(g), ""]
    lines += [record_note(g), ""]
    lines += [governance_text(), ""]
    lines += timeline_table(g) + disclosure_text(g)
    lines += rtx_checks_table(g)
    lines += [f"Teammate branch: local tip {str(src.get('tip'))[:12]} (committed {src.get('tip_time')}); this clone last fetched at {src.get('fetch_time')}.", ""]
    if not src.get("pinned_is_tip", True):
        lines += [f"The pinned commit {str(src['commit'])[:12]} is not the local tip of origin/soumitra-leg2 ({str(src.get('tip'))[:12]}), so these results stay provisional until the pin is moved.", ""]
    lines += ["AB-P, Holm over 3:", ""] + table(FAMILY_HEADER, family_rows("AB-P", g["families"]["AB-P"]))
    lines += scale_scope(g) + anon_scope(g)
    lines += ["AB-G, policy-level analyses first, then the registered family (Holm over 2).", ""]
    lines += step_paragraph(g) + per_policy_table(g)
    lines += ["AB-G, Holm over 2:", ""] + table(FAMILY_HEADER, family_rows("AB-G", g["families"]["AB-G"]))
    if g["families"]["AB-G"]["tests"]:
        lines += ["The AB-G levels share training seeds 100 to 109 and deployment seeds 0 to 9 (and the steps levels share level 0), so they are dependent. "
                  "Diya's AB4 (2ab2996, main): training has no step-dependent schedule, so a 30-step policy is, up to GPU nondeterminism, a prefix of the 60- and 120-step policies of its seed, "
                  "and the step-trend permutation p is anti-conservative. Main's docs/ISSUES.md (26 Nov) quotes a simulation by that amendment's critic of how far the false-positive rate "
                  "exceeds the nominal level; no committed simulation here reproduces it, so no figure is given. "
                  "The step trend and its Holm decision are reported as registered, with this caveat. The KL levels share seeds too; with balanced levels Diya's AB4 states that this makes the KL test conservative. "
                  "The detectable rho values are Fisher-z approximations, optimistic with tied levels. Diya's AB4 diagnostic (iii), the share of training-log rows shared with the "
                  "60-step log, is not computed here, so GPU nondeterminism is not measured.", ""]
    rnd = (load("LLM") or {}).get("mps", {}).get("revin", {}).get("committed_ILI", {}).get("random (E3)")
    if rnd:
        lines += [f"For context (post hoc, no test): E3's random search has ILI AUC {ci(rnd['auc'])} on the same scale.", ""]
    lines += rtx_arm_table(g) + hardware_table(g)
    if g.get("ETTh1_descriptive"):
        lines += [f"ETTh1 ({NO_CLAIM}; descriptive, one seed per config).", ""]
        lines += table(["arm", "AUC (t 95%)", "AUC, raw validation MSE (t 95%)", *(f"@{b}" for b in BUDGETS)],
                       [[{"ref": "1.5B (ref)"}.get(arm, arm), ci(next(iter(x["summary"].values()))["auc"]), ci(next(iter(x["summary"].values()))["auc_raw_mse"], 5),
                         *(ci(next(iter(x["summary"].values()))[str(b)]) for b in BUDGETS)] for arm, x in g["ETTh1_descriptive"].items()])
    tr = [(v["item"], v["training"]) for v in g["status"]["transfer"].values() if v.get("training", {}).get("policies")]
    if tr:
        lines += ["Training (RTX transfer items): " + "; ".join(f"{k} {x['policies']} policies, largest mean on-policy gap {x['onpolicy_mean_max']:.4f} (limit 0.05), errors {len(x['errors'])}" for k, x in tr) + ".", ""]
    return lines


def programme_section() -> list[str]:
    """Exploratory, post hoc: Holm over every registered ablation test (results/ablations/programme-holm.json, review B12)."""
    d = load("programme-holm")
    lines = ["## Exploratory: one Holm over every registered ablation test (post hoc)", ""]
    if not d:
        return lines + [MISSING, ""]
    sizes = ", ".join(f"{k} {v}" for k, v in d["family_sizes"].items())
    lines += [f"Added after every result was seen, as a sensitivity reading only: the registered decisions are the per-family Holm decisions, and they stand. One Holm over "
              f"all {d['size']} registered ablation tests ({sizes}), from the p values already in the result files. {RULE6} " + provenance_line(d), ""]
    lines += table(["family", "test", "p", "p, Holm in its family (registered)", "rejected in its family", f"p, Holm over {d['size']}", f"rejected over {d['size']}"],
                   [[r["family"], r["test"], pfmt(r["p"]), pfmt(r["family_p_holm"]), "yes" if r["family_reject"] else "no", pfmt(r["programme_p_holm"]),
                     "yes" if r["programme_reject"] else "no"] for r in sorted(d["rows"], key=lambda r: r["p"])])
    flipped = [f"{r['family']} {r['test']}" for r in d["rows"] if r["family_reject"] and not r["programme_reject"]]
    lines += [(f"Rejected in their family but not over all {d['size']}: {', '.join(flipped)}. " if flipped else f"Every family rejection also holds over all {d['size']}. ")
              + f"The tests {', '.join(d['tests_sharing_ABL_P_ref_local'])} all use the same 30 ABL_P_ref_local searches (as the reference, as the 1.5B level or as step level 0), "
              "so their errors are correlated across AB-P and AB-G.", ""]
    return lines


def llm() -> list[str]:
    d = load("LLM")
    lines = ["## AB-P and AB-G on the Mac (exploratory, AB3)", ""]
    if not d:
        return lines + [MISSING, ""]
    m = d["mps"]
    lines += ["The Modal items (ABL_P_ref, ABL_P_anon, ABL_P_repair, ABL_G_steps30, ABL_G_steps120) were never launched (AB3); their questions are registered on the RTX "
              "(AB2 and Soumitra's AB4, 15db163; the RTX section above). ",
              "Everything below ran on an Apple M4 Pro (torch MPS), each Mac arm against its own reference on the same machine (the A10 columns of the hardware check are committed runs). It is exploratory (AB1, AB3): "
              f"it replaces no registered family. {RULE6} {SCOPE_SHORT} Raw MSE is not shown for the Mac ILI arms: results/ablations/LLM.json, written by llm_tests.py, "
              "holds the AUC in seed-SD units only, and llm_tests.py stays as main pins it. " + provenance_line(d), ""]
    p = m["AB-P"]
    lines += ["### AB-P on the Mac: prompted Qwen, ILI held-out AUC", "", "Finished searches: " + ", ".join(f"{k} {v}" for k, v in p["complete"].items()) + ".", ""]
    if p["tests"]:
        lines += family(p["tests"], 3, "test not part of this exploratory set: the scale trend (moved to Soumitra by AB1)")
    if p["diagnostics"]:
        rep = p["diagnostics"].get("repair")
        lines += ["ILI diagnostics:", ""] + table(["arm", "searches", "lookups", "repeat rate", "repaired rate", "invalid rate"],
                       [[a, str(x["searches"]), f"{x['lookups']['mean']:.0f}", f"{x['repeat_rate']:.2f}", f"{x['repaired_rate']:.2f}", f"{x['invalid_rate']:.2f}"] for a, x in p["diagnostics"].items()])
        if rep:
            lines += [f"{REPAIR_DEFINITION} On the Mac {rep['repaired_rate']:.0%} of its ILI proposals went through that replacement, against "
                      f"{p['diagnostics']['ref']['repaired_rate']:.0%} for ref, so its comparison with ref measures decoding without the constraint together with the "
                      "replacement rule, not unconstrained proposals alone.", ""]
    if m.get("ETTh1_descriptive"):
        lines += [f"ETTh1 ({NO_CLAIM}; descriptive, one seed per config):", ""]
        if m.get("ETTh1_differences"):
            d_ = m["ETTh1_differences"]
            lines += table(["ETTh1 arm", f"AUC difference from ref (SD units; {WELCH}; no test)"],
                           [[k, f"{v['diff']:+.2f} [{v['low']:+.2f}, {v['high']:+.2f}] (30 seeds per arm)"] for k, v in d_.items()])
            lines += ["Both signs are the reverse of ILI: on ETTh1 the anonymised prompt does slightly better and repair worse. ETTh1 has one seed per config, so these are in-sample and no test or p value is given; no mechanism is claimed.", ""]
        rows = []
        for arm, x in m["ETTh1_descriptive"].items():
            for method, s in x["summary"].items():
                rows.append([arm, ci(s["auc"]), ci(s["auc_raw_mse"], 5), *(ci(s[str(b)]) for b in BUDGETS)])
        lines += table(["arm", "AUC (t 95%)", "AUC, raw validation MSE (t 95%)", *(f"@{b}" for b in BUDGETS)], rows)
    g = m["AB-G"]
    lines += ["### AB-G on the Mac: GRPO training steps, ETTh1 to ILI", "", "Finished runs per level: " + ", ".join(f"{k} steps: {v}" for k, v in g["complete"].items()) + ".", ""]
    if g["tests"]:
        t = g["tests"]["step trend"]
        lines += ["ILI held-out AUC per level. The trained levels are checkpoints 30, 60 and 120 of the same ten policies (one training run each), not "
                  f"independent replications; level 0 is the untrained model.", ""]
        lines += table(["steps", "runs", "ILI held-out AUC (t 95%)"], [[k, str(v["n"]), ci(v)] for k, v in sorted(t["levels"].items(), key=lambda kv: int(kv[0]))])
        u, dose = m["policy_level"]["trained_vs_untrained"], m["policy_level"]["dose"]
        u, dose, pair = m["policy_level"]["trained_vs_untrained"], m["policy_level"]["dose"], m["paired"]
        lines += [f"Trained at all (exploratory, AB3 addendum 3): the ten policies' AUC, averaged over their three checkpoints, is {u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}] ({WELCH}) against the 30 untrained searches "
                  f"(permutation p {pfmt(u['p'])}; the test could detect {u['detectable_80']:.2f} SD). Positive is worse. Against the untrained search of the same deployment seed the gap is {ci(pair['gap'])} SD, "
                  f"worse for {pair['worse']} of {pair['policies']} policies (addendum 4).", ""]
        lines += ["Scope: policies trained with league rivals on the ETTh1 table (one seed per config) and deployed on ILI. It shows that they do worse on ILI than the untrained model. It does not show that GRPO or the league is harmful: "
                  "there is no ILI-source arm, no same-table deployment and no uniform or none arm on the Mac. Policies' checkpoints are dependent (one training run each).", ""]
        rv = m["revin"]
        lines += ["RevIN preference (descriptive, addendum 4). The table gives the share of the lookups on ILI with RevIN off, and the AUC of the same committed runs beside it. "
                  "The committed runs ran on other cards (E3 and E4 on Modal A10 and A10G); they are shown beside the Mac levels, not paired with them.", ""]
        rows = [[f"level {k}", f"{v:.0%}", "", ""] for k, v in sorted(rv["mac_share_off_ILI"].items(), key=lambda kv: int(kv[0].split()[0]))] + \
               [[k, f"{v['share_off']:.0%}", ci(v["auc"]), str(v["runs"])] for k, v in rv["committed_ILI"].items()]
        lines += table(["runs", "RevIN-off share of ILI lookups", "ILI AUC, committed runs (t 95%)", "runs in group"], rows)
        b = rv["best_regret_by_revin"]
        lines += [f"The best RevIN-off config on ILI has regret {b['ILI_T24']['revin_off']:.2f} SD (RevIN-on best: {b['ILI_T24']['revin_on']:.2f}), and on ETTh1 ({NO_CLAIM}) the best RevIN-on config has {b['ETTh1_T336']['revin_on']:.1f} SD "
                  f"(RevIN-off best: {b['ETTh1_T336']['revin_off']:.2f}). Trained policies, at 30 steps already, look up almost only RevIN-off configs, so a preference that is right on ETTh1 costs on ILI. "
                  "This is consistent with the transfer loss but is not tested: no arm removes the preference while keeping the training.", ""]
        lines += [f"Dose (exploratory, addendum 3): the per-policy Spearman correlation of AUC with 30, 60, 120 steps averages {dose['mean_rho']:+.2f} over {dose['policies']} policies, exact sign flip p {dose['p_sign_flip']:.3f} "
                  "(three ordered points, little power).", ""]
        sd = m["step_diagnostics"]
        lines += table(["checkpoint", "searches", "lookups", "repeat rate", "invalid rate", "logged updates per policy"],
                       [[k, str(v["searches"]), f"{v['lookups']['mean']:.0f}", f"{v['repeat_rate']:.2f}", f"{v['invalid_rate']:.2f}", ci(v["logged_updates"], 1)] for k, v in sorted(sd.items(), key=lambda kv: int(kv[0]))])
        ordered = sorted(sd.items(), key=lambda kv: int(kv[0]))
        updates = ", ".join(f"{v['logged_updates']['mean']:.0f}" for _, v in ordered)
        lines += [f"A step with no budget left logs no update, so the levels are about {updates} updates at {', '.join(k for k, _ in ordered)} steps.", ""]
        lines += [f"AB-G's registered statistic, computed on these Mac runs (Spearman of the level with AUC over 60 values): rho {t['rho']:+.2f}, p {pfmt(t['p'])}, Holm over 2 {pfmt(t['p_holm'])}. "
                  "This p is not valid (AB3 addendum 3): levels 30, 60 and 120 are checkpoints of the same policies, and level 0 (untrained) drives the rank correlation. It is therefore not reported as a rejection. "
                  f"With 10 seeds per level: rho {m['step_trend_10_per_level']['rho']:+.2f}, p {pfmt(m['step_trend_10_per_level']['p'])} (same caveat). "
                  f"Trained levels only (30, 60, 120): rho {m['step_trend_trained_levels_only']['rho']:+.2f}, p {m['step_trend_trained_levels_only']['p']:.3f} (same caveat). ", ""]
    if g["training"]["policies"]:
        tr = g["training"]
        lines += [f"Training: {tr['policies']} policies, largest mean on-policy gap {tr['onpolicy_mean_max']:.4f} (limit 0.05), mean unique rate "
                  f"{tr['unique_rate_mean']:.2f}, errors: {len(tr['errors'])}.", ""]
    if d["hardware_check"]:
        lines += ["### Hardware check (no test)", "", "Each arm's mean AUC (t 95%) over the seeds both ran. Seeds are not paired across cards (an equal seed does not "
                  "reproduce a search on another card), so the difference is given with the unpaired Welch interval of the two sets of searches.", ""]
        rows = [[k, str(v["seeds"]), ci(v["mps"]), ci(v.get("a10_E3") or v.get("a10_J2")), ci_signed(welch(v["mps"], v.get("a10_E3") or v.get("a10_J2")))]
                for k, v in d["hardware_check"].items()]
        lines += table(["run", "shared seeds", "Mac", "A10 or A10G", f"Mac minus A10 or A10G ({WELCH}, unpaired)"], rows)
        h = d["hardware_check"]
        if "prompted_ILI" in h and "grpo_league_60_steps" in h:
            gap_a10 = h["grpo_league_60_steps"]["a10_J2"]["mean"] - h["prompted_ILI"]["a10_E3"]["mean"]
            gap_mac = h["grpo_league_60_steps"]["mps"]["mean"] - h["prompted_ILI"]["mps"]["mean"]
            lines += ["This is not a hardware test: E3 predates the A25 environment record, the samplers and random streams differ (equal seeds share only the Oracle draws), and J2 ran on five A10 and five A10G cards. "
                      f"No conclusion rests on it: the trained-minus-untrained gap on ILI is {gap_a10:+.2f} SD in the committed runs and {gap_mac:+.2f} SD on the Mac for the same seeds.", ""]
    return lines


def power() -> list[str]:
    d = load("ST-power")
    lines = ["## ST: what each comparison could detect (descriptive)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += ["Smallest true difference that the test would find with 80% power, with the observed per-arm SDs and the registered Holm size. "
              "An observed difference well below the detectable one is not evidence of no effect. " + t_quantile_note(d) + f" {SCOPE_SHORT} " + provenance_line(d), ""]
    prompted10 = "E3's 10-seed prompted arm against each 30-seed E3 baseline (descriptive, unadjusted; results review R6)"
    groups = {"E3 against random (ILI, SD units)": [], "Picks on fresh seeds (validation MSE)": [], "E4, AB-C, AB-X, AB-P, AB-L": [], prompted10: [],
              "Trends (rank correlation)": [], "E5 (interval half width)": [], "E6 planning (validation MSE)": []}
    for r in d["rows"]:
        name = r["comparison"]
        key = (prompted10 if name.startswith("E3 prompted Qwen (10 seeds") else
               "E5 (interval half width)" if name.startswith("E5") else "E6 planning (validation MSE)" if name.startswith("E6") else
               "Trends (rank correlation)" if "trend" in name else "Picks on fresh seeds (validation MSE)" if " picks: " in name else
               "E4, AB-C, AB-X, AB-P, AB-L" if name.startswith(("E4", "AB-")) else "E3 against random (ILI, SD units)")
        groups[key].append(r)
    for title, rows in groups.items():
        if not rows:
            continue
        lines += [f"{title}:", ""]
        body = []
        for r in rows:
            digits = 4 if "MSE" in r["unit"] else 2
            shown = f"{r['detectable_80']:.{digits}f}" if r.get("detectable_80") is not None else f"half width {r['half_width']:.3f}"
            body.append([r["comparison"], " / ".join(str(x) for x in r["n"]), f"{r['diff']:+.{digits}f}" if r.get("diff") is not None else "", shown,
                         f"{r['detectable_80_t']:.{digits}f}" if r.get("detectable_80_t") is not None else "", str(r["family"]),
                         p3(r["p"]) if r.get("p") is not None else ""])
        lines += table(["comparison", "n", "observed difference", "detectable at 80% (normal, registered)", "with t quantiles (descriptive)", "Holm size", "p"], body)
    return lines


def t_quantile_note(d: dict) -> str:
    """Review B18: how much larger the detectable effects are with t quantiles at the Welch df (descriptive)."""
    ratios = [r["t_ratio"] for r in d["rows"] if r.get("t_ratio")]
    e4 = [r["t_ratio"] for r in d["rows"] if r.get("t_ratio") and r["comparison"].startswith(("E4 family 1", "E4 family 2"))]
    if not ratios:
        return ""
    span = lambda xs: f"{min(xs) - 1:.0%} to {max(xs) - 1:.0%}"
    return ("The detectable column uses the normal approximation that AB3 registered; it is optimistic for small arms. With Student t quantiles at the Welch "
            f"degrees of freedom (next column, descriptive) the detectable differences are {span(ratios)} larger" + (f", and {span(e4)} larger for E4's 10-policy rows" if e4 else "")
            + ". The trends' detectable rho is a Fisher-z approximation, optimistic with tied levels.")


def robustness() -> list[str]:
    d = load("ST-robust")
    lines = ["## ST: robust checks (descriptive)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += [f"Interquartile mean of regret and probability of improvement over random search, 95% bootstrap intervals over search seeds within arms. {SCOPE_SHORT} " + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        share = t["summary"]["random"]["heldout_share"] if "heldout_share" in t["summary"]["random"] else {}
        rows = [[m, f"{p['iqm_sd']['50']:.2f} [{p['iqm_low']['50']:.2f}, {p['iqm_high']['50']:.2f}]",
                 f"{p['pi']['25']:.2f} [{p['pi_low']['25']:.2f}, {p['pi_high']['25']:.2f}]", f"{p['pi']['50']:.2f} [{p['pi_low']['50']:.2f}, {p['pi_high']['50']:.2f}]",
                 f"{t['summary'][m]['50_raw_mse']['mean']:.4f}"] for m, p in sorted(t["profiles"].items(), key=lambda kv: kv[1]["iqm_sd"]["50"])]
        note = f" (held-out share of incumbents, random search: {', '.join(f'@{b} {v:.0%}' for b, v in share.items())})" if share and dataset == "ETTh1" else ""
        lines += [f"{dataset}{f' ({NO_CLAIM})' if dataset == 'ETTh1' else ''}{note}:", ""]
        lines += table(["method", "IQM regret @50 (SD)", "P(improvement) @25", "P(improvement) @50", "mean regret @50 (raw MSE)"], rows)
    mad = d["tables"]["ETTh1_T336"]["mad"]
    pooled = d["tables"]["ETTh1_T336"]["pooled_sd"]
    lines += [f"ETTh1 ({NO_CLAIM}) with the robust seed SD: {mad['sd']:.5f} against the pooled {pooled:.5f} (ratio {mad['ratio_to_pooled']:.2f}). Replayed with the robust SD as noise and unit, AUC: "
              + ", ".join(f"{m} {v['mean']:.2f}" for m, v in sorted(mad["auc_sd"].items(), key=lambda kv: kv[1]["mean"])) + ".", ""]
    return lines


def rq3() -> list[str]:
    d = load("RQ3")
    lines = ["## RQ3: which design choices matter (descriptive)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += [f"{SCOPE_SHORT} " + provenance_line(d), ""]
    fields = sorted(next(iter(d["tables"].values()))["main_effect_share"])
    ili, ett = d["tables"]["ILI_T24"]["main_effect_share"], d["tables"]["ETTh1_T336"]["main_effect_share"]
    lv = d["cross_table"]["level_means"]
    c2 = {d2: d["tables"][d2]["conditional_share"] for d2 in ("ILI_T24", "ETTh1_T336")}
    lines += ["Marginal shares are registered; conditional shares (what a field adds to a model that has the other seven) are exploratory.", ""]
    lines += table(["field", "ILI marginal / conditional", f"ETTh1 marginal / conditional ({NO_CLAIM})", "best level on ILI", f"best level on ETTh1 ({NO_CLAIM})", "same"],
                   [[f, f"{ili[f]:.3f} / {c2['ILI_T24'][f]:.3f}", f"{ett[f]:.3f} / {c2['ETTh1_T336'][f]:.3f}", lv[f]["best_level"]["ILI"], lv[f]["best_level"]["ETTh1"], "yes" if lv[f]["same_best_level"] else "no"]
                    for f in sorted(fields, key=lambda f: -(ili[f] + ett[f]))])
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        top = sorted(t["interaction_increment"].items(), key=lambda kv: -kv[1])[:5]
        noise = t.get("seed_noise_share_of_config_means")
        by_seed = t.get("interaction_increment_by_seed")
        extra = ""
        if noise is not None:
            extra += f" Seed noise (after removing each seed's common shift) is {noise:.0%} of the variance of the config means, so 1 - that is a ceiling for any model of them; the additive model explains {t['additive_r2']:.0%} and all pairs {t['all_pairs_r2']:.0%} in-sample, "
            h = t["held_seed_r2"]
            extra += f"but on a held-out seed the additive model explains {h['additive']['mean']:.0%} and all pairs {h['all_pairs']['mean']:.0%}, so part of the pair increments fits seed noise."
        if by_seed:
            ranges = [(k, min(v[k] for v in by_seed.values()), max(v[k] for v in by_seed.values())) for k in [k for k, _ in sorted(t["interaction_increment"].items(), key=lambda kv: -kv[1])[:5]]]
            extra += " The five largest increments per single seed (min to max): " + ", ".join(f"{k} {lo:.3f} to {hi:.3f}" for k, lo, hi in ranges) + "."
        lines += [f"{dataset}{f' ({NO_CLAIM})' if dataset == 'ETTh1' else ''}: the additive model explains {t['additive_r2']:.3f} of the variance of the config means and all pairwise interactions "
                  f"{t['all_pairs_r2']:.3f}. Largest pair increments: " + ", ".join(f"{k} {v:.3f}" for k, v in top) + "." + extra, ""]
    c = d["cross_table"]
    lines += [f"Agreement of the field shares across the two tables: Spearman {c['share_spearman']:.2f}, Kendall {c['share_kendall']:.2f} (marginal shares, 8 fields); "
              f"Spearman {c['conditional_share_spearman']:.2f} for the conditional shares (the share a field adds to a model that has the others). Marginal shares and best levels come from an unbalanced set, "
              "so a best level can flip when the other fields are held fixed.", ""]
    r = d["revin"]
    lines += [f"RevIN on ETTh1 ({NO_CLAIM}), validation only, {r['pairs']} twin pairs: the RevIN-off advantage is {ci(r['epochs_30']['off_advantage'], 4)} at 30 epochs and "
              f"{ci(r['epochs_100']['off_advantage'], 4)} at 100 epochs (ratio {r['shrinkage_30_to_100']['ratio_of_means']:.2f}; change {ci(r['shrinkage_30_to_100']['mean_change'], 4)}); "
              f"off is better in {r['epochs_30']['share_pairs_off_better']:.0%} and {r['epochs_100']['share_pairs_off_better']:.0%} of pairs. Share of RevIN-on runs whose best epoch is the last one: "
              f"{r['epochs_30']['on_configs_best_epoch_at_cap']:.0%} at 30 epochs, {r['epochs_100']['on_configs_best_epoch_at_cap']:.0%} at 100. {r['caveat']}. By validation month at 100 epochs: "
              + ", ".join(f"{x:.4f}" for x in r["epochs_100"]["off_advantage_by_month"]) + " (prediction file, 30 epochs, all 1,184 pairs: "
              + ", ".join(f"{x:.4f}" for x in r["prediction"]["off_advantage_by_month"]) + "). The test part stays with Diya's single evaluation.", ""]
    return lines


def baseline_reading(d: dict) -> dict:
    """Results re-review N4: per baseline, the arms whose unadjusted interval lies below it, Holm survivors, and the Holm losses by arm kind."""
    base = d.get("baseline_order") or list(d["baselines"])
    rows = d["rows"]
    grpo = lambda arm: "grpo" in arm.lower()
    below = {b: sorted(r["arm"] for r in rows if r["beats_interval"] and r["baseline"] == b) for b in base}
    held = {b: sorted(r["arm"] for r in rows if r["beats_holm"] and r["baseline"] == b) for b in base}
    worse = [r for r in rows if r["worse_holm"]]
    return {"base": base, "never": [b for b in base if not below[b]], "below": {b: a for b, a in below.items() if a}, "held": {b: a for b, a in held.items() if a},
            "worse": len(worse), "worse_grpo": sum(grpo(r["arm"]) for r in worse), "worse_prompted": sum(not grpo(r["arm"]) for r in worse), "n": len(rows)}


def baseline_sentences(d: dict) -> str:
    x = baseline_reading(d)
    unadjusted_only = {b: a for b, a in x["below"].items() if b not in x["held"]}
    text = f"No arm's unadjusted interval lies below {', '.join(x['never'])}. " if x["never"] else ""
    if unadjusted_only:
        text += ("On unadjusted intervals only: " + "; ".join(f"{', '.join(a)} below {b}" for b, a in unadjusted_only.items())
                 + "; none of these survives Holm. ")
    if x["held"]:
        text += "After Holm, arms beat: " + "; ".join(f"{b} (by {', '.join(a)})" for b, a in x["held"].items()) + ". "
    return text + (f"{x['worse']} of {x['n']} comparisons are Holm-rejected in the baseline's favour: {x['worse_grpo']} with a GRPO arm and "
                   f"{x['worse_prompted']} with a prompted arm.")


def vs_baselines() -> list[str]:
    """Exploratory, post hoc: every LLM arm against every required baseline (results/ablations/LLM-vs-baselines.json)."""
    d = load("LLM-vs-baselines")
    lines = ["## Exploratory: every LLM arm against the required baselines (post hoc)", ""]
    if not d:
        return lines + [MISSING, ""]
    rows = d["rows"]
    lines += [f"Asked for on 26 Nov, after every LLM result had been seen; nothing here is registered and no claim rests on it. ILI held-out AUC, pooled seed-SD units, "
              f"lower is better; each cell is the arm minus the baseline ({WELCH}; negative: the arm searched better). Holm is applied over all {d['comparisons']} "
              "comparisons as a conservative reading; * marks a Holm rejection. The arms ran on different cards with 10 or 30 seeds, and no seed is paired across cards. "
              f"{TRANSFER_SCOPE} {SCOPE_SHORT} " + provenance_line(d), ""]
    base = d.get("baseline_order") or list(d["baselines"])
    lines += table(["baseline", "ILI held-out AUC (t 95%)"], [[b, ci(d["baselines"][b])] for b in base])
    cell = {(r["arm"], r["baseline"]): r for r in rows}
    lines += table([f"arm (cells: arm minus baseline, {WELCH})", "AUC (t 95%)", *base],
                   [[a, ci(v), *(f"{cell[(a, b)]['diff']:+.2f} [{cell[(a, b)]['low']:+.2f}, {cell[(a, b)]['high']:+.2f}]" + ("*" if cell[(a, b)]["reject"] else "") for b in base)]
                    for a, v in sorted(d["arms"].items(), key=lambda kv: kv[1]["mean"])])
    det = [r for r in (load("ST-power") or {}).get("rows", []) if r["comparison"].startswith("E3 prompted Qwen (10 seeds")]
    lines += [baseline_sentences(d) + f" The arm with the lowest mean AUC is {d['best_arm']}."
              + (f" With 10 seeds, E3's prompted arm could detect differences of {min(r['detectable_80'] for r in det):.2f} to {max(r['detectable_80'] for r in det):.2f} SD "
                 "against the 30-seed E3 baselines at 80% power (common.detectable with the observed SDs, one comparison each; descriptive, ST section), so a "
                 "difference not detected is not shown to be absent." if det else ""), ""]
    return lines


def rtx_glance(g: dict) -> list[str]:
    """The RTX families for the summary: the policy-level step result first, then each test with its scope (review B2 to B5, B11, B12)."""
    fams = g["families"]
    tests = {name: (fam, v) for fam in ("AB-P", "AB-G") for name, v in fams[fam]["tests"].items()}
    power_rows = (load("ST-power") or {}).get("rows", [])
    detect = lambda fam, name: next((r["detectable_80"] for r in power_rows if r["comparison"].startswith(f"{fam} {name} on the RTX")), None)
    decision = lambda v: f"Holm p {pfmt(v['p_holm'])}, {'rejected' if v['reject'] else 'not rejected'}"
    rf = g.get("record_file") or {}
    of_record = rf.get("working_tree") or any(v["exists"] for v in (rf.get("at") or {}).values())
    head = ("* Registered LLM ablations on Soumitra's RTX (AB2; Soumitra's AB4, 15db163), Anirudh's independent computation, "
            + ("complete" if g["final"] and not any(fams[f]["missing"] for f in fams) else "final by AB1's deadline" if g["final"] else "provisional")
            + "; not of record: " + ("LLM-registered.json exists and stands where it differs" if of_record else
                                     "the computation of record named by Diya's AB4 (2ab2996), LLM-registered.json, is not yet computed")
            + ". " + ("A verifier recomputed these values independently and they agree (docs/verdicts/ablations_rtx_reviews.md). " if verified(g) else "Not verified at this pin. ")
            + "Secondary tests, not claims under CLAUDE.md rule 6.")
    out = [head]
    pl = fams["AB-G"].get("exploratory_policy_level")
    steps = tests.get("steps trend")
    if pl and steps:
        u, dose = pl["trained_vs_untrained"], pl["dose"]
        trained = [f"{k} steps {v['mean']:.2f}" for k, v in by_level(steps[1]) if float(k) > 0]
        lead = ("no dose-response is detected" if dose["p_sign_flip"] > 0.05 else "a per-policy dose correlation is detected") + \
            f" across 30, 60 and 120 steps ({', '.join(trained)} SD; per-policy sign-flip p {dose['p_sign_flip']:.3f}, analysis (ii))."
        gap = (f" Policies trained on ETTh1 are {u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}] {'worse' if u['diff'] > 0 else 'better'} on ILI than the untrained model "
               f"({WELCH}; permutation p {pfmt(u['p'])}, analysis (i); (i) and (ii) are registered secondary in Diya's AB4, 2ab2996, and exploratory in Soumitra's, 15db163).")
        v = steps[1]
        trend = (f" The registered step trend (rho {v['rho']:+.2f}, {decision(v)}) is reported as registered, with a p that is anti-conservative because the levels are nested "
                 "(the trained levels share seeds and are prefixes of each other up to GPU nondeterminism).")
        out.append(f"  * Steps (AB-G): {lead}{gap}{trend}")
    elif steps:
        v = steps[1]
        out.append(f"  * Steps (AB-G): ILI held-out AUC {trend_words('steps trend', v)} SD (rho {v['rho']:+.2f}, {decision(v)}; p anti-conservative, nested levels).")
    if "scale trend" in tests:
        fam, v = tests["scale trend"]
        det = detect(fam, "scale trend")
        e = {n: etth1_auc(g, k) for k, n in (("0.5B", "0.5B"), ("ref", "1.5B"), ("3B", "3B")) if etth1_auc(g, k)}
        order = ", ".join(f"{k} {x['mean']:.2f}" for k, x in sorted(e.items(), key=lambda kv: kv[1]["mean"]))
        way = "a larger model gave a higher, that is worse," if v["rho"] > 0 else "a larger model gave a lower, that is better,"
        out.append(f"  * Scale (AB-P): on ILI, for these three Qwen2.5 checkpoints and this prompt, {way} AUC ({trend_words('scale trend', v)} SD; rho {v['rho']:+.2f}, {decision(v)}"
                   + (f", could detect |rho| of about {det:.2f}" if det else "") + "); size is not isolated."
                   + (f" On ETTh1 ({NO_CLAIM}) the order by AUC is {order}." if order else ""))
    if "anon vs ref_local" in tests:
        fam, v = tests["anon vs ref_local"]
        ph = programme_row("AB-P", "anon vs ref_local")
        ea, er = etth1_auc(g, "anon"), etth1_auc(g, "ref")
        out.append(f"  * Anonymised prompt (AB-P): on ILI, for this prompt, {v['diff']:+.2f} SD [{v['low']:+.2f}, {v['high']:+.2f}] against ref ({WELCH}; {decision(v)} within AB-P"
                   + (f"; over all {ph['size']} registered ablation tests, exploratory and post hoc, Holm p {pfmt(ph['programme_p_holm'])}, {'rejected' if ph['programme_reject'] else 'not rejected'}" if ph else "")
                   + "). The arm renames the fields and drops their descriptions, which also changes the decoding trie, so it does not isolate prior knowledge of the names."
                   + (f" On ETTh1 ({NO_CLAIM}) anon {ea['mean']:.2f} against ref {er['mean']:.2f}." if ea and er else ""))
    if "repair vs ref_local" in tests:
        fam, v = tests["repair vs ref_local"]
        rep = (g.get("diagnostics_ILI") or {}).get("prompted repair")
        out.append(f"  * Repair (AB-P): {v['diff']:+.2f} SD [{v['low']:+.2f}, {v['high']:+.2f}] against ref ({WELCH}; {decision(v)})"
                   + (f"; it decodes without the constraint, and {rep['repaired_rate']:.0%} of its proposals were over the parameter cap and replaced by the nearest "
                      "feasible unseen config, so it measures that combination." if rep else "."))
    if "KL trend" in tests:
        fam, v = tests["KL trend"]
        det = detect(fam, "KL trend")
        out.append(f"  * KL weight (AB-G): ILI held-out AUC {trend_words('KL trend', v)} SD (rho {v['rho']:+.2f}, {decision(v)}" + (f", could detect |rho| of about {det:.2f}" if det else "") + ").")
    missing = [name for fam in ("AB-P", "AB-G") for name in fams[fam]["missing"]]
    if missing:
        out.append(f"  * Missing, counted as not rejected: {', '.join(missing)}.")
    return out


def compute_glance() -> list[str]:
    d = load("compute")
    if not d:
        return []
    parts = [f"{t['card']} {t['kind']} {t['hours']:.1f} h" for t in d["totals"]]
    return ["* Compute (descriptive): recorded hours per card, summed from the run and training records: " + "; ".join(parts)
            + ". Equal target epochs are not equal compute: GRPO also trained on the source table, and the CPU replays record no time (Compute section)."]


def glance() -> list[str]:
    """The main readings, each built from the result files."""
    lines = ["## At a glance", ""]
    q, x, c, n, r, a, g, w = (load(k) for k in ("AB-Q", "AB-X", "AB-C", "AB-L-null", "AB-R", "RQ3", "LLM", "ST-power"))
    lines.append(f"* Scope of everything below: {SCOPE_SHORT} Every test here is secondary to the primary endpoints and none is a claim under CLAUDE.md rule 6 "
                 "(Welch t intervals and permutation p, no paired block-bootstrap interval, no gate). ETTh1 values are in-sample (one seed per config), so no claim rests on them.")
    if q:
        s, h = q["separation"], q["holdout"]["methods"]
        noise_free_gap = s["tpe"]["noise_free_diff_vs_random"]
        clear = [m for m in h if m != "random" and h[m]["diff_vs_random"]["high"] < 0]
        named = ", ".join(f"{m} {h[m]['diff_vs_random']['mean']:+.2f} [{h[m]['diff_vs_random']['low']:+.2f}, {h[m]['diff_vs_random']['high']:+.2f}]" for m in clear) or "none"
        e3 = s["tpe"]["e3_diff_vs_random"]
        per = h["tpe"]["diff_by_held_seed"]
        lines.append(f"* Noise-free lookups: searching and scoring on the same 3-seed means, TPE beats random by {noise_free_gap:+.2f} SD in AUC ({s['tpe']['noise_free_prob_better_than_random']:.0%} of seed pairs); "
                     f"with E3's noisy lookups its held-out gap is {e3:+.2f} SD ({s['tpe']['e3_prob_better_than_random']:.0%}). So removing noise does not widen TPE's margin. "
                     f"With lookups from two seeds scored on the third, TPE's difference per held seed is {', '.join(f'{v:+.2f}' for v in per.values())} SD (interval over search seeds only; by budget 100 its 30 runs hold few distinct configs; the in-sample gap of the same hold-out runs is {h['tpe']['in_sample_diff_vs_random']['mean']:+.2f} SD). "
                     "AB-Q neither supports nor refutes that seed noise explains the null on fresh seeds. (AB-Q; the hold-out is exploratory)")
    if w:
        nulls = [v for v in w["rows"] if v["comparison"].startswith(("E4 family 1", "E4 family 2")) and "REJECTED" not in v.get("note", "")]
        rejected = [v for v in w["rows"] if v["comparison"].startswith(("E4 family 1", "E4 family 2")) and "REJECTED" in v.get("note", "")]
        picks = [v for v in w["rows"] if " picks: " in v["comparison"] and "@50" in v["comparison"] and v["p"] > 0.05]
        sd = [v["detectable_80_sd"] for v in picks]
        e4 = [v["detectable_80"] for v in nulls]
        e4_t = [v["detectable_80_t"] for v in nulls if v.get("detectable_80_t")]
        ili = [v["detectable_80_sd"] for v in picks if v["comparison"].startswith("ILI")]
        ett = [v["detectable_80_robust_sd"] for v in picks if v["comparison"].startswith("ETTh1")]
        ratio = sorted(abs(v["diff"]) / v["detectable_80"] for v in picks)
        worse = f" The one E4 comparison that rejects, {rejected[0]['comparison'].split(': ')[1]}, is a loss, not a null: the league is {rejected[0]['diff']:+.2f} SD worse." if rejected else ""
        lines.append(f"* The nulls have limited resolution. At budget 50 the pick reseeds could detect differences of {min(ili):.2f} to {max(ili):.2f} SD on ILI and, in robust-SD units, {min(ett):.2f} to {max(ett):.2f} on ETTh1 (80% power, Holm size in the table); the "
                     f"observed non-significant differences are {ratio[0]:.0%} to {ratio[-1]:.0%} of that. E4's ILI nulls (league against the portfolio and against no league, 10 policies) could detect {min(e4):.2f} to {max(e4):.2f} SD "
                     "by the registered normal approximation" + (f", and {min(e4_t):.2f} to {max(e4_t):.2f} SD with t quantiles (descriptive)" if e4_t else "") + f".{worse} (ST)")
    if x:
        t = x["tables"]["ILI_T24"]["tests"]
        wins = [k for k, v in t.items() if v["reject"]]
        best = min(t, key=lambda k: t[k]["diff"])
        lines.append(f"* External HPO baselines: none of the {len(t)} is rejected against random search on ILI after Holm; the closest is {best} ({t[best]['diff']:+.2f} SD, Holm p {t[best]['p_holm']:.2f}), and effects below about "
                     f"{t[best]['detectable_80']:.2f} SD are not excluded. DEHB is handicapped: {x['tables']['ILI_T24']['repair_rate']['dehb']:.0%} of its proposals were infeasible or repeated and were replaced (AB-X).")
    if c:
        t = c["tables"]["ILI_T24"]["tests"]
        lines.append(f"* Controller ablations: none of the {len(t)} league settings is rejected against the reference; the test could detect about "
                     f"{max(v['detectable_80'] for v in t.values()):.2f} SD (AB-C).")
    rtx = load("LLM-rtx") or {}
    if "families" in rtx.get("registered", {}):
        lines += rtx_glance(rtx["registered"])
    vb = load("LLM-vs-baselines")
    if vb:
        x = baseline_reading(vb)
        winners = {a for arms in x["held"].values() for a in arms}
        only_prompted = bool(winners) and all("prompted" in a for a in winners)
        lines.append(f"* Exploratory, post hoc (Holm over {vb['comparisons']} comparisons): no LLM arm (prompted or GRPO, any card) beats {', '.join(x['never'])} on ILI held-out AUC, "
                     "even by an unadjusted interval"
                     + ("; on unadjusted intervals only, " + "; ".join(f"{', '.join(a)} lie below {b}" for b, a in x["below"].items() if b not in x["held"])
                        + ", and none of these survives Holm" if any(b not in x["held"] for b in x["below"]) else "")
                     + (f"; after Holm the only baselines an arm beats are {', '.join(x['held'])}" if x["held"] else "; after Holm no arm beats any baseline")
                     + (", each time by an untrained prompted arm, which searches ILI directly and is not the arm that transfer control is matched to" if only_prompted else "")
                     + f"; {x['worse']} comparisons are Holm-rejected in the baseline's favour ({x['worse_grpo']} with a GRPO arm, {x['worse_prompted']} with a prompted arm). "
                     f"The GRPO-trained arms are worse than the untrained prompted model (AB-G). {TRANSFER_SCOPE}")
    if g and g["mps"]["AB-P"]["tests"]:
        t = g["mps"]["AB-P"]["tests"]
        etth1_d = g["mps"].get("ETTh1_differences", {})
        opposite = f"; on ETTh1 ({NO_CLAIM}) the signs reverse: " + ", ".join(f"{k} {v['diff']:+.2f}" for k, v in etth1_d.items()) if etth1_d else ""
        lines.append("* Prompted Qwen on the Mac (exploratory; unadjusted 95% intervals, none is rejected after Holm): " + "; ".join(f"{k} {v['diff']:+.2f} SD [{v['low']:+.2f}, {v['high']:+.2f}], Holm p {v['p_holm']:.3f}" for k, v in t.items()) + opposite + " (AB-P).")
    if g and g["mps"]["AB-G"]["tests"]:
        t = g["mps"]["AB-G"]["tests"]["step trend"]
        u, dose, rv = g["mps"]["policy_level"]["trained_vs_untrained"], g["mps"]["policy_level"]["dose"], g["mps"]["revin"]
        mac = rv["mac_share_off_ILI"]
        lines.append(f"* GRPO training on the Mac (exploratory): ten league policies trained on ETTh1 are {u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}] worse on ILI than the untrained model (post hoc; all 10 worse than the same-seed untrained search). "
                     f"Trained policies look up RevIN-off configs {mac['120 (Mac)']:.0%} of the time on ILI (untrained {mac['0 (untrained, Mac)']:.0%}; the best RevIN-off ILI config has regret {rv['best_regret_by_revin']['ILI_T24']['revin_off']:.2f} SD), "
                     "which is consistent with a source-table preference that does not transfer; it is not evidence that GRPO or the league is harmful, and the cause is not tested. "
                     f"No dose effect is detected among 30, 60 and 120 steps (sign flip p {dose['p_sign_flip']:.2f}). The registered step-trend statistic is rho {t['rho']:+.2f} on these runs, but its p is not valid (dependent checkpoints), so it is not a rejection (AB-G).")
    if n:
        e, i = n["tables"]["ETTh1"]["E3_A10"], n["tables"]["ILI"]["E3_A10"]
        o = n["observed_adjusted_slopes"]
        none = load("AB-L")["by_source"]["ETTh1"]["arms"]["none"]["mean_slope"]
        mac = n["tables"]["ILI"].get("ABL_P_ref_mps (Mac, exploratory)")
        mac_e = n["tables"]["ETTh1"].get("ABL_P_ref_mps (Mac, exploratory)")
        holm = load("AB-L")["holm"]
        reg_e, reg_i = holm["league slopes vs 0, ETTh1 source"], holm["league slopes vs 0, ILI source"]
        lines.append(f"* The registered AB-L rejection (ETTh1 source, Holm p {reg_e['p_holm']:.3f}) rests on a test that is too liberal (false-positive rate {e['raw']['false_positive_rate']['0.025']:.2f} and "
                     f"{i['raw']['false_positive_rate']['0.025']:.2f} at the 0.025 level for a model that never learns, from a null of ten searches per table"
                     + (f"; from thirty Mac searches (exploratory) {mac_e['raw']['false_positive_rate']['0.025']:.2f} on ETTh1 and {mac['raw']['false_positive_rate']['0.025']:.2f} on ILI" if mac else "")
                     + f"); an exploratory, post hoc position-adjusted test agrees: exact sign flip of the {o['ETTh1']['n']} adjusted ETTh1 slopes p {o['ETTh1']['p_sign_flip']:.3f} (uncorrected; "
                     f"Holm over the two tables {min(1.0, 2 * o['ETTh1']['p_sign_flip']):.3f}). On the ILI source no trend was detected (registered Holm p {reg_i['p_holm']:.3f}; adjusted p {o['ILI']['p_sign_flip']:.2f}). "
                     f"The ETTh1 reward is in-sample (one seed per config), so no claim rests on it. The no-league arm's ETTh1 slope is {none:+.3f}, at least as large, so nothing here says the league did it (AB-L).")
    if r:
        t = r["tables"]["ILI_T24"]
        pick = lambda f, k, key="tau": next(v[key] if key == "tau" else v["tau_independent_seeds"]["tau"] for v in t["variants"] if v["features"] == f and v["size"] == 300 and v["model"] == "gbt" and v["k"] == k)
        lines.append(f"* The referee's skill on ILI comes mostly from the config fields: tau {pick('config', None):.2f} from fields alone, {pick('curves', 1):.2f} from curves alone, {pick('both', 1):.2f} from both, against the 3-seed mean. "
                     f"That target contains the seed-2021 value the referee is trained on; against the mean of the other two seeds the same three are {pick('config', None, 'ind'):.2f}, {pick('curves', 1, 'ind'):.2f} and {pick('both', 1, 'ind'):.2f} (AB-R).")
    if a:
        lines.append(f"* The two tables show no detectable agreement on which fields matter (Spearman {a['cross_table']['share_spearman']:.2f} between marginal field shares, {a['cross_table']['conditional_share_spearman']:.2f} between exploratory conditional shares, 8 fields); RevIN explains {a['tables']['ETTh1_T336']['main_effect_share']['instance_norm']:.0%} "
                     f"of ETTh1's variance and {a['tables']['ILI_T24']['main_effect_share']['instance_norm']:.0%} of ILI's, and its ETTh1 validation advantage for RevIN off does not shrink from 30 to 100 epochs (that comparison also changes patience from 10 to 100, RQ3; ETTh1 {NO_CLAIM}).")
    lines += compute_glance()
    return lines + [""]


def rerun_note() -> str:
    """Review B9: the first AB4 computation and this one, both kept; what changed, from the two result files."""
    first, now = load("LLM-first-AB4-run"), load("LLM-rtx")
    if not first or not now:
        return ""
    f0, f1 = first["registered"]["families"], now["registered"]["families"]
    changed, same, appeared = [], [], []
    for fam in ("AB-P", "AB-G"):
        for name, t in f1[fam]["tests"].items():
            o = f0[fam]["tests"].get(name)
            if o is None:
                appeared.append(f"{name} (counted missing then: {f0[fam].get('missing_reasons', {}).get(name, 'missing')}; "
                                f"{'rejected' if t['reject'] else 'not rejected'} now)")
            elif (o["p"], o["p_holm"], o["reject"]) != (t["p"], t["p_holm"], t["reject"]):
                changed.append(f"{name}: p {pfmt(o['p'])} then and {pfmt(t['p'])} now, Holm p {pfmt(o['p_holm'])} then and {pfmt(t['p_holm'])} now, "
                               f"{'rejected' if o['reject'] else 'not rejected'} then and {'rejected' if t['reject'] else 'not rejected'} now")
            else:
                same.append(name)
    flips = [name for fam in ("AB-P", "AB-G") for name, t in f1[fam]["tests"].items()
             if name in f0[fam]["tests"] and f0[fam]["tests"][name]["reject"] != t["reject"]]
    git = first.get("provenance", {}).get("git_hash", "")[:7]
    return ("* Rerun of the RTX computation (Diya's AB4 asks that a post-run fix keep both outputs and that the report disclose it): the first computation after the AB4 pin "
            f"(results/ablations/LLM-first-AB4-run.json, scripts/ablations/llm_tests.py at {git}) read the transfer items' training logs only from results/runs/ and so failed the "
            "two step arms' checks. f9c0d62 made item_rows read results/ablations/rtx/ as well; the bug was found after the AB4 prompted values were seen. Both outputs are kept. "
            + ("Newly computed: " + "; ".join(appeared) + ". " if appeared else "") + ("Changed: " + "; ".join(changed) + ". " if changed else "")
            + (f"Unchanged: {', '.join(same)}. " if same else "")
            + ("No decision of a test computed both times changed." if not flips else f"Decisions changed: {', '.join(flips)}."))


def ab5_correction() -> str:
    """Review B15: a dated correction of AB5's 'What existed' paragraph, from the queue lines and commit times; AB5 itself is not edited."""
    g = (load("LLM-rtx") or {}).get("registered", {})
    ab5 = commit_time(g, "510633d") if g else None
    if not ab5:
        return ""
    finished = [f"{q['name']} ({et(q['time'])})" for q in (g.get("timeline") or {}).get("queue", [])
                if q["event"] == "DONE" and q["file"].endswith("ab4_queue_status.txt") and when(q["time"]) < when(ab5)]
    if not finished:
        return ""
    return ("* Correction, dated 26 Nov and written after the results were seen (AB5 itself is not edited): AB5's 'What existed' paragraph (510633d, committed "
            f"{et(ab5)}) says Soumitra's AB4 was registered and not yet run. By results/logs/ab4_queue_status.txt at the pinned commit, these AB4 jobs had finished by then: "
            f"{', '.join(finished)}. This clone's copy of origin/soumitra-leg2 did not yet hold them when AB5 was written (by its reflog).")


def failure_case() -> list[str]:
    """Results review R2, exploratory: the RevIN-off share of ILI lookups per arm beside its AUC, from LLM.json and LLM-rtx.json; no causal claim."""
    q, rtx = load("LLM"), (load("LLM-rtx") or {}).get("registered", {})
    lines = ["## Exploratory: the failure case, RevIN-off lookups on ILI (post hoc)", ""]
    rv = (q or {}).get("mps", {}).get("revin")
    if not rv:
        return lines + [MISSING, ""]
    b = rv["best_regret_by_revin"]
    lines += ["Asked for by the results review (R2) after every result was seen; descriptive, no test. RevIN is the instance_norm field (reversible instance normalisation). "
              f"On ILI the best RevIN-off config has regret {b['ILI_T24']['revin_off']:.2f} SD and the best RevIN-on config {b['ILI_T24']['revin_on']:.2f} SD "
              f"(llm_tests.revin_regret on each table's config values: means of three seeds on ILI); on ETTh1 ({NO_CLAIM}) the best RevIN-on config has {b['ETTh1_T336']['revin_on']:.2f} SD and the best "
              f"RevIN-off {b['ETTh1_T336']['revin_off']:.2f}, values from configs with one seed each (ETTh1 has one seed per config). A search that looks up only RevIN-off configs on ILI "
              f"can reach nothing closer to ILI's best than {b['ILI_T24']['revin_off']:.2f} SD by the table's means. {SCOPE_SHORT}", ""]
    entries = []  # (arm, card, searches, AUC or None, share, trained by GRPO)
    compute_rows = (load("compute") or {}).get("rows", [])
    def cards_of(method: str) -> str:
        row = next((r for r in compute_rows if r["table"] == "ILI_T24" and r["kind"] == "search" and r["arm"].endswith(f": {method}")), None)
        if not row:
            return "-"
        return ", ".join(c for c in row["cards"]) if any(c != "not recorded" for c in row["cards"]) else "none recorded (CPU replay)"
    for k, v in rv["committed_ILI"].items():
        entries.append((k, cards_of(k.split(" (")[0]), v["runs"], v["auc"], v["share_off"], k.startswith("grpo")))
    summary = (rtx.get("descriptive") or {}).get("summary_ILI") or {}
    for k, share in ((rtx.get("descriptive") or {}).get("revin_off_share_ILI") or {}).items():
        s_ = summary.get(k)
        entries.append((f"{arm_label(k)}, RTX", "RTX 4080 Laptop", s_["auc"]["n"] if s_ else None, s_["auc"] if s_ else None, share, k.startswith("transfer")))
    mac = q["mps"].get("AB-G", {}).get("tests", {}).get("step trend", {}).get("levels", {})
    for k, share in rv.get("mac_share_off_ILI", {}).items():
        level = mac.get(k.split()[0])
        entries.append((f"level {k}", "Apple M4 Pro", level["n"] if level else None, level, share, k.split()[0] != "0"))
    lines += table(["arm", "card", "searches", "ILI AUC (t 95%)", "RevIN-off share of ILI lookups"],
                   [[a, c, str(n) if n else "-", ci(auc) if auc else "-", f"{share:.0%}"] for a, c, n, auc, share, _ in entries])
    grpo = [share for *_, share, trained in entries if trained]
    untrained = [share for *_, share, trained in entries if not trained]
    if grpo and untrained:
        lines += [f"The GRPO arms' RevIN-off shares of ILI lookups range from {min(grpo):.0%} to {max(grpo):.0%}; random search and the untrained prompted arms range from "
                  f"{min(untrained):.0%} to {max(untrained):.0%}, so training changed what the policies look up. "
                  f"The preference they show fits the ETTh1 table, whose best config has RevIN {'off' if b['ETTh1_T336']['revin_off'] == 0 else 'on'}, and costs on ILI. "
                  "This is consistent with a source-table preference that does not transfer. No intervention tested the cause (no arm removes the preference while keeping "
                  "the training), so no causal claim is made. Arms on different cards are shown side by side, not paired.", ""]
    return lines


def figure_section() -> list[str]:
    """Results review R16, exploratory: the anytime-regret figure and its table view (results/ablations/anytime-ILI.json)."""
    d = load("anytime-ILI")
    lines = ["## Exploratory: anytime regret on ILI (descriptive, post hoc)", ""]
    if not d:
        return lines + [MISSING, ""]
    rel = Path(d["figure"]).relative_to("docs")
    ref = d.get("reference")
    lines += [f"![Mean ILI held-out regret by budget, two panels]({rel})", "",
              "Mean held-out regret at each budget from 1 to 100 full-run equivalents, over each arm's searches (common.by_seed at that budget; the mean over the 100 "
              "budgets is the AUC of the tables above), with a 95% t band per budget over searches; arms with 10 searches have wider bands than arms with 30. "
              + (f"The dashed grey line in both panels is {ref['name']}. " if ref else "") + "Each series keeps one colour in both panels. "
              + ("Not drawn: " + ", ".join(d.get("not_drawn", [])) + ". " if d.get("not_drawn") else "")
              + f"Added after every result was seen. {TRANSFER_SCOPE} {SCOPE_SHORT} " + provenance_line(d), ""]
    marks = (1, 10, 25, 50, 100)
    every = ([ref] if ref else []) + [s for p in d["panels"] for s in p["series"]]
    rows = [[s["name"], str(s["n"]), *(f"{s['mean'][b - 1]:.2f}" for b in marks), f"{sum(s['mean']) / len(s['mean']):.2f}"] for s in every]
    return lines + ["Table view of the figure (means; the bands are in results/ablations/anytime-ILI.json):", ""] + \
        table(["arm (searches)", "n", *(f"@{b}" for b in marks), "mean over 1 to 100 (AUC)"], rows)


def compute_section() -> list[str]:
    """Area-chair review D (checklist item 8) and results review R13: compute per arm from results/ablations/compute.json (descriptive)."""
    d = load("compute")
    lines = ["## Compute (descriptive)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += ["Sums of what the committed records hold, added after every result was seen. Target epochs are the epochs a search was charged on the target table "
              "(the equal budget of A6); source epochs are those charged while training a policy on the source table; seconds are the wall time each record "
              "holds for one search or one training run, so model loading and anything outside a record are left out (the RTX queue wall clock below includes them). "
              "Equal target epochs are not equal compute: the GRPO arms also trained on the source table, and the E4 portfolio and warm-started TPE read the source table. "
              + provenance_line(d), ""]
    lines += ["Recorded hours per card:", ""] + table(["machine", "card", "kind", "hours"], [[t["machine"], t["card"], t["kind"], f"{t['hours']:.2f}"] for t in d["totals"]])
    timed = [r for r in d["rows"] if r.get("seconds") is not None]
    period = load_period((load("LLM-rtx") or {}).get("registered", {}))
    segments = {j["job"]: [] for jobs in d.get("queue_wall_clock", {}).values() for j in jobs}
    for jobs in d.get("queue_wall_clock", {}).values():
        for j in jobs:
            segments[j["job"]].append(j["end"])

    def rtx_job(r: dict) -> str | None:
        if not r["arm"].startswith("RTX "):
            return None
        item = r["arm"].removeprefix("RTX ").split(" ")[0].lower()
        return item if item.startswith("abl_g_") else f"{item}_{'ILI' if r['table'].startswith('ILI') else 'ETTh1'}"

    late = {r["arm"] + r["table"] for r in timed if rtx_job(r) and ran_after(segments.get(rtx_job(r), []), period)}
    lines += ["Per arm, records with seconds († the RTX job finished after the laptop's reboot; see below):", ""] + table(
        ["arm", "table", "card (records)", "records", "target epochs", "source epochs", "hours"],
        [[r["arm"], r["table"], ", ".join(f"{c} ({n})" for c, n in r["cards"].items()), str(r["records"]),
          f"{r['target_epochs']:,.0f}" if r.get("target_epochs") is not None else "-", f"{r['source_epochs']:,.0f}" if r.get("source_epochs") is not None else "-",
          f"{r['seconds'] / 3600:.2f}" + (" †" if r["arm"] + r["table"] in late else "")] for r in timed])
    if late and period:
        lines += [f"† The RTX laptop rebooted at {et(period['start'])} (queue line at {et(period['logged'])}); the marked jobs finished after it. {LOAD_TEXT}", ""]
        s30 = next((r for r in timed if r["arm"] == "RTX ABL_G_steps30_local (training)"), None)
        s60 = next((r for r in timed if r["arm"] == "RTX ABL_G_ref_local (training)"), None)
        if s30 and s60:
            lines += [f"For example, the 30-step training took {s30['seconds'] / 3600:.2f} h for {s30['source_epochs']:,.0f} source epochs, against {s60['seconds'] / 3600:.2f} h "
                      f"for {s60['source_epochs']:,.0f} at 60 steps; the 30-step job {'finished after the reboot' if s30['arm'] + s30['table'] in late else 'ran before the reboot'} "
                      f"and the 60-step job {'after' if s60['arm'] + s60['table'] in late else 'before'} it.", ""]
    superseded = [r for r in d["rows"] if r.get("superseded") and r.get("seconds") is not None]
    if superseded:
        lines += [f"The rows marked superseded are E4's first training attempt on 23 Nov ({sum(r['seconds'] for r in superseded) / 3600:.2f} h recorded): A25 superseded it and "
                  "none of its policies was deployed, so it counts as compute only.", ""]
    cpu = [r for r in d["rows"] if r["kind"] == "search" and r.get("seconds") is None]
    errors = [r for r in d["rows"] if r["kind"] == "errors"]
    if cpu:
        per = [r["target_epochs"] / r["records"] for r in cpu if r.get("target_epochs")]
        lines += [f"The {len(cpu)} CPU-replay arms (E3's CPU methods and E4's transfer controls) record target epochs, {min(per):,.0f} to {max(per):,.0f} per search, "
                  "but no seconds and no card." + (" Error records, which hold no seconds: " + ", ".join(f"{r['arm']} {r['records']}" for r in errors) + "." if errors else ""), ""]
    for rel, jobs in d.get("queue_wall_clock", {}).items():
        if jobs:
            stopped = [j for j in jobs if j["outcome"] == "INTERRUPTED"]
            lines += [f"RTX queue wall clock ({Path(rel).name}): {sum(j['seconds'] for j in jobs) / 3600:.2f} h over {len(jobs)} job segments"
                      + ("; interrupted and relaunched: " + ", ".join(f"{j['job']} after {j['seconds'] / 3600:.2f} h" for j in stopped) if stopped else "") + ".", ""]
    bill, spend = d.get("dollars"), d.get("e4_spend")
    text = "Dollars: no results file records them per arm, and the RTX and the Mac are local cards with no metered cost. "
    if bill:
        text += (f"The committed billing record {bill['path']} (at {str(bill['commit'])[:7]}) is a month-to-date summary of Soumitra's Modal profile "
                 f"({bill['comments'][0] if bill['comments'] else 'no note'}): metered ${float(bill['metered_cost']):.2f}, billed ${float(bill['billed_cost']):.2f}. ")
    if spend:
        text += f"Commit {spend['commit'][:7]} ({et(spend['time'])}) states, in its message: \"{spend['line']}\" (docs/STATE.md, 24 Nov, gives the same figure). "
    if bill and spend:
        later = bill.get("read_at") and when(bill["read_at"]) > when(spend["time"])
        text += ("The two figures differ" + (", and the billing summary, read later, is the lower one" if later and float(bill["metered_cost"]) < spend["metered"] else "")
                 + "; Soumitra is asked to reconcile them. Neither is split per arm.")
    lines += [text, ""]
    return lines


def history() -> list[str]:
    return ["## Runs, re-runs and open questions", "",
            "* Registration: AB3 is commit 31f9313 (00:58 on 25 Nov); the first CPU outputs were written after it. AB3 addendum 1 and 2 follow a verifier and a critic "
            "review of the first outputs and are exploratory. The branch was not on GitHub when the runs started, so the order is shown by commit times only.",
            "* AB-X ran three times: once with a provenance stamp marked dirty (the report generator had been added to the tree, not the code that ran), once from a clean commit, and once more after run records "
            "gained a git hash. The first two run files were byte-identical; the third differs only by the added field. The CPU analyses were re-run once after the verifier's fixes; the registered numbers "
            "were produced by scripts unchanged since 31f9313 except where addendum 1 and 2 say otherwise, and their first outputs are not kept.",
            *(x for x in (rerun_note(), ab5_correction()) if x),
            "* Added on 26 Nov after every result was seen, for two reviews (an area chair's and a results reviewer's), and labelled where they appear: the run-path check of the RTX arms "
            "(every arm passes), the timeline, the per-arm ILI table with raw MSE, the per-policy table, the t-quantile detectable column, the one-Holm sensitivity, the failure-case "
            "table, the compute totals and the scope sentences. No registered test, permutation, Holm size or completeness rule changed.",
            "* Not done from WP6: regret curves with intervals as figures (docs/ablations.md gives tables at fixed budgets only)." if not (load("anytime-ILI") or {}).get("figure") else
            "* From WP6: an anytime-regret figure of mean curves with 95% t bands (the figure section); the A3 noise-floor band and per-architecture regret are not done.",
            "* Not answered here: why TPE wins E3's held-out replay but not the pick reseeds (an effect-size question, no equivalence test exists yet); whether rankings survive resampling "
            "of the three table seeds; whether results survive a different fidelity ladder or a common first configuration; BOHB and one-epoch-then-top-K baselines; and, for AB-L, how much of the ETTh1 slope the league contributes.",
            "* Mac LLM runs: the ILI prompted reference (ABL_P_ref_mps) ran at commit 31f9313 and every other Mac item at 0929d26; scripts/ablations/llm_mps.py and leaguetst/ are identical between the two. Every run file's commit and digest is in results/ablations/LLM.json. "
            "AB3 addendum 3 (policy-level and dose analyses) was committed before its outputs; addendum 4 (RevIN shares, paired gap, diagnostics) was written after a critic had reported them and its code was run once before the commit, as it says.",
            "* Verifier and critic reports (AI subagents), quoted in full, are in docs/verdicts/ablations_cpu_reviews.md and docs/verdicts/ablations_llm_reviews.md, and for the RTX "
            "families (AB2 and AB4) in docs/verdicts/ablations_rtx_reviews.md.", ""]


def main() -> int:
    lines = ["# Ablations, robustness and external validity", "",
             "Generated by scripts/ablations/report.py from results/ablations/*.json. Do not edit by hand. Registered in "
             "docs/prereg_ablations.md (AB1, AB3 and AB5 on this branch; AB2 and Soumitra's AB4, 15db163, on origin/soumitra-leg2; Diya's AB4, 2ab2996, on main); every result here is secondary to the primary endpoints, "
             "and none is a claim under CLAUDE.md rule 6. Regret is in pooled seed-SD units unless stated; lower is better. Intervals are 95% t intervals (Welch for differences), "
             f"unadjusted, beside Holm-adjusted p where a family is tested, unless a section names a bootstrap interval (AB-R, the robust checks). ILI regret is scored on seeds the search never saw; ETTh1 has one seed per config, so ETTh1 replays are {NO_CLAIM}. {SCOPE}", ""]
    lines += (glance() + noise_free() + external() + controller() + registered_llm() + programme_section() + vs_baselines() + figure_section() + failure_case() + llm() + learned()
              + one_seed() + referee() + rq3() + robustness() + power() + compute_section() + history())
    TARGET.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
