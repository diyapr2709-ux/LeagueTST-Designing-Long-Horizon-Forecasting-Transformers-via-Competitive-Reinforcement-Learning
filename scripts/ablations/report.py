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
    return table(["test", "seeds", "difference (SD units)", "p", "p, Holm", "rejected", "detectable at 80%"], rows)


def provenance_line(data: dict) -> str:
    p = data["provenance"]
    return f"Source: {p['script']} at {p['git_hash'][:12]} (script sha256 {p['script_sha256'][:12]})."


def controller() -> list[str]:
    d = load("AB-C")
    lines = ["## AB-C: factorized-controller ablations (tested, CPU)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += [f"League rivals, {d['seeds']} search seeds per arm; held-out AUC regret on ILI in pooled seed-SD units; lower is better. "
              + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        lines += [f"{dataset}{'' if t['registered'] else ' (descriptive: one seed per config, in-sample)'}:", ""]
        lines += table(["arm", "AUC", "AUC, raw validation MSE", *(f"@{b}" for b in BUDGETS)],
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
              "tested on ILI against E3's random search. " + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        lines += [f"{dataset}{'' if t['registered'] else ' (descriptive)'}:", ""]
        lines += table(["method", "AUC", "AUC, raw validation MSE", *(f"@{b}" for b in BUDGETS)],
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
              "which the method beats random search: with seed noise (E3, held out) and without it. " + provenance_line(d), ""]
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
              "table would report it, and held out on the full 3-seed table. Optimism is held-out AUC minus reported AUC. " + provenance_line(d), ""]
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
    lines += ["Kendall tau against the multi-seed mean on the 150 scored configs of E2, 95% bootstrap interval. " + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        e = t["e2_referee"]
        ceiling = t["one_seed_ceiling"]
        lines += [f"{dataset}: E2's referee (both feature sets, k = {e['k']}, 300 configs, boosted trees) {e['tau']:.2f} [{e['low']:.2f}, {e['high']:.2f}]. "
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
        lines += [f"{dataset}:", ""] + table(["boosted trees, 300 configs", "tau against the multi-seed mean", "tau against the seed-2021 value", "tau against seeds 2022 and 2023 (exploratory)"], rows)
    return lines


def learned() -> list[str]:
    d, n = load("AB-L"), load("AB-L-null")
    lines = ["## AB-L: did GRPO learn? (tested by Soumitra; calibration here is exploratory and post hoc)", ""]
    if not d:
        return lines + [MISSING, ""]
    for dataset in ("ETTh1", "ILI"):
        t = d["tests"][f"league slopes vs 0, {dataset} source"]
        h = d["holm"][f"league slopes vs 0, {dataset} source"]
        lines.append(f"* {dataset} source: mean Spearman slope {t['mean']:+.3f} over {t['n']} league policies, p {t['p']:.3f}, Holm {h['p_holm']:.3f}, "
                     f"{'rejected' if h['reject'] else 'not rejected'} (results/ablations/AB-L.json).")
    lines.append("")
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


RECORD_NOTE = ("Not the result of record. Diya's amendment AB4 on main (2ab2996, 26 Nov 17:21 ET) names scripts/ablations/llm_registered.py, run once by Soumitra, "
               "as the computation of the registered AB-P and AB-G families (results/ablations/LLM-registered.json); Soumitra's branch also carries its own "
               "scripts/ablations/rtx_tests.py (e6926d8). AB5 reached GitHub on 26 Nov at 18:36 ET, after that amendment. The tables below are an independent "
               "computation of the same registered tests from the same run files; where they differ from LLM-registered.json, that file stands and the difference "
               "is a finding to report. docs/LEG3_soumitra.md asks Anirudh not to read results/ablations/rtx/ before LLM-registered.json is committed: the AB2 values "
               "here were computed before that file reached this clone, and the AB4 values after it, on Anirudh's decision, as a check. This branch does not change "
               "llm_registered.py or its inputs.")


def step_reading(steps: dict | None, u: dict, dose: dict) -> str:
    """Diya's AB4 reading rule for the step result, fixed before any AB4 value was seen."""
    if steps and dose["p_sign_flip"] <= 0.05 and (dose["mean_rho"] > 0) == (steps["rho"] > 0) and dose["mean_rho"] != 0:
        return "By AB4's rule the step result is read as a dose-response: (ii) has the registered trend's sign and p at most 0.05."
    if u["p"] <= 0.05:
        return "By AB4's rule the step result is not read as a dose-response; trained policies differ from the untrained model ((i), p at most 0.05)."
    return "By AB4's rule the step result is read neither as a dose-response nor as a difference between trained and untrained policies."


def registered_llm() -> list[str]:
    """AB2 and AB4: the registered AB-P and AB-G families on Soumitra's RTX items (results/ablations/LLM-rtx.json)."""
    d = load("LLM-rtx")
    lines = ["## AB-P and AB-G on Soumitra's RTX (AB2 and AB4): Anirudh's independent computation", ""]
    if not d or "families" not in d.get("registered", {}):
        return lines + [MISSING, ""]
    g = d["registered"]
    src = g["source"]
    if g["final"]:
        status_text = "Status: final by AB5's rule, for this computation."
    else:
        status_text = ("Status: provisional. Reasons: " + "; ".join(g.get("provisional_reasons") or ["not final"]) + ". "
                       "The pending tests count as not rejected for now (Holm treats them as p = 1). A rejection shown here holds whatever they show; "
                       "only the pending tests, and the decisions of tests not rejected, are still open.")
    rejected_scale = g["families"]["AB-P"]["tests"].get("scale trend")
    lines += [f"{status_text} Deadline for every arm: 28 Nov 23:59 ET (AB1). "
              f"Registered in AB2 (c2dda8d) and AB4 (15db163) on origin/soumitra-leg2; the scale trend's statistic (Spearman of model-size rank with AUC, two-sided permutation p) "
              "was fixed in AB5 (510633d, this branch), after the AB2 runs existed and before Anirudh had seen any AB2 value. "
              f"Run files read from {src['ref']} at {str(src['commit'])[:12]} "
              f"({len(src['files'])} files, each with its source, commit and sha256 in results/ablations/LLM-rtx.json; AB2's move manifest matches for "
              f"{sum(src['ab2_manifest_matches'].values())} of {len(src['ab2_manifest_matches'])} files). Every reference is local to the same card (AB1, AB4). "
              "ILI held-out AUC, pooled seed-SD units, lower is better. For a trend, positive rho means a higher (worse) AUC at the higher level: more steps, a larger KL weight, a larger model.", ""]
    lines += [RECORD_NOTE, ""]
    rows = [[v["item"] + (" ILI" if kind == "prompted" else ""), str(v.get("searches", v.get("deployed", ""))), ", ".join(map(str, v.get("stopped", []))) or "-",
             ", ".join(v.get("gpu", [])) or "-", "yes" if v["complete"] else "no",
             ("; ".join(v["problems"]) or "all pass") if v.get("searches", v.get("deployed", 0)) else "not checked (no runs)"]
            for kind in ("prompted", "transfer") for v in g["status"][kind].values()]
    lines += table(["item", "searches or deployed policies", "on-policy stops", "GPU", "complete", "checks (method, backend, revision, registration order, launcher, frozen files, settings)"], rows)
    lines += [f"Teammate branch: local tip {str(src.get('tip'))[:12]} (committed {src.get('tip_time')}); this clone last fetched at {src.get('fetch_time')}.", ""]
    if not src.get("pinned_is_tip", True):
        lines += [f"The pinned commit {str(src['commit'])[:12]} is not the local tip of origin/soumitra-leg2 ({str(src.get('tip'))[:12]}), so these results stay provisional until the pin is moved.", ""]
    for fam, size in (("AB-P", 3), ("AB-G", 2)):
        f = g["families"][fam]
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
        rows += [[name, f["missing_reasons"][name], "", "", "no (counted)"] for name in f["missing"]]
        lines += [f"{fam}, Holm over {size}:", ""] + table(["test", "result", "p", "p, Holm", "rejected"], rows)
    sc = g["families"]["AB-P"]["tests"].get("scale trend")
    if sc:
        lv = [sc["levels"][k]["mean"] for k in ("0.5", "1.5", "3.0")]
        e = {k: next(iter(x["summary"].values()))["auc"] for k, x in g.get("ETTh1_descriptive", {}).items()}
        order = ", ".join(f"{ {'ref': '1.5B'}.get(k, k)} {v['mean']:.2f}" for k, v in sorted(e.items(), key=lambda kv: kv[1]["mean"]))
        overlap = bool(e) and max(v["low"] for v in e.values()) <= min(v["high"] for v in e.values())
        lines += ["Scope of the scale trend: the three levels are three Qwen2.5-Instruct checkpoints (0.5B, 1.5B, 3B) under one fixed protocol "
                  "(the same prompt, temperature 1.0, constrained decoding, one card); they differ in more than size, so the test does not isolate size, "
                  f"and three points allow no extrapolation. The change is {lv[1] - lv[0]:+.2f} SD from 0.5B to 1.5B and {lv[2] - lv[1]:+.2f} SD from 1.5B to 3B; no pair was tested. "
                  "The levels share search seeds 0 to 29; the unpaired permutation test ignores that pairing. "
                  + (f"On ETTh1 (in-sample, descriptive, no test) the order by AUC is {order}" + (", and every pair of intervals overlaps" if overlap else "") + "; no test is made on ETTh1. " if order else "")
                  + "The 3B checkpoint is under the Qwen Research License (non-commercial; docs/neurips_checklist.md, item 12); releasing its generated proposals may need that licence's notice.", ""]
    if g["families"]["AB-G"]["tests"]:
        lines += ["The AB-G levels share training seeds 100 to 109 and deployment seeds 0 to 9 (and the steps levels share level 0), so they are dependent. "
                  "Diya's AB4 on main: training has no step-dependent schedule, so a 30-step policy is, up to GPU nondeterminism, a prefix of the 60- and 120-step policies of its seed, "
                  "and the step-trend permutation p is anti-conservative (its critic's simulation: a false-positive rate of 0.10 to 0.15 at nominal 0.05); "
                  "the step trend and its Holm decision are reported as registered, with this caveat. The KL levels share seeds too; with balanced levels AB4 states that this makes the KL test conservative. "
                  "The detectable rho values are Fisher-z approximations, optimistic with tied levels.", ""]
    rnd = (load("LLM") or {}).get("mps", {}).get("revin", {}).get("committed_ILI", {}).get("random (E3)")
    if rnd:
        lines += [f"For context (post hoc, no test): E3's random search has ILI AUC {ci(rnd['auc'])} on the same scale.", ""]
    pl = g["families"]["AB-G"].get("exploratory_policy_level")
    if pl:
        u, dose = pl["trained_vs_untrained"], pl["dose"]
        steps = g["families"]["AB-G"]["tests"].get("steps trend")
        lines += [f"Registered secondary step analyses (Diya's AB4 on main; policy as unit, no Holm): (i) trained at all, the ten policies' AUC averaged over their three steps levels against the 30 untrained searches, "
                  f"{u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}] (permutation p {pfmt(u['p'])}; the test could detect {u['detectable_80']:.2f} SD; positive is worse); "
                  f"(ii) dose, the per-policy Spearman correlation of AUC with 30, 60, 120 steps, mean {dose['mean_rho']:+.2f} over {dose['policies']} policies, exact sign flip p {dose['p_sign_flip']:.3f}. "
                  + step_reading(steps, u, dose) + " AB4's diagnostic (iii), the share of training-log rows shared with the 60-step log, is not computed here.", ""]
    desc = g["descriptive"]
    rows = []
    for key, label in (("hardware_G_ref_local_vs_J2", "ABL_G_ref_local (RTX) minus J2 league (A10/A10G), AB1 hardware check"),
                       ("prompted_ref_local_vs_mac", "ABL_P_ref_local (RTX) minus ABL_P_ref_mps (Mac), AB5"),
                       ("prompted_ref_local_vs_E3_A10", "ABL_P_ref_local (RTX) minus E3 prompted (A10), AB5")):
        v = desc.get(key)
        if v:
            rows.append([label, str(v["seeds"]), ci(v["a"]), ci(v["b"]), ci_signed(v["diff"])])
    if rows:
        widest = max((desc[k]["diff"]["high"] - desc[k]["diff"]["low"]) / 2 for k in ("hardware_G_ref_local_vs_J2", "prompted_ref_local_vs_mac", "prompted_ref_local_vs_E3_A10") if desc.get(k))
        lines += [f"AB1 hardware check and AB5 comparisons (descriptive, on shared seeds, no test; interval half-widths up to {widest:.2f} SD). These intervals do not show that the RTX matches the other machines; "
                  "the samplers differ, and the E3 searches predate the A25 environment pins.", ""]
        lines += table(["comparison", "seeds", "first", "second", "difference"], rows)
    if desc.get("revin_off_share_ILI"):
        sh = desc["revin_off_share_ILI"]
        sizes = [sh.get(f"prompted {k}") for k in ("0.5B", "ref", "3B")]
        monotone = None not in sizes and (sizes == sorted(sizes) or sizes == sorted(sizes, reverse=True))
        lines += ["RevIN-off share of ILI lookups (descriptive, AB5): " + "; ".join(f"{k} {v:.0%}" for k, v in sh.items()) + "."
                  + ("" if monotone or None in sizes else " Among the prompted models the share does not follow model size, so it does not follow the scale trend's order."), ""]
    if g.get("ETTh1_descriptive"):
        lines += ["ETTh1 (descriptive, in-sample):", ""]
        lines += table(["arm", "AUC", *(f"@{b}" for b in BUDGETS)],
                       [[{"ref": "1.5B (ref)"}.get(arm, arm), ci(next(iter(x["summary"].values()))["auc"]), *(ci(next(iter(x["summary"].values()))[str(b)]) for b in BUDGETS)]
                        for arm, x in g["ETTh1_descriptive"].items()])
    tr = [(v["item"], v["training"]) for v in g["status"]["transfer"].values() if v.get("training", {}).get("policies")]
    if tr:
        lines += ["Training (RTX transfer items): " + "; ".join(f"{k} {x['policies']} policies, largest mean on-policy gap {x['onpolicy_mean_max']:.4f} (limit 0.05), errors {len(x['errors'])}" for k, x in tr) + ".", ""]
    return lines


def llm() -> list[str]:
    d = load("LLM")
    lines = ["## AB-P and AB-G on the Mac (exploratory, AB3)", ""]
    if not d:
        return lines + [MISSING, ""]
    m = d["mps"]
    lines += ["The Modal items (ABL_P_ref, ABL_P_anon, ABL_P_repair, ABL_G_steps30, ABL_G_steps120) were never launched (AB3); their questions are registered on the RTX (AB4, previous section). ",
              "Everything below ran on an Apple M4 Pro (torch MPS), each Mac arm against its own reference on the same machine (the A10 columns of the hardware check are committed runs). It is exploratory (AB1, AB3): "
              "it replaces no registered family. " + provenance_line(d), ""]
    p = m["AB-P"]
    lines += ["### AB-P on the Mac: prompted Qwen, ILI held-out AUC", "", "Finished searches: " + ", ".join(f"{k} {v}" for k, v in p["complete"].items()) + ".", ""]
    if p["tests"]:
        lines += family(p["tests"], 3, "test not part of this exploratory set: the scale trend (moved to Soumitra by AB1)")
    if p["diagnostics"]:
        rep = p["diagnostics"].get("repair")
        lines += ["ILI diagnostics:", ""] + table(["arm", "searches", "lookups", "repeat rate", "repaired rate", "invalid rate"],
                       [[a, str(x["searches"]), f"{x['lookups']['mean']:.0f}", f"{x['repeat_rate']:.2f}", f"{x['repaired_rate']:.2f}", f"{x['invalid_rate']:.2f}"] for a, x in p["diagnostics"].items()])
        if rep:
            lines += [f"The repair arm's comparison is confounded by design: {rep['repaired_rate']:.0%} of its ILI proposals are replaced by the nearest unseen config (its repeat rate is "
                      f"{rep['repeat_rate']:.2f}, against {p['diagnostics']['ref']['repeat_rate']:.2f} for ref), so it measures removing the constrained decoding together with that replacement.", ""]
    if m.get("ETTh1_descriptive"):
        lines += ["ETTh1 (descriptive, in-sample):", ""]
        if m.get("ETTh1_differences"):
            d_ = m["ETTh1_differences"]
            lines += table(["ETTh1 arm", "AUC difference from ref (SD units, no test)"],
                           [[k, f"{v['diff']:+.2f} [{v['low']:+.2f}, {v['high']:+.2f}] (30 seeds per arm)"] for k, v in d_.items()])
            lines += ["Both signs are the reverse of ILI: on ETTh1 the anonymised prompt does slightly better and repair worse. ETTh1 has one seed per config, so these are in-sample and no test or p value is given; no mechanism is claimed.", ""]
        rows = []
        for arm, x in m["ETTh1_descriptive"].items():
            for method, s in x["summary"].items():
                rows.append([arm, ci(s["auc"]), *(ci(s[str(b)]) for b in BUDGETS)])
        lines += table(["arm", "AUC", *(f"@{b}" for b in BUDGETS)], rows)
    g = m["AB-G"]
    lines += ["### AB-G on the Mac: GRPO training steps, ETTh1 to ILI", "", "Finished runs per level: " + ", ".join(f"{k} steps: {v}" for k, v in g["complete"].items()) + ".", ""]
    if g["tests"]:
        t = g["tests"]["step trend"]
        lines += table(["steps", "runs", "ILI held-out AUC"], [[k, str(v["n"]), ci(v)] for k, v in sorted(t["levels"].items(), key=lambda kv: int(kv[0]))])
        u, dose = m["policy_level"]["trained_vs_untrained"], m["policy_level"]["dose"]
        u, dose, pair = m["policy_level"]["trained_vs_untrained"], m["policy_level"]["dose"], m["paired"]
        lines += [f"Trained at all (exploratory, AB3 addendum 3): the ten policies' AUC, averaged over their three checkpoints, is {u['diff']:+.2f} SD [{u['low']:+.2f}, {u['high']:+.2f}] against the 30 untrained searches "
                  f"(permutation p {pfmt(u['p'])}; the test could detect {u['detectable_80']:.2f} SD). Positive is worse. Against the untrained search of the same deployment seed the gap is {ci(pair['gap'])} SD, "
                  f"worse for {pair['worse']} of {pair['policies']} policies (addendum 4).", ""]
        lines += ["Scope: policies trained with league rivals on the ETTh1 table (one seed per config) and deployed on ILI. It shows that they do worse on ILI than the untrained model. It does not show that GRPO or the league is harmful: "
                  "there is no ILI-source arm, no same-table deployment and no uniform or none arm on the Mac. Policies' checkpoints are dependent (one training run each).", ""]
        rv = m["revin"]
        lines += ["RevIN preference (descriptive, addendum 4). The table gives the share of the lookups on ILI with RevIN off, and the AUC of the same committed runs beside it.", ""]
        rows = [[f"level {k}", f"{v:.0%}", "", ""] for k, v in sorted(rv["mac_share_off_ILI"].items(), key=lambda kv: int(kv[0].split()[0]))] + \
               [[k, f"{v['share_off']:.0%}", ci(v["auc"]), str(v["runs"])] for k, v in rv["committed_ILI"].items()]
        lines += table(["runs", "RevIN-off share of ILI lookups", "ILI AUC (committed runs)", "runs in group"], rows)
        b = rv["best_regret_by_revin"]
        lines += [f"The best RevIN-off config on ILI has regret {b['ILI_T24']['revin_off']:.2f} SD (RevIN-on best: {b['ILI_T24']['revin_on']:.2f}), and on ETTh1 the best RevIN-on config has {b['ETTh1_T336']['revin_on']:.1f} SD "
                  f"(RevIN-off best: {b['ETTh1_T336']['revin_off']:.2f}). Trained policies, at 30 steps already, look up almost only RevIN-off configs, so a preference that is right on ETTh1 costs on ILI. "
                  "This is consistent with the transfer loss but is not tested: no arm removes the preference while keeping the training.", ""]
        lines += [f"Dose (exploratory, addendum 3): the per-policy Spearman correlation of AUC with 30, 60, 120 steps averages {dose['mean_rho']:+.2f} over {dose['policies']} policies, exact sign flip p {dose['p_sign_flip']:.3f} "
                  "(three ordered points, little power).", ""]
        sd = m["step_diagnostics"]
        lines += table(["checkpoint", "searches", "lookups", "repeat rate", "invalid rate", "logged updates per policy"],
                       [[k, str(v["searches"]), f"{v['lookups']['mean']:.0f}", f"{v['repeat_rate']:.2f}", f"{v['invalid_rate']:.2f}", ci(v["logged_updates"], 1)] for k, v in sorted(sd.items(), key=lambda kv: int(kv[0]))])
        lines += ["A step with no budget left logs no update, so the levels are about 29, 57 and 113 updates at 30, 60 and 120 steps.", ""]
        lines += [f"AB-G's registered statistic, computed on these Mac runs (Spearman of the level with AUC over 60 values): rho {t['rho']:+.2f}, p {pfmt(t['p'])}, Holm over 2 {pfmt(t['p_holm'])}. "
                  "This p is not valid (AB3 addendum 3): levels 30, 60 and 120 are checkpoints of the same policies, and level 0 (untrained) drives the rank correlation. It is therefore not reported as a rejection. "
                  f"With 10 seeds per level: rho {m['step_trend_10_per_level']['rho']:+.2f}, p {pfmt(m['step_trend_10_per_level']['p'])} (same caveat). "
                  f"Trained levels only (30, 60, 120): rho {m['step_trend_trained_levels_only']['rho']:+.2f}, p {m['step_trend_trained_levels_only']['p']:.3f} (same caveat). ", ""]
    if g["training"]["policies"]:
        tr = g["training"]
        lines += [f"Training: {tr['policies']} policies, largest mean on-policy gap {tr['onpolicy_mean_max']:.4f} (limit 0.05), mean unique rate "
                  f"{tr['unique_rate_mean']:.2f}, errors: {len(tr['errors'])}.", ""]
    if d["hardware_check"]:
        lines += ["### Hardware check (no test)", ""]
        rows = [[k, str(v["seeds"]), ci(v["mps"]), ci(v.get("a10_E3") or v.get("a10_J2")), ci(v["diff"])] for k, v in d["hardware_check"].items()]
        lines += table(["run", "shared seeds", "Mac", "A10", "Mac minus A10"], rows)
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
              "An observed difference well below the detectable one is not evidence of no effect. " + provenance_line(d), ""]
    groups = {"E3 against random (ILI, SD units)": [], "Picks on fresh seeds (validation MSE)": [], "E4, AB-C, AB-X, AB-P, AB-L": [], "Trends (rank correlation)": [],
              "E5 (interval half width)": [], "E6 planning (validation MSE)": []}
    for r in d["rows"]:
        name = r["comparison"]
        key = ("E5 (interval half width)" if name.startswith("E5") else "E6 planning (validation MSE)" if name.startswith("E6") else
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
            body.append([r["comparison"], " / ".join(str(x) for x in r["n"]), f"{r['diff']:+.{digits}f}" if r.get("diff") is not None else "", shown, str(r["family"]),
                         p3(r["p"]) if r.get("p") is not None else ""])
        lines += table(["comparison", "n", "observed difference", "detectable at 80%", "Holm size", "p"], body)
    return lines


def robustness() -> list[str]:
    d = load("ST-robust")
    lines = ["## ST: robust checks (descriptive)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += ["Interquartile mean of regret and probability of improvement over random search, 95% bootstrap intervals over search seeds within arms. " + provenance_line(d), ""]
    for dataset, _ in SETTINGS:
        t = d["tables"][setting(dataset)]
        share = t["summary"]["random"]["heldout_share"] if "heldout_share" in t["summary"]["random"] else {}
        rows = [[m, f"{p['iqm_sd']['50']:.2f} [{p['iqm_low']['50']:.2f}, {p['iqm_high']['50']:.2f}]",
                 f"{p['pi']['25']:.2f} [{p['pi_low']['25']:.2f}, {p['pi_high']['25']:.2f}]", f"{p['pi']['50']:.2f} [{p['pi_low']['50']:.2f}, {p['pi_high']['50']:.2f}]",
                 f"{t['summary'][m]['50_raw_mse']['mean']:.4f}"] for m, p in sorted(t["profiles"].items(), key=lambda kv: kv[1]["iqm_sd"]["50"])]
        note = f" (held-out share of incumbents, random search: {', '.join(f'@{b} {v:.0%}' for b, v in share.items())})" if share and dataset == "ETTh1" else ""
        lines += [f"{dataset}{note}:", ""]
        lines += table(["method", "IQM regret @50 (SD)", "P(improvement) @25", "P(improvement) @50", "mean regret @50 (raw MSE)"], rows)
    mad = d["tables"]["ETTh1_T336"]["mad"]
    pooled = d["tables"]["ETTh1_T336"]["pooled_sd"]
    lines += [f"ETTh1 with the robust seed SD: {mad['sd']:.5f} against the pooled {pooled:.5f} (ratio {mad['ratio_to_pooled']:.2f}). Replayed with the robust SD as noise and unit, AUC: "
              + ", ".join(f"{m} {v['mean']:.2f}" for m, v in sorted(mad["auc_sd"].items(), key=lambda kv: kv[1]["mean"])) + ".", ""]
    return lines


def rq3() -> list[str]:
    d = load("RQ3")
    lines = ["## RQ3: which design choices matter (descriptive)", ""]
    if not d:
        return lines + [MISSING, ""]
    lines += [provenance_line(d), ""]
    fields = sorted(next(iter(d["tables"].values()))["main_effect_share"])
    ili, ett = d["tables"]["ILI_T24"]["main_effect_share"], d["tables"]["ETTh1_T336"]["main_effect_share"]
    lv = d["cross_table"]["level_means"]
    c2 = {d2: d["tables"][d2]["conditional_share"] for d2 in ("ILI_T24", "ETTh1_T336")}
    lines += ["Marginal shares are registered; conditional shares (what a field adds to a model that has the other seven) are exploratory.", ""]
    lines += table(["field", "ILI marginal / conditional", "ETTh1 marginal / conditional", "best level on ILI", "best level on ETTh1", "same"],
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
        lines += [f"{dataset}: the additive model explains {t['additive_r2']:.3f} of the variance of the config means and all pairwise interactions "
                  f"{t['all_pairs_r2']:.3f}. Largest pair increments: " + ", ".join(f"{k} {v:.3f}" for k, v in top) + "." + extra, ""]
    c = d["cross_table"]
    lines += [f"Agreement of the field shares across the two tables: Spearman {c['share_spearman']:.2f}, Kendall {c['share_kendall']:.2f} (marginal shares, 8 fields); "
              f"Spearman {c['conditional_share_spearman']:.2f} for the conditional shares (the share a field adds to a model that has the others). Marginal shares and best levels come from an unbalanced set, "
              "so a best level can flip when the other fields are held fixed.", ""]
    r = d["revin"]
    lines += [f"RevIN on ETTh1, validation only, {r['pairs']} twin pairs: the RevIN-off advantage is {ci(r['epochs_30']['off_advantage'], 4)} at 30 epochs and "
              f"{ci(r['epochs_100']['off_advantage'], 4)} at 100 epochs (ratio {r['shrinkage_30_to_100']['ratio_of_means']:.2f}; change {ci(r['shrinkage_30_to_100']['mean_change'], 4)}); "
              f"off is better in {r['epochs_30']['share_pairs_off_better']:.0%} and {r['epochs_100']['share_pairs_off_better']:.0%} of pairs. Share of RevIN-on runs whose best epoch is the last one: "
              f"{r['epochs_30']['on_configs_best_epoch_at_cap']:.0%} at 30 epochs, {r['epochs_100']['on_configs_best_epoch_at_cap']:.0%} at 100. {r['caveat']}. By validation month at 100 epochs: "
              + ", ".join(f"{x:.4f}" for x in r["epochs_100"]["off_advantage_by_month"]) + " (prediction file, 30 epochs, all 1,184 pairs: "
              + ", ".join(f"{x:.4f}" for x in r["prediction"]["off_advantage_by_month"]) + "). The test part stays with Diya's single evaluation.", ""]
    return lines


def glance() -> list[str]:
    """The main readings, each built from the result files."""
    lines = ["## At a glance", ""]
    q, x, c, n, r, a, g, w = (load(k) for k in ("AB-Q", "AB-X", "AB-C", "AB-L-null", "AB-R", "RQ3", "LLM", "ST-power"))
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
        ili = [v["detectable_80_sd"] for v in picks if v["comparison"].startswith("ILI")]
        ett = [v["detectable_80_robust_sd"] for v in picks if v["comparison"].startswith("ETTh1")]
        ratio = sorted(abs(v["diff"]) / v["detectable_80"] for v in picks)
        worse = f" The one E4 comparison that rejects, {rejected[0]['comparison'].split(': ')[1]}, is a loss, not a null: the league is {rejected[0]['diff']:+.2f} SD worse." if rejected else ""
        lines.append(f"* The nulls have limited resolution. At budget 50 the pick reseeds could detect differences of {min(ili):.2f} to {max(ili):.2f} SD on ILI and, in robust-SD units, {min(ett):.2f} to {max(ett):.2f} on ETTh1 (80% power, Holm size in the table); the "
                     f"observed non-significant differences are {ratio[0]:.0%} to {ratio[-1]:.0%} of that. E4's ILI nulls (league against the portfolio and against no league, 10 policies) could detect {min(e4):.2f} to {max(e4):.2f} SD.{worse} (ST)")
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
        reg = rtx["registered"]
        def said(fam: str, name: str, v: dict) -> str:
            decision = f"Holm p {pfmt(v['p_holm'])}, {'rejected' if v['reject'] else 'not rejected'}"
            if "rho" not in v:
                return f"{fam} {name}: {v['diff']:+.2f} SD, {decision}"
            det = next((r["detectable_80"] for r in (load("ST-power") or {}).get("rows", []) if r["comparison"].startswith(f"{fam} {name} on the RTX")), None)
            could = f", could detect |rho| of about {det:.2f}" if det else ""
            if name == "scale trend":
                way = "a larger model gave a higher, that is worse," if v["rho"] > 0 else "a larger model gave a lower, that is better,"
                return f"{fam} {name}: {way} ILI held-out AUC ({trend_words(name, v)} SD; rho {v['rho']:+.2f}, {decision}{could}; ILI only, three checkpoints, size not isolated)"
            nested = "; p anti-conservative, nested levels (AB4)" if name == "steps trend" else ""
            return f"{fam} {name}: ILI held-out AUC {trend_words(name, v)} SD (rho {v['rho']:+.2f}, {decision}{could}{nested})"

        done = [said(fam, name, v) for fam in ("AB-P", "AB-G") for name, v in reg["families"][fam]["tests"].items()]
        pl = reg["families"]["AB-G"].get("exploratory_policy_level")
        if pl and "steps trend" in reg["families"]["AB-G"]["tests"]:
            done.append("read by AB4's rule, " + step_reading(reg["families"]["AB-G"]["tests"]["steps trend"], pl["trained_vs_untrained"], pl["dose"])
                        .removeprefix("By AB4's rule ").rstrip(".") + f" (trained minus untrained {pl['trained_vs_untrained']['diff']:+.2f} SD)")
        missing = [name for fam in ("AB-P", "AB-G") for name in reg["families"][fam]["missing"]]
        lines.append("* Registered LLM ablations on Soumitra's RTX, Anirudh's independent computation (" + ("final" if reg["final"] else "provisional")
                     + "; the result of record is results/ablations/LLM-registered.json, Diya's AB4 on main): " + ("; ".join(done) or "none complete")
                     + (f"; missing so far, counted as not rejected: {', '.join(missing)}" if missing else "") + " (AB2, AB4).")
    if g and g["mps"]["AB-P"]["tests"]:
        t = g["mps"]["AB-P"]["tests"]
        etth1_d = g["mps"].get("ETTh1_differences", {})
        opposite = "; on ETTh1 (in-sample, descriptive) the signs reverse: " + ", ".join(f"{k} {v['diff']:+.2f}" for k, v in etth1_d.items()) if etth1_d else ""
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
        lines.append(f"* On the ETTh1 source GRPO raised the league policies' mean reward and on ILI no trend was detected. The no-league arm's ETTh1 slope is {none:+.3f}, at least as large, so nothing here says the league did it. The registered AB-L test is too liberal (false-positive rate {e['raw']['false_positive_rate']['0.025']:.2f} and "
                     f"{i['raw']['false_positive_rate']['0.025']:.2f} at the 0.025 level for a model that never learns, from a null of ten searches per table"
                     + (f"; from thirty Mac searches (exploratory) {mac_e['raw']['false_positive_rate']['0.025']:.2f} on ETTh1 and {mac['raw']['false_positive_rate']['0.025']:.2f} on ILI" if mac else "") + "), but the position-adjusted slopes give the same reading: "
                     f"exact sign flip of the {o['ETTh1']['n']} adjusted ETTh1 slopes p {o['ETTh1']['p_sign_flip']:.3f} (uncorrected; Holm over the two tables {min(1.0, 2 * o['ETTh1']['p_sign_flip']):.3f}), ILI p {o['ILI']['p_sign_flip']:.2f}; exploratory, post hoc.")
    if r:
        t = r["tables"]["ILI_T24"]
        pick = lambda f, k, key="tau": next(v[key] if key == "tau" else v["tau_independent_seeds"]["tau"] for v in t["variants"] if v["features"] == f and v["size"] == 300 and v["model"] == "gbt" and v["k"] == k)
        lines.append(f"* The referee's skill on ILI comes mostly from the config fields: tau {pick('config', None):.2f} from fields alone, {pick('curves', 1):.2f} from curves alone, {pick('both', 1):.2f} from both, against the 3-seed mean. "
                     f"That target contains the seed-2021 value the referee is trained on; against the mean of the other two seeds the same three are {pick('config', None, 'ind'):.2f}, {pick('curves', 1, 'ind'):.2f} and {pick('both', 1, 'ind'):.2f} (AB-R).")
    if a:
        lines.append(f"* The two tables show no detectable agreement on which fields matter (Spearman {a['cross_table']['share_spearman']:.2f} between marginal field shares, {a['cross_table']['conditional_share_spearman']:.2f} between exploratory conditional shares, 8 fields); RevIN explains {a['tables']['ETTh1_T336']['main_effect_share']['instance_norm']:.0%} "
                     f"of ETTh1's variance and {a['tables']['ILI_T24']['main_effect_share']['instance_norm']:.0%} of ILI's, and its ETTh1 validation advantage for RevIN off does not shrink from 30 to 100 epochs (that comparison also changes patience from 10 to 100, RQ3).")
    return lines + [""]


def history() -> list[str]:
    return ["## Runs, re-runs and open questions", "",
            "* Registration: AB3 is commit 31f9313 (00:58 on 25 Nov); the first CPU outputs were written after it. AB3 addendum 1 and 2 follow a verifier and a critic "
            "review of the first outputs and are exploratory. The branch was not on GitHub when the runs started, so the order is shown by commit times only.",
            "* AB-X ran three times: once with a provenance stamp marked dirty (the report generator had been added to the tree, not the code that ran), once from a clean commit, and once more after run records "
            "gained a git hash. The first two run files were byte-identical; the third differs only by the added field. The CPU analyses were re-run once after the verifier's fixes; the registered numbers "
            "were produced by scripts unchanged since 31f9313 except where addendum 1 and 2 say otherwise, and their first outputs are not kept.",
            "* Not done from WP6: regret curves with intervals as figures (docs/ablations.md gives tables at fixed budgets only).",
            "* Not answered here: why TPE wins E3's held-out replay but not the pick reseeds (an effect-size question, no equivalence test exists yet); whether rankings survive resampling "
            "of the three table seeds; whether results survive a different fidelity ladder or a common first configuration; BOHB and one-epoch-then-top-K baselines; and, for AB-L, how much of the ETTh1 slope the league contributes.",
            "* Mac LLM runs: the ILI prompted reference (ABL_P_ref_mps) ran at commit 31f9313 and every other Mac item at 0929d26; scripts/ablations/llm_mps.py and leaguetst/ are identical between the two. Every run file's commit and digest is in results/ablations/LLM.json. "
            "AB3 addendum 3 (policy-level and dose analyses) was committed before its outputs; addendum 4 (RevIN shares, paired gap, diagnostics) was written after a critic had reported them and its code was run once before the commit, as it says.",
            "* Verifier and critic reports (AI subagents), quoted in full, are in docs/verdicts/ablations_cpu_reviews.md and docs/verdicts/ablations_llm_reviews.md.", ""]


def main() -> int:
    lines = ["# Ablations, robustness and external validity", "",
             "Generated by scripts/ablations/report.py from results/ablations/*.json. Do not edit by hand. Registered in "
             "docs/prereg_ablations.md (AB1, AB3 and AB5 on this branch; AB2 and AB4 on origin/soumitra-leg2; Diya's AB4 on main); every result here is secondary to the primary endpoints. Regret is in pooled "
             "seed-SD units unless stated; lower is better. ILI regret is scored on seeds the search never saw; ETTh1 has one seed per config, "
             "so ETTh1 replays are in-sample and descriptive.", ""]
    lines += glance() + noise_free() + external() + controller() + registered_llm() + llm() + learned() + one_seed() + referee() + rq3() + robustness() + power() + history()
    TARGET.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
