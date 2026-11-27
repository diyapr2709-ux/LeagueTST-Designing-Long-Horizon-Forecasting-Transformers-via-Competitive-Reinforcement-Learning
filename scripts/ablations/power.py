"""ST, resolution part: the smallest difference each registered comparison could detect (descriptive). CPU.

For every registered two-sample comparison with runs in the repository: the
seeds and per-seed SDs of both arms, the observed difference, and the smallest
true difference that a two-sided test at 0.05/family would find with 80% power
(normal approximation with the observed SDs; family = the registered Holm
size). Trends report the smallest detectable rank correlation (Fisher z).
AB3 registered that normal approximation, so it stays the detectable_80 column. Added 26 Nov, after the
results were seen (area-chair review B18), descriptive: detectable_80_t, the same quantity with Student t
quantiles at the Welch degrees of freedom, and t_ratio, its ratio to detectable_80, for rows with per-arm SDs.
Sources: docs/claims/E3.json, results/picks, the E4 runs files, AB-C, AB-X,
the LLM arms, AB-L, and, as a planning figure only, the E6 validation errors
(the E6 primary endpoint is the test split, which nothing here reads).

    .venv/bin/python scripts/ablations/power.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust import mad_sd  # noqa: E402
from common import ALPHA, OUT, POWER, ROOT, by_seed, detectable, load_table, provenance, read_runs, setting, write_json  # noqa: E402

from leaguetst.ledger import read_rows  # noqa: E402


def detectable_t(sd_a: float, n_a: int, sd_b: float, n_b: int, family: int) -> float | None:
    """As common.detectable, with Student t quantiles at the Welch degrees of freedom (descriptive, not registered)."""
    from scipy.stats import t as student_t

    va, vb = sd_a**2 / n_a, sd_b**2 / n_b
    if n_a < 2 or n_b < 2 or va + vb <= 0:
        return None
    df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
    return float((student_t.ppf(1 - ALPHA / family / 2, df) + student_t.ppf(POWER, df)) * math.sqrt(va + vb))


def row(name: str, a, b, family: int, unit: str, p=None, note: str = "") -> dict:
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    out = {"comparison": name, "n": [len(a), len(b)], "sd": [float(a.std(ddof=1)), float(b.std(ddof=1))],
           "mean": [float(a.mean()), float(b.mean())], "diff": float(a.mean() - b.mean()), "family": family, "unit": unit,
           "detectable_80": detectable(a.std(ddof=1), len(a), b.std(ddof=1), len(b), family), "p": p, "note": note}
    t_based = detectable_t(a.std(ddof=1), len(a), b.std(ddof=1), len(b), family)
    out["detectable_80_t"] = t_based
    out["t_ratio"] = t_based / out["detectable_80"] if t_based and out["detectable_80"] else None
    return out


def detectable_rho(n: int, family: int) -> float:
    from scipy.stats import norm

    return float(math.tanh((norm.ppf(1 - ALPHA / family / 2) + norm.ppf(POWER)) / math.sqrt(n - 3)))


def e3_rows() -> list[dict]:
    claims = json.loads((ROOT / "docs/claims/E3.json").read_text())["claims"]
    sizes: dict = defaultdict(int)
    for c in claims:
        sizes[c["family"]] += 1
    return [row(c["name"], c["a_values"], c["b_values"], sizes[c["family"]], "pooled seed SD", c["p"], "held-out regret on ILI")
            for c in claims]


def pick_rows(dataset: str) -> list[dict]:
    """Fresh-seed validation MSE of the picks of the methods in results/picks (E3 and the E4 transfer controls)."""
    import picks
    from leaguetst.search.replay import Table

    horizon = int(setting(dataset).split("_T")[1])
    analysis = json.loads((ROOT / "results/picks" / f"{setting(dataset)}_analysis.json").read_text())
    table = Table(picks.ledger("E1", dataset, horizon))
    fresh = picks.fresh_values(dataset, horizon)
    runs = [r for item in ("E3", "E4") for r in picks.load_runs(ROOT / "results/search" / item / setting(dataset))]
    values: dict = defaultdict(list)
    for method, _, b, cid in picks.incumbents(runs, table):
        if method in analysis["methods"] and cid in fresh:
            values[(method, b)].append(fresh[cid])
    family = len(analysis["vs_random"])
    sigma = table.seed_sd()
    out = []
    for name, test in sorted(analysis["vs_random"].items()):
        method, b = name.split(" vs random @")
        a, r = values[(method, int(b))], values[("random", int(b))]
        entry = row(f"{dataset} picks: {name}", a, r, family, "validation MSE", test["p"], "fresh seeds 4001 and 4002")
        entry["detectable_80_sd"] = entry["detectable_80"] / sigma
        if dataset == "ETTh1":
            entry["detectable_80_robust_sd"] = entry["detectable_80"] / mad_sd(table)
        out.append(entry)
    return out


def e4_rows() -> list[dict]:
    table, _ = load_table("ILI")
    folder = ROOT / "results/search/E4/ILI_T24"
    runs = [r for p in sorted(folder.glob("*_runs.jsonl")) for r in read_runs(p)]
    auc = lambda m: list(by_seed(runs, m, table).values())
    e3 = read_runs(ROOT / "results/search/E3/ILI_T24/cpu_runs.jsonl") + read_runs(ROOT / "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl")
    league = auc("grpo_league_from_ETTh1")
    e4 = json.loads((ROOT / "results/independent/E4_soumitra.json").read_text())
    reject = {name: h["reject"] for name, h in e4["family_1"]["holm_with_pending_as_not_rejected"].items()}
    key = {"portfolio_from_ETTh1": "F1 ILI AUC league vs portfolio", "tpe_warm_from_ETTh1": "F1 ILI AUC league vs tpe_warm"}
    out = [row(f"E4 family 1, ILI AUC: league vs {b}", league, auc(b), 6, "pooled seed SD", None,
               "Holm size 6 (A25, A30); " + ("REJECTED with the league worse, not a null" if reject[key[b]] else "not rejected"))
           for b in ("portfolio_from_ETTh1", "tpe_warm_from_ETTh1")]
    out.append(row("E4 family 2, ILI AUC: league vs none", league, auc("grpo_none_from_ETTh1"), 6, "pooled seed SD", None, "Holm size 6 (A25, A30)"))
    q = list(by_seed(e3, "qwen_prompted", table).values())
    for m in ("random", "tpe"):
        out.append(row(f"E4 family 3, ILI AUC (shown on AUC): prompted Qwen vs {m}", q, by_seed(e3, m, table).values(), 1,
                       "pooled seed SD", None, "unadjusted; the registered tests are at budgets 25 and 50"))
    return out


def family_rows(path: Path, name: str, source: dict) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    tests = source(data)
    return [{"comparison": f"{name}: {k}", "n": t["n"], "mean": t["mean"], "diff": t["diff"], "family": 3, "unit": "pooled seed SD",
             "detectable_80": t["detectable_80"], "p": t["p"], "note": t.get("note", "")} for k, t in tests.items()]


def abl_rows() -> list[dict]:
    out = family_rows(OUT / "AB-C.json", "AB-C, ILI AUC", lambda d: d["tables"]["ILI_T24"]["tests"])
    out += family_rows(OUT / "AB-X.json", "AB-X, ILI AUC", lambda d: d["tables"]["ILI_T24"]["tests"])
    out += family_rows(OUT / "LLM.json", "AB-P on the Mac (exploratory), ILI AUC", lambda d: d["mps"]["AB-P"]["tests"])
    out += family_rows(OUT / "LLM-rtx.json", "AB-P on the RTX (registered in AB2 and Soumitra's AB4, 15db163; not of record), ILI AUC",
                       lambda d: {k: v for k, v in d.get("registered", {}).get("families", {}).get("AB-P", {}).get("tests", {}).items() if "diff" in v})
    data = json.loads((OUT / "AB-L.json").read_text())
    for dataset in ("ILI", "ETTh1"):
        slopes = [v["spearman_slope"] for v in data["by_source"][dataset]["per_policy"].values() if v["arm"] == "league"]
        sd = float(np.std(slopes, ddof=1))
        from scipy.stats import norm
        out.append({"comparison": f"AB-L: league slopes vs zero, {dataset} source", "n": [len(slopes)], "sd": [sd],
                    "mean": [float(np.mean(slopes))], "diff": float(np.mean(slopes)), "family": 2, "unit": "Spearman slope",
                    "detectable_80": float((norm.ppf(1 - ALPHA / 2 / 2) + norm.ppf(POWER)) * sd / math.sqrt(len(slopes))),
                    "p": data["tests"][f"league slopes vs 0, {dataset} source"]["p"],
                    "note": "Soumitra's AB-L; the registered test is against zero, but the null is not centred at zero (AB-L section)"})
    return out


TREND_AMENDMENT = {"scale trend": "AB2, statistic fixed in AB5", "KL trend": "AB2", "steps trend": "Soumitra's AB4, 15db163"}


def prompted_rows() -> list[dict]:
    """Added 26 Nov after the results were seen (results review R6), descriptive: what E3's 10-seed prompted arm could detect
    against each 30-seed E3 baseline on ILI AUC (common.detectable with the observed SDs, one comparison, unadjusted)."""
    table, _ = load_table("ILI")
    e3 = ROOT / "results/search/E3/ILI_T24"
    q = list(by_seed(read_runs(e3 / "qwen_prompted_runs.jsonl"), "qwen_prompted", table).values())
    cpu = read_runs(e3 / "cpu_runs.jsonl")
    return [row(f"E3 prompted Qwen (10 seeds, A10) vs {m} (30), ILI AUC", q, by_seed(cpu, m, table).values(), 1, "pooled seed SD", None,
                "descriptive, unadjusted; no test") for m in ("random", "re", "tpe", "asha", "referee", "pg", "pg_league")]


def trend_rows() -> list[dict]:
    """Planned sizes; the detectable rho is a Fisher-z approximation, optimistic with heavily tied level ranks."""
    note = "approximate: Fisher z for a rank correlation with heavily tied level ranks"
    rows = [{"comparison": "AB-G step trend (30 + 10 + 10 + 10 runs)", "n": [60], "family": 2, "unit": "Spearman rho", "detectable_80": detectable_rho(60, 2)},
            {"comparison": "AB-G step trend, 10 per level", "n": [40], "family": 2, "unit": "Spearman rho", "detectable_80": detectable_rho(40, 2)},
            {"comparison": "AB-G KL trend (10 + 10 + 10 runs)", "n": [30], "family": 2, "unit": "Spearman rho", "detectable_80": detectable_rho(30, 2)},
            {"comparison": "AB-P scale trend (30 + 30 + 30 runs)", "n": [90], "family": 3, "unit": "Spearman rho", "detectable_80": detectable_rho(90, 3)}]
    if (OUT / "LLM-rtx.json").exists():
        reg = json.loads((OUT / "LLM-rtx.json").read_text()).get("registered", {})
        fams = reg.get("families", {})
        for fam, size in (("AB-P", 3), ("AB-G", 2)):
            for name, t in fams.get(fam, {}).get("tests", {}).items():
                if "rho" in t:
                    rows.append({"comparison": f"{fam} {name} on the RTX (registered in {TREND_AMENDMENT.get(name, 'AB2')}; not of record)", "n": [t["n"]], "family": size, "unit": "Spearman rho",
                                 "detectable_80": detectable_rho(t["n"], size), "diff": t["rho"], "p": t["p"]})
    return [{**r, "note": r.get("note") or note} for r in rows]


def e6_rows() -> list[dict]:
    """Planning figure: the 4-horizon mean of validation MSE per seed, E6 arms (the registered endpoint is test MSE)."""
    per: dict = defaultdict(lambda: defaultdict(dict))
    for r in read_rows(ROOT / "results" / "ledger.jsonl"):
        if r["item"] == "E6":
            per[(r["dataset"], r["config_id"])][int(r["seed"])][int(r["horizon"])] = float(r["best_val_mse"])
    out = []
    for dataset in ("ETTh1", "ILI"):
        arms = {cid: [np.mean(list(h.values())) for h in seeds.values() if len(h) == 4] for (d, cid), seeds in per.items() if d == dataset}
        names = sorted(arms)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out.append(row(f"E6 {dataset}, validation: {a} vs {b}", arms[a], arms[b], 2, "validation MSE, 4-horizon mean", None,
                               "planning only: the registered endpoint is the test split, evaluated once by Diya"))
    return out


def e5_rows() -> list[dict]:
    out = []
    for dataset in ("ILI", "ETTh1"):
        data = json.loads((ROOT / "results/e5" / f"{setting(dataset)}.json").read_text())
        for method, block in sorted(data["methods"].items()):
            v = block["summary"]["tau_true"]["matched"]
            out.append({"comparison": f"E5 {dataset}: {method}, tau against matched sets", "n": [block["summary"]["runs"]], "diff": v["diff"],
                        "family": len(data["methods"]), "unit": "Kendall tau", "detectable_80": None,
                        "half_width": (v["high"] - v["low"]) / 2, "note": "half width of the registered interval; margin 0.05"})
    return out


def main() -> int:
    rows = e3_rows() + pick_rows("ILI") + pick_rows("ETTh1") + e4_rows() + prompted_rows() + abl_rows() + trend_rows() + e5_rows() + e6_rows()
    files = ["results/ledger.jsonl", "docs/claims/E3.json", "results/independent/E4_soumitra.json", "results/picks/ILI_T24_analysis.json",
             "results/picks/ETTh1_T336_analysis.json", "results/e5/ILI_T24.json", "results/e5/ETTh1_T336.json",
             *(f"results/search/E3/{s}/{f}" for s in ("ILI_T24", "ETTh1_T336") for f in ("cpu_runs.jsonl", "qwen_prompted_runs.jsonl")),
             *(f"results/search/E4/{s}/{f}" for s in ("ILI_T24", "ETTh1_T336") for f in ("grpo_league_from_ETTh1_runs.jsonl", "grpo_none_from_ETTh1_runs.jsonl", "grpo_league_from_ILI_runs.jsonl") if (ROOT / f"results/search/E4/{s}/{f}").exists()),
             *(f"results/ablations/{n}.json" for n in ("AB-C", "AB-X", "AB-L", "LLM") if (OUT / f"{n}.json").exists())]
    write_json(OUT / "ST-power.json", {"analysis": "ST detectable effects", "descriptive": True, "rows": rows, "provenance": provenance(__file__, files)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
