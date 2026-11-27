"""Exploratory, post hoc (26 Nov, after every LLM result was seen): every LLM arm against every required baseline. CPU.

Asked for after the AB2 and AB4 results: does any LLM arm beat the baselines RQ1 and E4 require? On ILI held-out AUC
(pooled seed-SD units, lower is better; common.by_seed, the AUC every other analysis uses), each LLM arm is compared
with each baseline by common.compare (permutation p, Welch interval). Nothing here is registered; no claim rests on
it. Holm over every comparison in the table is shown as a conservative reading of "beats". The registered tests
are E3's and E4's (scripts/replay.py) and AB-P and AB-G (llm_rtx.py, llm_registered.py); this file adds none.

Comparability: every arm and baseline is scored on the same table and held-out seeds, but the LLM arms ran on
different cards (A10/A10G on Modal, RTX 4080, Apple M4 Pro), with 10 or 30 seeds, and the GRPO arms are
transfers from ETTh1 while the CPU baselines search ILI directly. ETTh1 is in-sample (A18) and is left out.

    .venv/bin/python scripts/ablations/llm_vs_baselines.py   # writes results/ablations/LLM-vs-baselines.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, apply_holm, by_seed, compare, load_table, mean_ci, provenance, read_runs, write_json  # noqa: E402
from llm_rtx import PROMPTED, RTX_REF, TRANSFER, TRANSFER_METHOD, RtxSource  # noqa: E402
from llm_tests import runs_of  # noqa: E402

E3 = ROOT / "results/search/E3/ILI_T24"
E4 = ROOT / "results/search/E4/ILI_T24"
# RQ1 (docs/PLAN.md, E3) and E4's transfer controls (docs/prereg_E4.md), in that order.
BASELINES = {"random": (E3 / "cpu_runs.jsonl", "random"), "regularized evolution": (E3 / "cpu_runs.jsonl", "re"),
             "TPE": (E3 / "cpu_runs.jsonl", "tpe"), "ASHA": (E3 / "cpu_runs.jsonl", "asha"),
             "random + referee": (E3 / "cpu_runs.jsonl", "referee"), "PG controller": (E3 / "cpu_runs.jsonl", "pg"),
             "prompted Qwen 1.5B (E3, A10)": (E3 / "qwen_prompted_runs.jsonl", "qwen_prompted"),
             "warm-started TPE from ETTh1 (E4)": (E4 / "transfer_from_ETTh1_runs.jsonl", "tpe_warm_from_ETTh1"),
             "portfolio from ETTh1 (E4)": (E4 / "transfer_from_ETTh1_runs.jsonl", "portfolio_from_ETTh1")}


def values(runs: list[dict], method: str, table) -> list[float]:
    return [v for _, v in sorted(by_seed(runs, method, table).items())]


def arms(table) -> dict[str, list[float]]:
    """Every LLM arm with runs: RTX (AB2, AB4), Mac (AB3), E4 (Modal)."""
    out = {}
    src = RtxSource(RTX_REF)
    labels = {"ref": "1.5B", "0.5B": "0.5B", "3B": "3B", "anon": "1.5B anon", "repair": "1.5B repair"}
    for arm, (item, method) in PROMPTED.items():
        out[f"prompted Qwen {labels.get(arm, arm)} (RTX)"] = values(src.runs(item, "ILI_T24"), method, table)
    for arm, item in TRANSFER.items():
        out[f"GRPO league from ETTh1, {item.removeprefix('ABL_G_').removesuffix('_local')} (RTX)"] = values(src.runs(item, "ILI_T24"), TRANSFER_METHOD, table)
    mac = OUT / "mps"
    for name, method in (("ref", "qwen_prompted_mps"), ("anon", "qwen_prompted_anon_mps"), ("repair", "qwen_prompted_repair_mps")):
        out[f"prompted Qwen 1.5B {name} (Mac)"] = values(runs_of(mac / f"ABL_P_{name}_mps" / "ILI_T24"), method, table)
    steps = runs_of(mac / "ABL_G_steps_mps" / "ILI_T24")
    for step in (30, 60, 120):
        out[f"GRPO league from ETTh1, {step} steps (Mac)"] = values(steps, f"grpo_league_from_ETTh1_steps{step}_mps", table)
    for method in ("grpo_league_from_ETTh1", "grpo_none_from_ETTh1"):
        out[f"{method} (E4, A10/A10G)"] = values(read_runs(E4 / f"{method}_runs.jsonl"), method, table)
    return {k: v for k, v in out.items() if len(v) > 1}


def main() -> int:
    from common import code_hash

    if code_hash().endswith("-dirty"):
        raise SystemExit("scripts/ or leaguetst/ has uncommitted changes; commit before writing results")
    table, _ = load_table("ILI")
    base = {name: values(read_runs(path), method, table) for name, (path, method) in BASELINES.items()}
    llm = arms(table)
    tests = {f"{a} vs {b}": compare(llm[a], base[b], 1) for a in llm for b in base}
    held = apply_holm(tests, len(tests))
    rows = []
    for a in llm:
        for b in base:
            t = held[f"{a} vs {b}"]
            rows.append({"arm": a, "baseline": b, **t, "beats_interval": t["high"] < 0, "beats_holm": bool(t["reject"] and t["diff"] < 0),
                         "worse_holm": bool(t["reject"] and t["diff"] > 0)})
    result = {"exploratory": "post hoc, asked for on 26 Nov after every LLM result was seen; no claim rests on it",
              "provenance": provenance(__file__, [str(p.relative_to(ROOT)) for p, _ in BASELINES.values()]),
              "baselines": {k: mean_ci(v) for k, v in base.items()}, "baseline_order": list(BASELINES), "arms": {k: mean_ci(v) for k, v in llm.items()},
              "comparisons": len(rows), "rows": rows,
              "any_arm_beats_any_baseline_holm": any(r["beats_holm"] for r in rows),
              "best_arm": min(llm, key=lambda k: sum(llm[k]) / len(llm[k]))}
    write_json(OUT / "LLM-vs-baselines.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
