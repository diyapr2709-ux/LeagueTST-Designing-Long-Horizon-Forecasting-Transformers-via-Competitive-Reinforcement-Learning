"""Exploratory, descriptive: anytime held-out regret on ILI, budget 1 to 100, mean per arm, as a plain SVG figure. CPU.

Added on 26 Nov, after every result was seen (results review R16). No test, no interval. Each value is
common.by_seed's held-out regret at budget b (full-run equivalents of 30 epochs), in pooled seed-SD units, the
quantity whose mean over b = 1 to 100 is the AUC every table uses; the curve is its mean over an arm's searches.
Runs: E3's CPU methods and the E4 arms from results/search/, the RTX arms (AB2) at llm_rtx.RTX_REF. The figure is
written without plotting libraries (none is installed).

    .venv/bin/python scripts/ablations/anytime.py   # writes results/ablations/anytime-ILI.json and docs/figures/anytime_regret_ILI.svg
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, by_seed, load_table, provenance, read_runs, write_json  # noqa: E402
from llm_rtx import PROMPTED, RTX_REF, TRANSFER, TRANSFER_METHOD, RtxSource  # noqa: E402

FIGURE = ROOT / "docs" / "figures" / "anytime_regret_ILI.svg"
BUDGETS = range(1, 101)
E3 = ROOT / "results/search/E3/ILI_T24"
E4 = ROOT / "results/search/E4/ILI_T24"

# (panel title, [(short label, full label, source, item or path, method)]); four series per panel, slots 1 to 4
PANELS = [
    ("CPU baselines (E3, 30 searches each)", [
        ("random", "random search (E3)", "file", E3 / "cpu_runs.jsonl", "random"),
        ("RE", "regularized evolution (E3)", "file", E3 / "cpu_runs.jsonl", "re"),
        ("TPE", "TPE (E3)", "file", E3 / "cpu_runs.jsonl", "tpe"),
        ("ASHA", "ASHA (E3)", "file", E3 / "cpu_runs.jsonl", "asha")]),
    ("LLM arms and a transfer control", [
        ("prompted 1.5B", "prompted Qwen 1.5B, RTX (ABL_P_ref_local, 30)", "rtx", PROMPTED["ref"][0], PROMPTED["ref"][1]),
        ("GRPO 60 steps", "GRPO league from ETTh1, 60 steps, RTX (ABL_G_ref_local, 10)", "rtx", TRANSFER["ref"], TRANSFER_METHOD),
        ("GRPO league E4", "GRPO league from ETTh1, E4 on A10/A10G (10)", "file", E4 / "grpo_league_from_ETTh1_runs.jsonl", "grpo_league_from_ETTh1"),
        ("warm TPE E4", "TPE warm-started from ETTh1, E4 (30)", "file", E4 / "transfer_from_ETTh1_runs.jsonl", "tpe_warm_from_ETTh1")]),
]
REFERENCE = "random"  # drawn as a grey dashed line in the second panel

LIGHT = {"surface": "#ffffff", "ink": "#191919", "ink2": "#666666", "grid": "#e7e7e7", "axis": "#c2c2c2", "ref": "#8a8a8a",
         "series": ["#257de0", "#f47230", "#00b894", "#f2a600"]}
DARK = {"surface": "#191919", "ink": "#efefef", "ink2": "#9c9c9c", "grid": "#303030", "axis": "#474747", "ref": "#8a8a8a",
        "series": ["#338ced", "#e76624", "#00a888", "#d28e05"]}


def curve(runs: list[dict], method: str, table) -> dict:
    """Mean held-out regret per budget over an arm's searches (common.by_seed at each budget)."""
    per_budget = [list(by_seed(runs, method, table, b).values()) for b in BUDGETS]
    return {"n": len(per_budget[0]), "mean": [float(np.mean(v)) for v in per_budget]}


def curves(table) -> list[dict]:
    src = RtxSource(RTX_REF)
    panels = []
    for title, arms in PANELS:
        series = []
        for short, full, kind, where, method in arms:
            runs = src.runs(where, "ILI_T24") if kind == "rtx" else read_runs(where)
            source = f"{RTX_REF[:7]}:results/ablations/rtx/{where}/ILI_T24" if kind == "rtx" else str(Path(where).relative_to(ROOT))
            series.append({"label": short, "name": full, "method": method, "source": source, **curve(runs, method, table)})
        panels.append({"title": title, "series": series})
    return panels


def svg(panels: list[dict]) -> str:
    """Two small multiples on one y scale; 2px lines, direct labels at the right end, a legend, hairline grid."""
    width, height, top, bottom, left, gap, label_w = 1060, 430, 78, 52, 56, 40, 140
    pw = (width - left - 2 * label_w - gap) / 2
    ph = height - top - bottom
    ymax = float(np.ceil(max(max(s["mean"]) for p in panels for s in p["series"])))
    x = lambda i, b: left + i * (pw + label_w + gap) + (b - 1) / 99 * pw
    y = lambda v: top + ph - v / ymax * ph
    css = lambda c: (f".surface{{fill:{c['surface']}}} .ink{{fill:{c['ink']}}} .ink2{{fill:{c['ink2']}}} .grid{{stroke:{c['grid']}}} .axis{{stroke:{c['axis']}}} "
                     f".ref{{stroke:{c['ref']}}} " + " ".join(f".s{k}{{stroke:{v}}} .f{k}{{fill:{v}}}" for k, v in enumerate(c["series"])))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           'aria-label="Mean held-out regret on ILI by budget, two panels">',
           "<style>text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:12px} .title{font-size:14px;font-weight:600} "
           f"line,path{{fill:none}} {css(LIGHT)} @media (prefers-color-scheme: dark){{{css(DARK)}}}</style>",
           f'<rect class="surface" x="0" y="0" width="{width}" height="{height}"/>',
           f'<text class="ink title" x="{left}" y="22">ILI held-out regret by budget (mean over searches, pooled seed-SD units; no interval)</text>']
    ticks = [t for t in range(0, int(ymax) + 1)]
    for i, panel in enumerate(panels):
        x0 = x(i, 1)
        out.append(f'<text class="ink" x="{x0}" y="44" font-weight="600">{escape(panel["title"])}</text>')
        lx = x0
        for k, s in enumerate(panel["series"]):  # legend row
            out.append(f'<line class="s{k}" x1="{lx}" y1="60" x2="{lx + 16}" y2="60" stroke-width="2"/>'
                       f'<text class="ink2" x="{lx + 20}" y="64">{escape(s["label"])}</text>')
            lx += 34 + 6.5 * len(s["label"])
        for t in ticks:
            out.append(f'<line class="grid" x1="{x0}" y1="{y(t):.1f}" x2="{x0 + pw:.1f}" y2="{y(t):.1f}" stroke-width="1"/>')
            out.append(f'<text class="ink2" x="{x0 - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        out.append(f'<line class="axis" x1="{x0}" y1="{y(0):.1f}" x2="{x0 + pw:.1f}" y2="{y(0):.1f}" stroke-width="1"/>')
        for b in (1, 25, 50, 75, 100):
            out.append(f'<text class="ink2" x="{x(i, b):.1f}" y="{y(0) + 18:.1f}" text-anchor="middle">{b}</text>')
        out.append(f'<text class="ink2" x="{x0 + pw / 2:.1f}" y="{height - 12}" text-anchor="middle">budget (full-run equivalents of 30 epochs)</text>')
        if i == 0:
            out.append(f'<text class="ink2" x="14" y="{top + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {top + ph / 2:.1f})">regret (seed SD)</text>')
        lines = list(panel["series"])
        if i == 1:
            ref = next(s for s in panels[0]["series"] if s["label"] == REFERENCE)
            path = " ".join(f"{'M' if b == 1 else 'L'}{x(i, b):.1f},{y(v):.2f}" for b, v in zip(BUDGETS, ref["mean"]))
            out.append(f'<path class="ref" d="{path}" stroke-width="1.5" stroke-dasharray="4 4"><title>{escape(ref["name"])} (reference)</title></path>')
            lines = lines + [{**ref, "label": "random (ref.)", "reference": True}]
        ends = sorted(((y(s["mean"][-1]), s) for s in lines), key=lambda e: e[0])
        placed = []
        for yy, s in ends:  # direct labels, at least 14 px apart
            yy = max(yy, placed[-1] + 14) if placed else yy
            placed.append(yy)
            k = next((j for j, t in enumerate(panel["series"]) if t is s or t["label"] == s["label"]), None)
            cls = "ink2" if s.get("reference") or k is None else "ink"
            out.append(f'<text class="{cls}" x="{x(i, 100) + 6:.1f}" y="{yy + 4:.1f}">{escape(s["label"])} {s["mean"][-1]:.2f}</text>')
        for k, s in enumerate(panel["series"]):
            path = " ".join(f"{'M' if b == 1 else 'L'}{x(i, b):.1f},{y(v):.2f}" for b, v in zip(BUDGETS, s["mean"]))
            out.append(f'<path class="s{k}" d="{path}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round">'
                       f'<title>{escape(s["name"])}: n {s["n"]}; budget 10 {s["mean"][9]:.2f}, 50 {s["mean"][49]:.2f}, 100 {s["mean"][99]:.2f}</title></path>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    from common import code_hash

    if code_hash().endswith("-dirty"):
        raise SystemExit("scripts/ or leaguetst/ has uncommitted changes; commit before writing results")
    table, _ = load_table("ILI")
    panels = curves(table)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.write_text(svg(panels))
    inputs = sorted({s["source"] for p in panels for s in p["series"] if not s["source"].startswith(RTX_REF[:7])})
    write_json(OUT / "anytime-ILI.json", {"descriptive": "added 26 Nov after every result was seen; mean held-out regret per budget, no interval, no test",
                                          "provenance": provenance(__file__, inputs), "budgets": list(BUDGETS), "panels": panels,
                                          "figure": str(FIGURE.relative_to(ROOT))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
