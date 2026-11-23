"""Regenerate docs/REPORT_DRAFT.md from results/ledger.jsonl.

Every number in the draft comes from the ledger through this script, so a
table in the report can always be rebuilt and checked.

Example:
    python scripts/report.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402

OUT = ROOT / "docs" / "REPORT_DRAFT.md"


def table_section(rows: list[dict]) -> list[str]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["item"] in ("E1", "E3b", "PICK"):
            groups[(row["item"], row["dataset"], row["horizon"], row["seed"])].append(row)
    lines = [
        "## Tabular benchmark",
        "",
        "| item | dataset | T | seed | configs | best val MSE | median val MSE | mean epochs run | mean best epoch | GPU hours |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(groups):
        group = groups[key]
        vals = [r["best_val_mse"] for r in group]
        hours = sum(r.get("wall_seconds") or 0 for r in group) / 3600
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(group)} | {min(vals):.4f} | {median(vals):.4f} | "
            f"{mean(r['epochs_run'] for r in group):.1f} | {mean(r['best_epoch'] for r in group):.1f} | {hours:.2f} |"
        )
    return lines


def seed_section(rows: list[dict]) -> list[str]:
    by_config: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_config[(row["item"], row["dataset"], row["horizon"], row["config_id"])][row["seed"]] = row["best_val_mse"]
    lines = ["", "## Seed spread of table entries", "",
             "| item | dataset | T | configs with 3 seeds | median seed SD | median SD / mean |", "|---|---|---|---|---|---|"]
    summary: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for (item, dataset, horizon, _), seeds in by_config.items():
        if len(seeds) >= 3:
            values = list(seeds.values())
            summary[(item, dataset, horizon)].append((stdev(values), stdev(values) / mean(values)))
    for key in sorted(summary):
        sds = summary[key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(sds)} | {median(s for s, _ in sds):.4f} | {median(r for _, r in sds):.3f} |")
    return lines


def arms_section(rows: list[dict]) -> list[str]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row["item"] == "E6":
            groups[(row["dataset"], int(row["horizon"]), row["config_id"])].append(row["best_val_mse"])
    if not groups:
        return []
    lines = ["", "## E6: final training, validation MSE by arm (test split not opened)", "",
             "| dataset | T | arm | seeds | val MSE mean | SD |", "|---|---|---|---|---|---|"]
    for (dataset, horizon, arm), values in sorted(groups.items()):
        sd = stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {dataset} | {horizon} | {arm} | {len(values)} | {mean(values):.4f} | {sd:.4f} |")
    return lines


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def resolution_section() -> list[str]:
    stats = load(ROOT / "results" / "table_stats.json")
    if not stats:
        return []
    lines = ["", "## Resolution of the tables", "",
             "| dataset | configs | pooled seed SD | within 1 SD of best | within 2 SD | one-seed winner's rank on other seeds |",
             "|---|---|---|---|---|---|"]
    for name, t in sorted(stats["tables"].items()):
        ranks = ", ".join(str(r) for r in t.get("one_seed_winner_rank_on_others", [])) or "n/a"
        lines.append(f"| {name} | {t['configs']} | {t['pooled_sd']:.4f} | {t['within_1sd']} | {t['within_2sd']} | {ranks} |")
    between = stats.get("between")
    if between:
        lines += ["", f"Spearman between the ETTh1 and ILI rankings over {between['configs']} configs: "
                      f"{between['spearman']:.3f}; top-100 overlap {between['top_k_overlap']['100']}."]
    return lines


def fidelity_section() -> list[str]:
    lines = []
    for path in sorted((ROOT / "results" / "fidelity").glob("*.json")):
        f = load(path)
        if not lines:
            lines = ["", "## E2: screening epoch and referee", "",
                     "| setting | k | tau epoch-k vs final | tau referee vs final | referee kept | K1 |", "|---|---|---|---|---|---|"]
        for k, v in sorted(f["k"].items()):
            lines.append(f"| {f['dataset']} T={f['horizon']} | {k}{' (chosen)' if int(k) == f['chosen_k'] else ''} | "
                         f"{v['tau_epoch_k_vs_final']:.3f} | {v['tau_referee_vs_final']:.3f} | "
                         f"{'yes' if f['referee_beats_epoch_k'] else 'no'} | {'pass' if f['k1_referee_tau_at_least_0_5'] else 'fail'} |")
    return lines


def e3b_section() -> list[str]:
    lines = []
    for path in sorted((ROOT / "results" / "e3b").glob("*_T*.json")):
        e = load(path)
        if not lines:
            lines = ["", "## E3b: table validity (live retrain, seed 3001)", "",
                     "| setting | mean signed gap | 95% interval | median abs gap (SD) | threshold | Spearman all / top 20 | K7 |",
                     "|---|---|---|---|---|---|---|"]
        lines.append(f"| {e['dataset']} T={e['horizon']} | {e['mean_signed_gap']:+.4f} | [{e['gap_ci95'][0]:+.4f}, {e['gap_ci95'][1]:+.4f}] | "
                     f"{e['median_abs_gap_sd']:.2f} | {e['threshold_sd']} | {e['spearman_all']:.2f} / {e['spearman_top20']:.2f} | "
                     f"{'pass' if e['k7_pass'] else 'fail'} |")
    return lines


def replay_sections() -> list[str]:
    lines = []
    for path in sorted((ROOT / "results" / "search").glob("*/*/summary.md")):
        lines += ["", f"## {path.parent.parent.name} {path.parent.name}", ""]
        lines += path.read_text().splitlines()[2:]
    return lines


def anova_section() -> list[str]:
    lines = []
    for path in sorted((ROOT / "results" / "anova").glob("*.json")):
        a = load(path)
        if not lines:
            lines = ["", "## E8: where the variance is", "",
                     "| setting | scale | seeds | architecture | training setting | interaction | seed noise |", "|---|---|---|---|---|---|---|"]
        share = a["share"]
        fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"
        lines.append(f"| {a['dataset']} T={a['horizon']} | {'log' if a['log_scale'] else 'raw'} | {a['seeds_used']} | "
                     f"{fmt(share['architecture'])} | {fmt(share['training_setting'])} | "
                     f"{fmt(share['interaction'] if share['interaction'] is not None else share['interaction_plus_noise'])} | "
                     f"{fmt(share['seed_noise'])} |")
    return lines


def main() -> int:
    rows = list(read_rows(ROOT / "results" / "ledger.jsonl"))
    lines = ["# Report draft (generated)", "", "Generated by scripts/report.py from results/ledger.jsonl and the result files the",
             "analysis scripts derive from it. Do not edit by hand.", ""]
    if not rows:
        lines.append("The ledger is empty.")
    else:
        lines += table_section(rows) + seed_section(rows)
        lines += resolution_section() + fidelity_section() + e3b_section() + replay_sections() + anova_section()
        lines += arms_section(rows)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} from {len(rows)} ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
