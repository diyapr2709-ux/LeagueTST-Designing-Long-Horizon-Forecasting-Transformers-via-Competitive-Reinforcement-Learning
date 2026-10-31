"""Re-evaluate saved runs from their checkpoints and write detailed statistics.

For every run record the script rebuilds the model, loads the saved weights,
scores the validation and test windows again on the CPU with an evaluator
that shares no code with the training loop, and compares the result with
the record. It then writes, per setting: the seed spread, a block-bootstrap
interval over test windows, the error by forecast step and by variable, two
naive baselines, the paired PatchTST minus DLinear difference, and the
comparison with the published numbers under the matching protocol.

Example:
    python scripts/revalidate.py --results-dir results/preliminary results/baselines
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaguetst.analysis import (  # noqa: E402
    SEASON_LENGTH,
    block_bootstrap_ci,
    collect_forecasts,
    error_profile,
    naive_forecasts,
    window_targets,
)
from leaguetst.data import build_splits, load_benchmark_frame  # noqa: E402
from leaguetst.experiment import build_model  # noqa: E402
from leaguetst.models import count_parameters  # noqa: E402
from leaguetst.published import PUBLISHED, PUBLISHED_SEED  # noqa: E402
from leaguetst.utils import load_state_dict, sha256_file  # noqa: E402

REQUIRED_KEYS = ("model", "dataset", "horizon", "input_len", "seed", "model_config", "training", "n_params", "fit", "val", "test")
COMPARED = ("mse", "mae", "mse_drop_last", "mae_drop_last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", nargs="+", default=["results/preliminary"])
    parser.add_argument("--checkpoint-dir", default=None, help="look here before the path stored in each record")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/revalidation")
    parser.add_argument("--label", default="", help="suffix for the output file names")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance against the record")
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args()


def run_name(record: dict[str, Any]) -> str:
    base = f"{record['model']}_{record['dataset']}_h{record['horizon']}_s{record['seed']}"
    return f"{base}_{record['tag']}" if record.get("tag") else base


def load_records(results_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.json")):
            if path.name.startswith("summary"):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            missing = [k for k in REQUIRED_KEYS if k not in record]
            if missing:
                raise ValueError(f"{path}: missing keys {missing}")
            records.append((path, record))
    return records


def find_checkpoint(record: dict[str, Any], checkpoint_dir: str | None) -> Path:
    candidates = []
    if checkpoint_dir:
        candidates.append(Path(checkpoint_dir) / f"{run_name(record)}.pt")
    if record.get("checkpoint"):
        candidates.append(Path(record["checkpoint"]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no checkpoint for {run_name(record)} among {[str(c) for c in candidates]}")


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def spread(values: list[float]) -> tuple[float, float]:
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def pm(pair: tuple[float, float], digits: int = 4) -> str:
    return f"{pair[0]:.{digits}f} ± {pair[1]:.{digits}f}"


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""

    records = load_records([Path(d) for d in args.results_dir])
    if not records:
        print("no run records found")
        return 1

    split_cache: dict[tuple[str, int, int], dict] = {}
    columns: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    per_window: dict[tuple, dict[int, np.ndarray]] = defaultdict(dict)
    steps_by_setting: dict[tuple, list[list[float]]] = defaultdict(list)
    vars_by_setting: dict[tuple, list[list[float]]] = defaultdict(list)
    seen_runs: dict[str, Path] = {}

    for path, record in records:
        name = run_name(record)
        if name in seen_runs:
            raise ValueError(f"{name} appears twice: {seen_runs[name]} and {path}")
        seen_runs[name] = path
        dataset, horizon, input_len = record["dataset"], int(record["horizon"]), int(record["input_len"])
        key = (dataset, input_len, horizon)
        if key not in split_cache:
            split_cache[key], _ = build_splits(args.data_dir, dataset, input_len, horizon)
        if dataset not in columns:
            columns[dataset] = list(load_benchmark_frame(args.data_dir, dataset).columns)
        splits = split_cache[key]

        checkpoint = find_checkpoint(record, args.checkpoint_dir)
        digest = sha256_file(checkpoint)
        recorded_digest = record.get("checkpoint_sha256")
        state = load_state_dict(checkpoint, expected_sha256=recorded_digest)
        model = build_model(record["model"], record["model_config"], splits["test"].n_vars, input_len, horizon)
        model.load_state_dict(state, strict=True)
        batch = int(record["training"]["batch_size"])

        val_forecasts, val_targets = collect_forecasts(model, splits["val"])
        val = error_profile(val_forecasts, val_targets, batch)
        test_forecasts, test_targets = collect_forecasts(model, splits["test"])
        if not np.array_equal(test_targets, window_targets(splits["test"])):
            raise AssertionError(f"{name}: loader targets differ from direct slicing")
        test = error_profile(test_forecasts, test_targets, batch)

        problems = []
        if count_parameters(model) != record["n_params"]:
            problems.append("parameter count")
        if val["n_windows"] != record["val"]["n_samples"] or test["n_windows"] != record["test"]["n_samples"]:
            problems.append("window count")
        if "n_samples_drop_last" in record["test"] and test["n_keep"] != record["test"]["n_samples_drop_last"]:
            problems.append("drop-last window count")
        gaps = {"val mse": relative_gap(val["mse"], record["val"]["mse"])}
        for metric in COMPARED:
            if metric in record["test"]:
                gaps[f"test {metric}"] = relative_gap(test[metric], record["test"][metric])
        # NaN compares false with everything, so non-finite values are caught first.
        non_finite = [k for k, v in gaps.items() if not math.isfinite(v)]
        if non_finite:
            problems.append(f"non-finite {', '.join(non_finite)}")
            gaps = {k: (math.inf if k in non_finite else v) for k, v in gaps.items()}
        worst = max(gaps, key=gaps.get)
        if not non_finite and gaps[worst] > args.rtol:
            problems.append(f"{worst} off by {gaps[worst]:.1e}")
        if recorded_digest and recorded_digest != digest:
            problems.append("checkpoint digest")

        setting = (record["model"], dataset, horizon, record.get("tag", ""))
        per_window[setting][int(record["seed"])] = test["per_window_mse"]
        steps_by_setting[setting].append(test["per_step_mse"])
        vars_by_setting[setting].append(test["per_variable_mse"])
        fit = record["fit"]
        rows.append(
            {
                "run": name,
                "model": record["model"],
                "dataset": dataset,
                "horizon": horizon,
                "seed": int(record["seed"]),
                "tag": record.get("tag", ""),
                "device": record.get("environment", {}).get("device", ""),
                "n_params": record["n_params"],
                "best_epoch": fit["best_epoch"],
                "epochs_run": fit["epochs_run"],
                "train_seconds": fit["train_seconds"],
                "seconds_per_epoch": fit["train_seconds"] / max(fit["epochs_run"], 1),
                "recorded": {"val_mse": record["val"]["mse"], **{m: record["test"].get(m) for m in COMPARED}},
                "recomputed": {"val_mse": val["mse"], **{m: test[m] for m in COMPARED}},
                "max_relative_gap": gaps[worst],
                "worst_metric": worst,
                "mse_dropped": test["mse_dropped"],
                "n_windows": test["n_windows"],
                "n_keep": test["n_keep"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "status": "FAIL: " + ", ".join(problems) if problems else "pass",
            }
        )
        print(f"{rows[-1]['status']:6s} {name:30s} max rel gap {gaps[worst]:.1e} ({worst})", flush=True)

    naive: dict[tuple, dict[str, float]] = {}
    for (dataset, input_len, horizon), splits in split_cache.items():
        targets = window_targets(splits["test"])
        last = error_profile(naive_forecasts(splits["test"], "last"), targets)
        seasonal = error_profile(naive_forecasts(splits["test"], "seasonal", SEASON_LENGTH[dataset]), targets)
        naive[(dataset, horizon)] = {"last": last["mse"], "seasonal": seasonal["mse"], "last_mae": last["mae"], "seasonal_mae": seasonal["mae"]}

    settings = []
    by_setting: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_setting[(row["model"], row["dataset"], row["horizon"], row["tag"])].append(row)
    for setting in sorted(by_setting):
        model, dataset, horizon, tag = setting
        group = sorted(by_setting[setting], key=lambda r: r["seed"])
        seed_mean_window = np.mean(np.stack([per_window[setting][r["seed"]] for r in group]), axis=0)
        low, high = block_bootstrap_ci(seed_mean_window, block=horizon, n_resamples=args.resamples)
        exact = spread([r["recomputed"]["mse"] for r in group])
        drop = spread([r["recomputed"]["mse_drop_last"] for r in group])
        published = PUBLISHED.get((model, dataset, horizon))
        matched = [r for r in group if r["seed"] == PUBLISHED_SEED]
        step_curve = np.mean(np.array(steps_by_setting[setting]), axis=0)
        marks = sorted({0, horizon // 4 - 1, horizon // 2 - 1, 3 * horizon // 4 - 1, horizon - 1})
        dropped = [r["mse_dropped"] for r in group if not math.isnan(r["mse_dropped"])]
        settings.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                "tag": tag,
                "seeds": [r["seed"] for r in group],
                "n_params": group[0]["n_params"],
                "val_mse": spread([r["recomputed"]["val_mse"] for r in group]),
                "test_mse": exact,
                "test_mae": spread([r["recomputed"]["mae"] for r in group]),
                "test_mse_ci95": (low, high),
                "test_mse_drop_last": drop,
                "test_mae_drop_last": spread([r["recomputed"]["mae_drop_last"] for r in group]),
                "seed2021_mse_drop_last": matched[0]["recomputed"]["mse_drop_last"] if matched else None,
                "published_mse": published[0] if published else None,
                "published_mae": published[1] if published else None,
                "gap_mean_vs_published_pct": 100 * (drop[0] - published[0]) / published[0] if published else None,
                "gap_seed2021_vs_published_pct": (
                    100 * (matched[0]["recomputed"]["mse_drop_last"] - published[0]) / published[0] if (published and matched) else None
                ),
                "published_within_one_std": abs(drop[0] - published[0]) <= drop[1] if published else None,
                "mse_dropped_windows": mean(dropped) if dropped else None,
                "mse_kept_windows": drop[0],
                "step_marks": [m + 1 for m in marks],
                "per_step_mse": [float(step_curve[m]) for m in marks],
                "per_variable_mse": np.mean(np.array(vars_by_setting[setting]), axis=0).tolist(),
                "naive_last_mse": naive[(dataset, horizon)]["last"],
                "naive_seasonal_mse": naive[(dataset, horizon)]["seasonal"],
                "best_epochs": [r["best_epoch"] for r in group],
                "epochs_run": [r["epochs_run"] for r in group],
                "seconds_per_epoch": spread([r["seconds_per_epoch"] for r in group]),
                "devices": sorted({r["device"] for r in group}),
            }
        )

    paired = []
    for dataset, horizon, tag in sorted({(s[1], s[2], s[3]) for s in per_window}):
        a, b = ("patchtst", dataset, horizon, tag), ("dlinear", dataset, horizon, tag)
        if a not in per_window or b not in per_window:
            continue
        common = sorted(set(per_window[a]) & set(per_window[b]))
        diff = np.mean(np.stack([per_window[a][s] - per_window[b][s] for s in common]), axis=0)
        low, high = block_bootstrap_ci(diff, block=horizon, n_resamples=args.resamples)
        paired.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "tag": tag,
                "seeds": common,
                "mean_difference": float(diff.mean()),
                "ci95": (low, high),
                "share_patchtst_better": float((diff < 0).mean()),
            }
        )

    np.savez_compressed(
        out_dir / f"per_window_test_mse{suffix}.npz",
        **{f"{m}_{d}_h{h}{'_' + t if t else ''}_s{s}": v for (m, d, h, t), seeds in per_window.items() for s, v in seeds.items()},
    )
    report = {"runs": rows, "settings": settings, "paired": paired, "naive": {f"{d}_h{h}": v for (d, h), v in naive.items()}, "columns": columns}
    with open(out_dir / f"revalidation{suffix}.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    (out_dir / f"report{suffix}.md").write_text(render(report, args), encoding="utf-8")

    failures = [r for r in rows if r["status"] != "pass"]
    print(f"\n{len(rows) - len(failures)} of {len(rows)} runs agree with their records within rtol {args.rtol:g}")
    return 1 if failures else 0


def render(report: dict[str, Any], args: argparse.Namespace) -> str:
    out = ["# Revalidation report", ""]
    out += [
        f"Sources: {', '.join(args.results_dir)}. Every checkpoint was reloaded and scored again on the CPU",
        "with a NumPy evaluator that shares no code with the training loop. Errors are on the standardised",
        "scale used by the published tables. Exact means cover every test window; drop-last means keep the",
        "first floor(n / batch) * batch windows, the protocol behind the published numbers.",
        "",
        "## 1. Agreement between records and recomputation",
        "",
        "| run | device | recorded test MSE | recomputed | max relative gap | checkpoint SHA-256 | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["runs"]:
        out.append(
            f"| {r['run']} | {r['device']} | {fmt(r['recorded']['mse'], 6)} | {fmt(r['recomputed']['mse'], 6)} "
            f"| {r['max_relative_gap']:.1e} | {r['checkpoint_sha256'][:12]} | {r['status']} |"
        )
    out += [
        "",
        "## 2. Test error per setting",
        "",
        "Mean ± sample standard deviation over seeds. The 95% interval is a circular block bootstrap over",
        "test windows of the seed-averaged error, with blocks one horizon long.",
        "",
        "| model | dataset | T | params | test MSE exact | 95% interval | test MAE exact | MSE drop-last | seed 2021 drop-last | published | gap of mean | gap of seed 2021 | published within 1 sd |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["settings"]:
        gap_mean = f"{s['gap_mean_vs_published_pct']:+.1f}%" if s["gap_mean_vs_published_pct"] is not None else ""
        gap_2021 = f"{s['gap_seed2021_vs_published_pct']:+.1f}%" if s["gap_seed2021_vs_published_pct"] is not None else ""
        within = {True: "yes", False: "no", None: ""}[s["published_within_one_std"]]
        out.append(
            f"| {s['model']}{'/' + s['tag'] if s['tag'] else ''} | {s['dataset']} | {s['horizon']} | {s['n_params']:,} | {pm(s['test_mse'])} "
            f"| {s['test_mse_ci95'][0]:.4f} to {s['test_mse_ci95'][1]:.4f} | {pm(s['test_mae'])} | {pm(s['test_mse_drop_last'])} "
            f"| {fmt(s['seed2021_mse_drop_last'])} | {fmt(s['published_mse'], 3)} | {gap_mean} | {gap_2021} | {within} |"
        )
    out += [
        "",
        "## 3. Learned models against naive forecasts (exact test MSE)",
        "",
        "Repeat-last copies the final input step over the horizon; seasonal repeats the last day (ETTh1) or",
        "the last 52 weeks (ILI).",
        "",
        "| model | dataset | T | model MSE | repeat-last MSE | seasonal MSE | model / best naive |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in report["settings"]:
        best = min(s["naive_last_mse"], s["naive_seasonal_mse"])
        out.append(
            f"| {s['model']} | {s['dataset']} | {s['horizon']} | {s['test_mse'][0]:.4f} | {s['naive_last_mse']:.4f} "
            f"| {s['naive_seasonal_mse']:.4f} | {s['test_mse'][0] / best:.2f} |"
        )
    out += ["", "## 4. Error by forecast step (exact test MSE, seed mean)", "", "| model | dataset | T | steps | MSE at those steps |", "|---|---|---|---|---|"]
    for s in report["settings"]:
        out.append(
            f"| {s['model']} | {s['dataset']} | {s['horizon']} | {', '.join(map(str, s['step_marks']))} "
            f"| {', '.join(f'{v:.4f}' for v in s['per_step_mse'])} |"
        )
    for dataset, names in report["columns"].items():
        subset = [s for s in report["settings"] if s["dataset"] == dataset]
        if not subset:
            continue
        out += ["", f"## 5. Error by variable on {dataset} (exact test MSE, seed mean)", "", "| model | T | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for s in subset:
            out.append(f"| {s['model']} | {s['horizon']} | " + " | ".join(f"{v:.3f}" for v in s["per_variable_mse"]) + " |")
    out += [
        "",
        "## 6. PatchTST/42 minus DLinear, paired over test windows",
        "",
        "Negative means PatchTST is better. The interval is the block bootstrap of the mean difference.",
        "",
        "| dataset | T | seeds | mean difference | 95% interval | windows where PatchTST is better |",
        "|---|---|---|---|---|---|",
    ]
    for p in report["paired"]:
        out.append(
            f"| {p['dataset']} | {p['horizon']} | {len(p['seeds'])} | {p['mean_difference']:+.4f} | {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f} "
            f"| {100 * p['share_patchtst_better']:.0f}% |"
        )
    out += [
        "",
        "## 7. Windows removed by the drop-last protocol",
        "",
        "| model | dataset | T | windows | kept | MSE of kept windows | MSE of dropped windows |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in report["settings"]:
        group = [r for r in report["runs"] if (r["model"], r["dataset"], r["horizon"], r["tag"]) == (s["model"], s["dataset"], s["horizon"], s["tag"])]
        out.append(
            f"| {s['model']} | {s['dataset']} | {s['horizon']} | {group[0]['n_windows']} | {group[0]['n_keep']} "
            f"| {s['mse_kept_windows']:.4f} | {fmt(s['mse_dropped_windows'])} |"
        )
    out += [
        "",
        "## 8. Training",
        "",
        "| model | dataset | T | devices | best epoch per seed | epochs run | seconds per epoch | validation MSE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["settings"]:
        out.append(
            f"| {s['model']} | {s['dataset']} | {s['horizon']} | {', '.join(s['devices'])} | {', '.join(map(str, s['best_epochs']))} "
            f"| {', '.join(map(str, s['epochs_run']))} | {pm(s['seconds_per_epoch'], 1)} | {pm(s['val_mse'])} |"
        )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
