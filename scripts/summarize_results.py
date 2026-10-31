"""Aggregate run records into a table of mean and standard deviation over seeds.

Example:
    python scripts/summarize_results.py --results-dir results/preliminary

Several directories can be given; their records are pooled and the table is
written next to the first one unless --out names another file.

The script refuses to average what should not be averaged: a run that
appears twice, runs of one setting made with different configs, a record
with missing fields, or a run flagged as diverged. Each error names the
files involved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaguetst.experiment import normalized_training  # noqa: E402
from leaguetst.published import PUBLISHED  # noqa: E402

REQUIRED = ("model", "dataset", "horizon", "seed", "n_params", "input_len", "model_config", "training", "fit", "test")
REQUIRED_TEST = ("mse", "mae")
# Fields that must agree across the seeds of one table row.
SHARED = ("input_len", "model_config", "training", "n_params")


class RecordError(ValueError):
    """A record that cannot be summarised, with the file that holds it."""


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_records(results_dirs: list[Path]) -> list[dict]:
    """Read and check every run record under the given directories.

    Summary files are skipped. Each record remembers its path in ``_path``
    so that later errors can name it.
    """
    records = []
    seen: dict[tuple, Path] = {}
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.json")):
            if path.name.startswith("summary"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except json.JSONDecodeError as error:
                raise RecordError(f"{path}: not valid JSON ({error})") from error
            if not isinstance(record, dict):
                raise RecordError(f"{path}: not a run record")
            missing = [k for k in REQUIRED if k not in record]
            missing += [f"test.{k}" for k in REQUIRED_TEST if k not in record.get("test", {})]
            if missing:
                raise RecordError(f"{path}: missing {', '.join(missing)}")
            metrics = [("test", k) for k in ("mse", "mae", "mse_drop_last", "mae_drop_last")] + [("val", "mse")]
            bad = [f"{split}.{k}" for split, k in metrics if not _finite(record.get(split, {}).get(k, 0.0))]
            if bad:
                raise RecordError(f"{path}: non-finite {', '.join(bad)}")
            if record["fit"].get("diverged"):
                raise RecordError(f"{path}: the run diverged; rerun it or move it out of the results directory")
            identity = (record["model"], record["dataset"], record["horizon"], record["seed"], record.get("tag", ""))
            if identity in seen:
                raise RecordError(f"the same run appears twice: {seen[identity]} and {path}")
            seen[identity] = path
            record["_path"] = path
            records.append(record)
    return records


def check_group(runs: list[dict]) -> None:
    """Raise if the seeds of one table row were made with different settings."""
    for field in SHARED:
        read = normalized_training if field == "training" else (lambda value: value)
        values = {json.dumps(read(r[field]), sort_keys=True) for r in runs}
        if len(values) > 1:
            files = ", ".join(str(r["_path"]) for r in runs)
            raise RecordError(f"{field} differs between runs of one setting: {files}")


def spread(values: list[float]) -> str:
    """Format mean and sample standard deviation over seeds."""
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def summarize(records: list[dict]) -> tuple[list[str], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["model"], record["dataset"], record["horizon"], record.get("tag", ""))
        groups[key].append(record)

    lines = [
        "| model | dataset | T | tag | seeds | params | epochs | test MSE | test MAE "
        "| MSE drop-last | MAE drop-last | published MSE / MAE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for key in sorted(groups):
        model, dataset, horizon, tag = key
        runs = groups[key]
        check_group(runs)
        mses = [r["test"]["mse"] for r in runs]
        maes = [r["test"]["mae"] for r in runs]
        mses_drop = [r["test"]["mse_drop_last"] for r in runs if "mse_drop_last" in r["test"]]
        maes_drop = [r["test"]["mae_drop_last"] for r in runs if "mae_drop_last" in r["test"]]
        epochs = [r["fit"]["epochs_run"] for r in runs]
        published = PUBLISHED.get((model, dataset, horizon))
        published_text = f"{published[0]:.3f} / {published[1]:.3f}" if published else ""
        lines.append(
            f"| {model} | {dataset} | {horizon} | {tag} | {len(runs)} | {runs[0]['n_params']} "
            f"| {mean(epochs):.0f} | {spread(mses)} | {spread(maes)} "
            f"| {spread(mses_drop) if mses_drop else ''} | {spread(maes_drop) if maes_drop else ''} "
            f"| {published_text} |"
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                "tag": tag,
                "seeds": len(runs),
                "n_params": runs[0]["n_params"],
                "mean_epochs": mean(epochs),
                "test_mse_mean": mean(mses),
                "test_mse_std": stdev(mses) if len(mses) > 1 else 0.0,
                "test_mae_mean": mean(maes),
                "test_mae_std": stdev(maes) if len(maes) > 1 else 0.0,
                "test_mse_drop_last_mean": mean(mses_drop) if mses_drop else None,
                "test_mae_drop_last_mean": mean(maes_drop) if maes_drop else None,
                "published_mse": published[0] if published else None,
                "published_mae": published[1] if published else None,
            }
        )
    return lines, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", nargs="+", default=["results/preliminary"])
    parser.add_argument("--out", default=None, help="markdown file; defaults to summary.md inside the first results-dir")
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dir]
    try:
        records = load_records(results_dirs)
        if not records:
            print(f"no run records in {', '.join(str(d) for d in results_dirs)}")
            return 1
        lines, rows = summarize(records)
    except RecordError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    table = "\n".join(lines)
    print(table)

    out = Path(args.out) if args.out else results_dirs[0] / "summary.md"
    out.write_text(table + "\n", encoding="utf-8")
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
