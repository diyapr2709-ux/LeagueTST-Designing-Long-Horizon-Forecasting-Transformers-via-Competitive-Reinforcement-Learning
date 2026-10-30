"""Aggregate run records into a table of mean and standard deviation over seeds.

Example:
    python scripts/summarize_results.py --results-dir results/preliminary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PUBLISHED = {
    ("patchtst", "ETTh1", 96): (0.375, 0.399),
    ("patchtst", "ETTh1", 192): (0.414, 0.421),
    ("patchtst", "ETTh1", 336): (0.431, 0.436),
    ("patchtst", "ETTh1", 720): (0.449, 0.466),
    ("dlinear", "ETTh1", 96): (0.375, 0.399),
    ("dlinear", "ETTh1", 192): (0.405, 0.416),
    ("dlinear", "ETTh1", 336): (0.439, 0.443),
    ("dlinear", "ETTh1", 720): (0.472, 0.490),
    ("patchtst", "ILI", 24): (1.522, 0.814),
    ("patchtst", "ILI", 36): (1.430, 0.834),
    ("patchtst", "ILI", 48): (1.673, 0.854),
    ("patchtst", "ILI", 60): (1.529, 0.862),
    ("dlinear", "ILI", 24): (2.215, 1.081),
    ("dlinear", "ILI", 36): (1.963, 0.963),
    ("dlinear", "ILI", 48): (2.130, 1.024),
    ("dlinear", "ILI", 60): (2.368, 1.096),
}


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/preliminary")
    parser.add_argument("--out", default=None, help="markdown file; defaults to summary.md inside results-dir")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir)
    if not records:
        print(f"no run records in {results_dir}")
        return
    lines, rows = summarize(records)
    table = "\n".join(lines)
    print(table)

    out = Path(args.out) if args.out else results_dir / "summary.md"
    out.write_text(table + "\n", encoding="utf-8")
    with open(results_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
