"""Write docs/wide_space.md: the wide design space, its feasible sets and its price.

Counts come from leaguetst.search.wide_space in closed form. The cost model
is fitted to the per-epoch timings of the E1 rows of the ledger on the A10G
and prices training every feasible config once at the table settings
(ETTh1 T=336, ILI T=24). Every number in the page comes from this script.

Example:
    python scripts/wide_space.py
    python scripts/wide_space.py --ledger ../main/results/ledger.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.models.config import ForecasterConfig  # noqa: E402
from leaguetst.models.wide import WideConfig  # noqa: E402
from leaguetst.search import cost_model as cm  # noqa: E402
from leaguetst.search.feasible import BUDGETS, INPUT_LEN, SETTINGS  # noqa: E402
from leaguetst.search.space import SEARCH_SPACE  # noqa: E402
from leaguetst.search.wide_space import (  # noqa: E402
    DATASETS,
    HEAD_CHOICES,
    LOOKBACKS,
    feasible_mask,
    grid_parameter_parts,
    parameter_parts,
    shape_grid,
    shared_mask,
    summarize,
    wide_feasible_table,
    wide_space_size,
)
from leaguetst.table import TABLE_EPOCHS, patchtst42_config, recipe  # noqa: E402

OUT = ROOT / "docs" / "wide_space.md"
TABLE_SETTING = {"ETTh1": 336, "ILI": 24}
# A quarter of the A10G's 24 GB, less the CUDA context of each worker.
MEMORY_PER_WORKER = 5e9


def n(value: float) -> str:
    return f"{int(round(value)):,}"


def flatten_default(dataset: str) -> np.ndarray:
    g = shape_grid(dataset)
    return (g["head_code"] == 0) & (g["input_len"] == INPUT_LEN[dataset])


def head_shares(dataset: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Flatten-head share of the parameters over the dataset's four horizons.

    Returns the (smallest, largest) share for PatchTST/42 and for today's
    2,368 configs.
    """
    reference = WideConfig.from_forecaster(ForecasterConfig(**patchtst42_config(dataset)), INPUT_LEN[dataset])
    mask = shared_mask(dataset) & (shape_grid(dataset)["head_code"] == 0)
    baseline, today = [], []
    for horizon in (t for ds, t in SETTINGS if ds == dataset):
        backbone, head = grid_parameter_parts(dataset, horizon)
        today.append(head[mask] / (backbone[mask] + head[mask]))
        backbone, head = parameter_parts(
            reference.patch_len, reference.stride, reference.n_layers, reference.d_model, reference.ff_ratio,
            False, 0, 0, reference.input_len, horizon,
        )
        baseline.append(float(head / (backbone + head)))
    joined = np.concatenate(today)
    return (min(baseline), max(baseline)), (float(joined.min()), float(joined.max()))


def fewest_patches(dataset: str) -> int:
    """Fewest patches of any of today's 2,368 configs at the dataset's default lookback."""
    g = shape_grid(dataset)
    mask = shared_mask(dataset) & (g["head_code"] == 0)
    return int(((g["input_len"] - g["patch_len"]) // g["stride"] + 2)[mask].min())


def smallest_backbone(width: int) -> int:
    """Fewest backbone weights of any shape of this width (one layer, smallest ff ratio)."""
    smallest = []
    for dataset in DATASETS:
        g = shape_grid(dataset)
        backbone, _ = grid_parameter_parts(dataset, TABLE_SETTING[dataset])
        smallest.append(int(backbone[g["d_model"] == width].min()))
    return min(smallest)


def space_section(table: dict[str, Any]) -> list[str]:
    shares = {ds: head_shares(ds) for ds in DATASETS}

    def pct(pair: tuple[float, float]) -> str:
        return f"{100 * pair[0]:.0f} to {100 * pair[1]:.0f}%"

    per_setting = table["per_setting"]
    sets = [
        ("ETTh1, all four horizons", table["per_dataset"]["ETTh1"], "ETTh1"),
        ("ILI, all four horizons", table["per_dataset"]["ILI"], "ILI"),
        ("Both datasets, default lookbacks, all eight settings", table["shared"], None),
    ]
    lines = [
        "## Why the current space is narrow",
        "",
        "PatchTST's flatten head has d_model x n_patches x T weights, seven times more with channel mixing. "
        f"Over the four horizons it holds {pct(shares['ETTh1'][0])} of PatchTST/42's parameters on ETTh1 and "
        f"{pct(shares['ILI'][0])} on ILI. The head grows with width, so under the PatchTST/42 cap at all eight "
        "settings only widths 16 and 32 and channel-independent configs fit, and the set is 74 architectures "
        "crossed with 32 training settings (docs/feasible_set.md). Within that set the head holds "
        f"{pct(shares['ETTh1'][1])} of the parameters on ETTh1 and {pct(shares['ILI'][1])} on ILI, where a "
        f"coarse patching of the 104-step lookback leaves as few as {fewest_patches('ILI')} patches.",
        "",
        "## The wide space",
        "",
        "`leaguetst/models/wide.py` adds three fields to the nine of `ForecasterConfig`:",
        "",
        "* `input_len`, the lookback: " + "; ".join(f"{ds} {', '.join(map(str, LOOKBACKS[ds]))}" for ds in DATASETS) + ".",
        "* `head_type`: `flatten` (PatchTST's head), `pooled` (mean over patches, then d_model -> T), "
        "`lowrank` (flatten -> r -> T) and `shared_linear` (one d_model -> T map per patch, then a learned "
        "mix over patches for each output step).",
        "* `head_rank`, the r of the low-rank head: " + ", ".join(str(r) for h, r in HEAD_CHOICES if h == "lowrank")
        + ", all below the shortest horizon (24), so the head is low rank at every setting.",
        "",
        "With the flatten head and today's fields, `WideTST` builds the same modules in the same order as "
        "`PatchTST`: same parameter count, same weights and same forecasts under the same seed "
        "(tests/test_wide_space.py). The grid crosses today's 30,720 configs with the "
        f"{len(HEAD_CHOICES)} head choices and the lookbacks: {n(wide_space_size('ETTh1'))} configs on ETTh1 and "
        f"{n(wide_space_size('ILI'))} on ILI. Parameter counts are closed form "
        "(`leaguetst.search.wide_space.parameter_parts`); the tests check them against models instantiated on "
        "the meta device for 600 random configs and every head at the largest shape, and against "
        "`patchtst_parameters` on all 30,720 configs at all eight settings.",
        "",
        "Feasibility uses the caps of `leaguetst.search.feasible.BUDGETS`. A config is feasible at a setting "
        "when its parameter count is at most the cap. \"All four horizons\" is the set a winner must fit to be "
        "retrained at every horizon of its dataset (E6). The last set keeps each dataset's default lookback "
        "(336, 104) and asks for all eight settings, the definition of today's 2,368; with the flatten head "
        "the wide grid reproduces every count of docs/feasible_set.md exactly.",
        "",
        "## Feasible counts",
        "",
        "An architecture is a shape: patch length, stride, layers, width, ff ratio, channel mode, head and "
        "lookback. Heads, dropout and instance norm remain training settings, 32 per architecture at every "
        "width. \"Today\" is the flatten head at the default lookback.",
        "",
        "| Set | Cap | Grid | Feasible | Architectures | Channel-mixing | Width >= 64 | Today |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset, horizon in SETTINGS:
        row = per_setting[f"{dataset}_T{horizon}"]
        wide = sum(v for k, v in row["by_width"].items() if int(k) >= 64)
        lines.append(
            f"| {dataset} T={horizon} | {n(row['budget'])} | {n(row['grid'])} | {n(row['configs'])} | "
            f"{n(row['architectures'])} | {n(row['by_channel']['mixing'])} | {n(wide)} | "
            f"{n(table['flatten_default'][f'{dataset}_T{horizon}'])} |"
        )
    for label, row, dataset in sets:
        wide = sum(v for k, v in row["by_width"].items() if int(k) >= 64)
        if dataset is None:
            today, grid = table["flatten_default"]["all_settings"], "-"
        else:
            today = summarize(dataset, feasible_mask(dataset) & flatten_default(dataset))["configs"]
            grid = n(row["grid"])
        lines.append(
            f"| {label} | - | {grid} | {n(row['configs'])} | {n(row['architectures'])} | "
            f"{n(row['by_channel']['mixing'])} | {n(wide)} | {n(today)} |"
        )

    lines += ["", "Feasible configs per head type:", "", "| Head | " + " | ".join(s[0] for s in sets) + " |",
              "|---|" + "---|" * len(sets)]
    for label in sets[0][1]["by_head"]:
        lines.append(f"| {label} | " + " | ".join(n(s[1]["by_head"][label]) for s in sets) + " |")

    widths = [str(w) for w in SEARCH_SPACE["d_model"]]
    lines += ["", "Feasible configs per width:", "", "| Set | " + " | ".join(widths) + " |", "|---|" + "---|" * len(widths)]
    for dataset, horizon in SETTINGS:
        row = per_setting[f"{dataset}_T{horizon}"]
        lines.append(f"| {dataset} T={horizon} | " + " | ".join(n(row["by_width"][w]) for w in widths) + " |")
    for label, row, _ in sets:
        lines.append(f"| {label} | " + " | ".join(n(row["by_width"][w]) for w in widths) + " |")

    lines += ["", "Feasible configs per lookback, all four horizons:", ""]
    for dataset in DATASETS:
        by = table["per_dataset"][dataset]["by_lookback"]
        lines.append(f"* {dataset}: " + ", ".join(f"L={k}: {n(v)}" for k, v in by.items()) + ".")

    etth1, ili, shared = (s[1] for s in sets)
    min_cap = min(BUDGETS[(ds, t)] for ds, t in SETTINGS if ds == "ETTh1")
    max_ili = max(BUDGETS[(ds, t)] for ds, t in SETTINGS if ds == "ILI")
    lines += [
        "",
        "## What the cap now allows",
        "",
        f"* Size. The four-horizon sets hold {n(etth1['configs'])} configs ({n(etth1['architectures'])} "
        f"architectures) on ETTh1 and {n(ili['configs'])} ({n(ili['architectures'])}) on ILI; the eight-setting "
        f"set holds {n(shared['configs'])} ({n(shared['architectures'])}), against 2,368 (74) today.",
        f"* Width 64 becomes feasible at every horizon through the cheaper heads: {n(etth1['by_width']['64'])} "
        f"configs on ETTh1 and {n(ili['by_width']['64'])} on ILI. Width 128 is feasible only at ETTh1 T=192 and "
        f"above, and width 256 only at ETTh1 T=720. That limit is the encoder, not the head: the smallest backbone "
        f"at width 128 (one layer, ff ratio 1) has {n(smallest_backbone(128))} weights, above the ETTh1 T=96 cap "
        f"({n(min_cap)}) and every ILI cap (at most {n(max_ili)}); at width 256 it has {n(smallest_backbone(256))}.",
        f"* Channel mixing becomes feasible: {n(etth1['by_channel']['mixing'])} of the ETTh1 four-horizon configs "
        f"and {n(ili['by_channel']['mixing'])} of ILI's, against none in today's set.",
        "* The flatten head stays in the space and keeps today's configs, so a search over the wide space "
        "can still return a PatchTST.",
    ]
    return lines


def quality_row(label: str, quality: dict[str, float]) -> str:
    return (
        f"| {label} | {n(quality['rows'])} | {quality['r2']:.3f} | {quality['r2_log']:.3f} | "
        f"{100 * quality['median_rel_error']:.1f}% | {100 * quality['p90_rel_error']:.1f}% | {quality['total_ratio']:.3f} |"
    )


def cost_section(timings: Sequence[cm.Timing], other_gpu: Sequence[cm.Timing], ledger: Path) -> list[str]:
    lines = ["", "## Cost model", ""]
    if len(timings) < 50:
        return lines + [f"Fewer than 50 E1 rows on the A10G in {ledger.name}; the cost model was not fitted."]
    x = np.stack([t.features for t in timings])
    y = np.array([t.epoch_seconds for t in timings])
    model = cm.fit_cost_model(x, y)
    coef = model.describe()
    datasets = np.array([t.dataset for t in timings])
    present = [ds for ds in DATASETS if np.any(datasets == ds)]
    counts = ", ".join(f"{ds} {n(np.sum(datasets == ds))}" for ds in DATASETS)
    lines += [
        f"Fitted to the steady epochs (2 onward) of {n(len(timings))} E1 ledger rows on the {cm.A10G} "
        f"({counts}), four worker processes per GPU. Each feature counts one epoch (leaguetst/search/cost_model.py):",
        "",
        "| Feature | Coefficient |",
        "|---|---|",
        f"| step (training steps + 1/3 of validation steps) | {1000 * coef['steps']:.3f} ms |",
        f"| step x encoder layer | {1000 * coef['layer_steps']:.3f} ms |",
        f"| matmul work, per TFLOP | {1000 * coef['gflops']:.3f} s |",
        f"| elementwise work, per 10^9 elements | {coef['gelems']:.4f} s |",
        "",
    ]
    if coef["gflops"] == 0:
        lines += [
            "The matmul coefficient is zero: on these rows a step costs a few hundred short kernel launches, "
            "and matmul time is too small a share of an epoch to be told apart from the elementwise term. "
            "The FLOP floor in the price table bounds the matmul cost from below without the fit.",
            "",
        ]
    else:
        lines += [
            f"The matmul coefficient implies {1 / (1000 * coef['gflops']):.2f} TFLOPS per worker with four "
            "workers sharing the GPU, far below its peak: at widths 16 and 32 the matrices are too small to "
            "keep it busy.",
            "",
        ]

    held = cm.held_out_predictions(timings)
    lines += [
        "Fit quality. Held-out rows are scored by a model fitted without them, and the rows of one config "
        "(its seeds) are always held out together:",
        "",
        "| Check | Rows scored | R^2 | R^2 (log) | Median rel. error | 90th pct rel. error | Predicted / measured total |",
        "|---|---|---|---|---|---|---|",
        quality_row("In sample", cm.fit_quality(model.predict(x), y)),
        quality_row("5-fold cross-validation, all rows", cm.fit_quality(held, y)),
    ]
    for ds in present:
        lines.append(quality_row(f"5-fold cross-validation, {ds} rows", cm.fit_quality(held[datasets == ds], y[datasets == ds])))
    splits = [
        ("Fit on width 16, scored on 32", lambda t: t.config.d_model == 16, lambda t: t.config.d_model == 32),
        ("Fit on 1 to 3 layers, scored on 4", lambda t: t.config.n_layers <= 3, lambda t: t.config.n_layers == 4),
        ("Fit on ILI, scored on ETTh1", lambda t: t.dataset == "ILI", lambda t: t.dataset == "ETTh1"),
        ("Fit on ETTh1, scored on ILI", lambda t: t.dataset == "ETTh1", lambda t: t.dataset == "ILI"),
    ]
    for label, train, test in splits:
        a = [t for t in timings if train(t)]
        b = [t for t in timings if test(t)]
        if len(a) >= 50 and b:
            lines.append(quality_row(label, cm.split_quality(a, b)))
    if other_gpu:
        predicted = model.predict(np.stack([t.features for t in other_gpu]))
        measured = np.array([t.epoch_seconds for t in other_gpu])
        lines.append(quality_row(f"A10G fit, scored on {other_gpu[0].gpu} rows", cm.fit_quality(predicted, measured)))
    lines += [
        "",
        "R^2 on a subset with little spread in epoch time is low even when the relative errors are small; the "
        "relative errors and the totals are the numbers that matter for a price. The cross-dataset rows show "
        "how far the fit carries: ILI epochs (16 windows per batch) do not predict ETTh1 epochs (128), which is "
        "why both datasets are in the fit.",
    ]

    factors = {ds: float(y[datasets == ds].sum() / held[datasets == ds].sum()) for ds in present}
    container = cm.container_seconds(timings) / 3600
    predicted_hours = sum(t.epochs_run * float(model.predict(t.features)) * factors[t.dataset] + t.overhead_seconds
                          for t in timings) / cm.WORKERS_PER_GPU / 3600
    lines += [
        "",
        f"Accounting check on the fitted rows: their wall time divided by four workers is {container:.2f} "
        f"GPU-hours; epochs run times the corrected predicted epoch time, plus each row's overhead, gives "
        f"{predicted_hours:.2f}.",
    ]
    return lines + price_section(model, timings, factors)


def price_section(model: cm.CostModel, timings: Sequence[cm.Timing], factors: dict[str, float]) -> list[str]:
    lines = [
        "",
        "## Price of enumerating the wide space",
        "",
        f"One seed per config at the table setting, on an A10G at ${cm.DOLLARS_PER_GPU_HOUR:.2f} per GPU-hour "
        f"(CPU and memory included), {cm.WORKERS_PER_GPU} workers per GPU. Each config is charged the mean "
        f"epochs run by its dataset's E1 rows (the {TABLE_EPOCHS}-epoch cap when there are none) plus the mean "
        "per-config overhead. The fit minimises relative error per row, which leaves the long ETTh1 epochs "
        "under-predicted on average, so each predicted epoch is multiplied by the dataset's measured over "
        "predicted total on held-out rows (the correction column). \"Over 5 GB\" counts the configs whose rough "
        "activation memory exceeds a quarter of the A10G's 24 GB less each worker's CUDA context; they need "
        "fewer workers per GPU but are not charged more, since a config that large keeps the GPU busy on its "
        f"own. The FLOP floor is the matmul work alone at the A10G's FP32 peak ({cm.A10G_FP32_TFLOPS} TFLOPS, "
        "TF32 off), a bound no fit can undercut.",
        "",
        "| Set | Setting | Configs | Correction | Mean epoch (s) | Epochs charged | GPU-hours | Dollars | "
        f"GPU-hours at {TABLE_EPOCHS} epochs | Dollars at {TABLE_EPOCHS} epochs | FLOP floor ($) | Over 5 GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    pooled_overhead = float(np.mean([t.overhead_seconds for t in timings]))
    for label, kind in (("All four horizons", "dataset"), ("Both datasets, all eight settings", "shared"), ("Today's 2,368", "today")):
        for dataset in DATASETS:
            horizon = TABLE_SETTING[dataset]
            rows = [t for t in timings if t.dataset == dataset]
            epochs = float(np.mean([t.epochs_run for t in rows])) if rows else float(TABLE_EPOCHS)
            overhead = float(np.mean([t.overhead_seconds for t in rows])) if rows else pooled_overhead
            scale = factors.get(dataset, 1.0)
            mask = {
                "dataset": feasible_mask(dataset),
                "shared": shared_mask(dataset),
                "today": shared_mask(dataset) & (shape_grid(dataset)["head_code"] == 0),
            }[kind]
            batch = int(recipe(dataset)["batch_size"])
            price = cm.price_enumeration(model, dataset, horizon, mask, batch, epochs, overhead, scale, MEMORY_PER_WORKER)
            capped = cm.price_enumeration(model, dataset, horizon, mask, batch, float(TABLE_EPOCHS), overhead, scale)
            source = f"{epochs:.1f}" if rows else f"{TABLE_EPOCHS} (cap)"
            lines.append(
                f"| {label} | {dataset} T={horizon} | {n(price['configs'])} | {scale:.3f} | "
                f"{price['mean_epoch_seconds']:.2f} | {source} | {price['gpu_hours']:,.1f} | {price['dollars']:,.0f} | "
                f"{capped['gpu_hours']:,.1f} | {capped['dollars']:,.0f} | {price['floor_dollars']:,.1f} | "
                f"{n(price['configs_over_memory'])} |"
            )
    lines += [
        "",
        "Prices are per seed; E1 trains ILI at three seeds. The fitted rows have widths 16 and 32, the flatten "
        "head, channel-independent inputs and the default lookback, so prices for wider, mixing or long-lookback "
        "configs extrapolate the fit. The per-TFLOP cost was measured on small matrices, which use the GPU "
        "least efficiently, so for widths 64 and 128 it is more likely to overstate than understate the time. "
        "The measured epoch times include contention between the four workers. The memory estimate counts "
        "every elementwise output of a batch at four bytes, more than autograd keeps, so it overstates the "
        "configs over 5 GB.",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    ledger = Path(args.ledger)

    table = wide_feasible_table()
    timings = cm.ledger_timings(ledger, item="E1", gpus=[cm.A10G])
    other = [t for t in cm.ledger_timings(ledger, item="E1") if t.gpu != cm.A10G]
    lines = [
        "# Wide design space (amendment A16, paper stage)",
        "",
        "Generated by scripts/wide_space.py from leaguetst.search.wide_space and the E1 timings in "
        "results/ledger.jsonl. Do not edit by hand.",
        "",
    ]
    lines += space_section(table) + cost_section(timings, other, ledger)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} from {len(timings)} A10G E1 rows and {len(other)} rows on other GPUs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
