"""LLM architect jobs on one local CUDA GPU (for example an RTX card), same plans as Modal.

The plan, the job order, the skip rules and the output layout are those of
scripts/modal_llm.py (leaguetst/architect/runner.py); only the executor
differs. Outputs go to <runs-root>/<item>/<name>/ and adapters to
<runs-root>/<item>/adapters/. A relaunch keeps finished adapters and deployed
seeds. Install the pinned packages first: pip install -r requirements-llm.txt.

Each GRPO process peaks near 7.6 GB, so --per-gpu 2 needs a 24 GB card
(RTX 3090, 4090, A5000) and --per-gpu 1 a 12 GB one. Every E4 policy of one
run must come from the same backend (docs/prereg_E4.md, A25).

Example:
    python scripts/llm_local.py transfer --source ILI --target ETTh1 --arms league:10,none:10,uniform:5 --per-gpu 2
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.architect import runner  # noqa: E402

PEAK_GB = 8.0


def execute(kind: str, batches: list):
    """Run each batch in parallel processes on the local GPU, one batch after another."""
    from leaguetst.architect.llm import run_job

    context = mp.get_context("spawn")
    for batch in batches:
        try:
            with context.Pool(len(batch)) as pool:
                yield pool.starmap(run_job, [(kind, job) for job in batch])
        except Exception as error:  # the driver logs it with the batch's run names
            yield error


def check_gpu(per_gpu: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU found; run on Modal instead (scripts/modal_llm.py)")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 2**30
    print(f"GPU: {props.name}, {total:.1f} GB, {per_gpu} job(s) at a time")
    if per_gpu * PEAK_GB > total:
        raise SystemExit(f"{per_gpu} jobs need about {per_gpu * PEAK_GB:.0f} GB; use --per-gpu 1 or a larger card")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("transfer",))
    parser.add_argument("--source", required=True, choices=sorted(runner.HORIZON))
    parser.add_argument("--target", required=True, choices=sorted(runner.HORIZON))
    parser.add_argument("--arms", default="league:10,none:10,uniform:5")
    parser.add_argument("--per-gpu", type=int, default=2)
    parser.add_argument("--runs-root", default=str(ROOT / "results" / "runs" / "llm_local"))
    parser.add_argument("--item", default="E4")
    args = parser.parse_args()
    check_gpu(args.per_gpu)
    src_rows, tgt_rows = runner.table_rows(args.source), runner.table_rows(args.target)
    plan = runner.transfer_plan(args.source, args.target, args.arms, runs_root=args.runs_root, per_gpu=args.per_gpu,
                                backend="local", rows_source=src_rows, rows_target=tgt_rows,
                                noise_source=runner.noise_sd(src_rows), noise_target=runner.noise_sd(tgt_rows),
                                item=args.item)
    print(f"{len(plan['train_jobs'])} training runs on {args.source}, deployed on {args.target}, git {plan['train_jobs'][0]['git_hash']}")
    out = runner.run_plan(plan, Path(args.runs_root) / plan["item"] / plan["name"], execute)
    print("results in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
