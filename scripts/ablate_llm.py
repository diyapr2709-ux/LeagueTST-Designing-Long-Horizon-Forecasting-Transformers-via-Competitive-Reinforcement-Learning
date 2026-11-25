"""Ablation launcher for the LLM architect on one local CUDA GPU (docs/LEG2_soumitra.md L4, AB1, AB2).

scripts/llm_local.py is frozen (one of the ten files in docs/e4_rerun.sha256) and has no flags for the
ablation arms, so this new file builds plans with leaguetst.architect.runner (unchanged) and reuses
llm_local.py's executor and GPU check by import, never by copy. Defaults equal the E4 recipe
(docs/prereg_E4.md "Fixed settings"): 60 GRPO steps, KL weight 0.04, lr 1e-4, budget 100 full-run
equivalents, Qwen2.5-1.5B-Instruct at the A25 revision.

Items (each its own folder under <runs-root>/<item>/<name>/, never reused for another configuration):
    ABL_P_ref_local, ABL_P_05b_local, ABL_P_3b_local     prompted, 30 seeds, ILI and ETTh1
    ABL_G_ref_local, ABL_G_beta0_local, ABL_G_beta01_local  ETTh1 -> ILI, league:10

Examples:
    python scripts/ablate_llm.py prompted --dataset ILI --seeds 30 --model Qwen/Qwen2.5-3B-Instruct \\
        --item ABL_P_3b_local --per-gpu 1
    python scripts/ablate_llm.py transfer --source ETTh1 --target ILI --arms league:10 --beta 0 \\
        --item ABL_G_beta0_local --per-gpu 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import llm_local  # noqa: E402  (frozen: its executor and GPU check are reused, not copied)
from leaguetst.architect import runner  # noqa: E402

# Hugging Face commits, pinned in amendment AB2 before any run. The 1.5B pin is the A25 revision.
PINNED = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": runner.REVISION,
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
}
TAG = {"Qwen/Qwen2.5-0.5B-Instruct": "0.5b", "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
       "Qwen/Qwen2.5-3B-Instruct": "3b"}
E4_RECIPE = {"steps": 60, "beta": 0.04, "lr": 1e-4, "budget_eq": 100}


def method_suffix(model: str) -> str:
    """Names the model and the local backend, e.g. '_3b_local' (prompted method qwen_prompted_3b_local)."""
    return f"_{TAG[model]}_local"


def prompted_plan(args, rows: list, noise: float) -> dict:
    if args.model not in PINNED:
        raise SystemExit(f"unpinned model {args.model}; pinned: {sorted(PINNED)}")
    revision = args.revision or PINNED[args.model]
    if revision != PINNED[args.model]:
        raise SystemExit(f"{args.model} revision {revision} is not the pinned {PINNED[args.model]} (AB2)")
    return runner.prompted_plan(args.dataset, args.seeds, per_gpu=args.per_gpu, backend="local", rows=rows,
                                noise=noise, budget_eq=E4_RECIPE["budget_eq"], anonymous=args.anonymous,
                                constrained=not args.no_constrained, item=args.item, model=args.model,
                                revision=revision, suffix=method_suffix(args.model))


def transfer_plan(args, src_rows: list, tgt_rows: list, noise_source: float, noise_target: float) -> dict:
    return runner.transfer_plan(args.source, args.target, args.arms, runs_root=args.runs_root,
                                per_gpu=args.per_gpu, backend="local", rows_source=src_rows, rows_target=tgt_rows,
                                noise_source=noise_source, noise_target=noise_target, steps=args.steps,
                                budget_eq=E4_RECIPE["budget_eq"], lr=E4_RECIPE["lr"], beta=args.beta,
                                item=args.item)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("transfer", "prompted"))
    p.add_argument("--item", required=True, help="ablation item, e.g. ABL_G_beta0_local")
    p.add_argument("--per-gpu", type=int, default=1)
    p.add_argument("--runs-root", default=str(ROOT / "results" / "runs" / "llm_local"))
    # transfer
    p.add_argument("--source", choices=sorted(runner.HORIZON))
    p.add_argument("--target", choices=sorted(runner.HORIZON))
    p.add_argument("--arms", default="league:10")
    p.add_argument("--steps", type=int, default=E4_RECIPE["steps"])
    p.add_argument("--beta", type=float, default=E4_RECIPE["beta"])
    # prompted
    p.add_argument("--dataset", choices=sorted(runner.HORIZON))
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--anonymous", action="store_true")
    p.add_argument("--no-constrained", action="store_true")
    p.add_argument("--model", default=runner.MODEL, choices=sorted(PINNED))
    p.add_argument("--revision", default=None, help="must equal the AB2 pin (default: the pin)")
    return p


def build(args) -> dict:
    """The plan for the parsed arguments (no GPU needed; used by the tests)."""
    if args.command == "prompted":
        if not args.dataset:
            raise SystemExit("prompted needs --dataset")
        rows = runner.table_rows(args.dataset)
        return prompted_plan(args, rows, runner.noise_sd(rows))
    if not (args.source and args.target):
        raise SystemExit("transfer needs --source and --target")
    if args.model != runner.MODEL or args.revision not in (None, runner.REVISION):
        raise SystemExit("transfer arms use the E4 model and revision only")
    src, tgt = runner.table_rows(args.source), runner.table_rows(args.target)
    return transfer_plan(args, src, tgt, runner.noise_sd(src), runner.noise_sd(tgt))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    plan = build(args)
    llm_local.check_gpu(args.per_gpu)
    jobs = plan["train_jobs"] or plan["search_jobs"]
    print(f"{args.item}: {len(plan['train_jobs'])} training and {len(plan['search_jobs'])} search jobs, "
          f"git {jobs[0]['git_hash']}, model {jobs[0]['model']}@{jobs[0]['revision']}", flush=True)
    out = runner.run_plan(plan, Path(args.runs_root) / plan["item"] / plan["name"], llm_local.execute)
    print("results in", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
