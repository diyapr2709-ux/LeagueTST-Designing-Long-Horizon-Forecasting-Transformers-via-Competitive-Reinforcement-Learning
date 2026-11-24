"""LLM architect jobs on Modal: prompted search, GRPO training, transfer deployment.

Plans and the driver loop live in leaguetst/architect/runner.py, shared with
scripts/llm_local.py (a local CUDA GPU). Tables are passed in as ledger rows
(train and validation results only); every lookup is charged through
leaguetst.search.replay.Oracle and decoding is constrained to feasible
configs. Each A10G container runs ``per_gpu`` jobs in separate processes
(default 2; three ran out of memory on 2026-11-23). A remote driver runs
training and then deployment, so a sleeping laptop loses nothing; results land
on the leaguetst-runs volume under /<item>/<name>/. A relaunch keeps finished
adapters and deployed seeds.

    M=.venv/bin/modal; A=scripts/modal_llm.py
    MODAL_PROFILE=diyapr $M run --detach $A::prompted --dataset ILI --seeds 10
    MODAL_PROFILE=diyapr $M run --detach $A::transfer --source ETTh1 --target ILI
    MODAL_PROFILE=diyapr $M run $A::versions
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaguetst.architect import runner  # noqa: E402

GPU = os.environ.get("LEAGUETST_GPU", "A10G")
HF, RUNS = "/hf", "/runs"

# Pinned to the versions every LLM run on diyapr used (read with ::versions on 2026-11-24).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pandas==3.1.0", "PyYAML==6.0.3", "scipy==1.18.1",
                 "scikit-learn==1.9.1", "transformers==5.24.0", "peft==0.22.0", "accelerate==1.16.0",
                 "huggingface_hub==2.4.0")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("leaguetst")
    .add_local_dir(ROOT / "configs", "/root/configs")
)
app = modal.App("leaguetst-llm")
hf_vol = modal.Volume.from_name("leaguetst-hf", create_if_missing=True)
runs_vol = modal.Volume.from_name("leaguetst-runs", create_if_missing=True)
VOLUMES = {HF: hf_vol, RUNS: runs_vol}


@app.function(image=image, volumes=VOLUMES, gpu=GPU, cpu=4, memory=49152, timeout=8 * 3600,
              max_containers=10, retries=modal.Retries(max_retries=1, initial_delay=30))
def llm_batch(kind: str, jobs: list) -> list:
    """Run one batch of jobs of one kind in parallel processes on one GPU."""
    import multiprocessing as mp

    from huggingface_hub import snapshot_download

    runs_vol.reload()
    for model, revision in {(job["model"], job.get("revision")) for job in jobs}:
        snapshot_download(model, revision=revision)
    hf_vol.commit()
    from leaguetst.architect.llm import run_job

    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.starmap(run_job, [(kind, job) for job in jobs])
    runs_vol.commit()
    return results


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=4096, timeout=24 * 3600)
def driver(plan: dict) -> str:
    """Training (if any), then search, with every result written to the volume as it arrives."""
    def execute(kind: str, batches: list) -> list:
        return llm_batch.starmap([(kind, batch) for batch in batches], return_exceptions=True)

    return str(runner.run_plan(plan, Path(RUNS) / plan["item"] / plan["name"], execute, runs_vol.commit))


@app.function(image=image, cpu=1, memory=2048, timeout=600)
def image_versions() -> dict:
    """Installed versions of the packages that decide sampling and training."""
    from importlib.metadata import version

    from leaguetst.architect.llm import PACKAGES

    return {name: version(name) for name in PACKAGES}


@app.local_entrypoint()
def versions():
    print(json.dumps(image_versions.remote(), sort_keys=True))


@app.local_entrypoint()
def prompted(dataset: str = "ILI", seeds: int = 10, budget_eq: int = 100, anonymous: bool = False,
             constrained: bool = True, item: str = "E3", per_gpu: int = 2):
    """Qwen without RL, one search per seed; constrained=False runs the repair ablation."""
    rows = runner.table_rows(dataset)
    plan = runner.prompted_plan(dataset, seeds, per_gpu=per_gpu, backend="modal", rows=rows, noise=runner.noise_sd(rows),
                                budget_eq=budget_eq, anonymous=anonymous, constrained=constrained, item=item)
    print("results in", driver.remote(plan))


@app.local_entrypoint()
def transfer(source: str = "ETTh1", target: str = "ILI", arms: str = "league:10,none:10,uniform:5", steps: int = 60,
             budget_eq: int = 100, lr: float = 1e-4, beta: float = 0.04, item: str = "E4", deploy: bool = True,
             per_gpu: int = 2):
    """Train GRPO policies on ``source`` (one per arm and seed), then deploy each frozen on ``target``.

    With ``--no-deploy`` only the training runs; ``deploy_saved`` deploys the saved adapters later.
    """
    src_rows = runner.table_rows(source)
    tgt_rows = runner.table_rows(target) if deploy else []
    plan = runner.transfer_plan(source, target, arms, runs_root=RUNS, per_gpu=per_gpu, backend="modal",
                                rows_source=src_rows, rows_target=tgt_rows, noise_source=runner.noise_sd(src_rows),
                                noise_target=runner.noise_sd(tgt_rows) if deploy else 0.0, steps=steps,
                                budget_eq=budget_eq, lr=lr, beta=beta, item=item, deploy=deploy)
    print(f"{len(plan['train_jobs'])} training runs on {source}" + (f", deployed on {target}" if deploy else ""))
    print("results in", driver.remote(plan))


@app.local_entrypoint()
def deploy_saved(source: str = "ILI", target: str = "ETTh1", arms: str = "league:10,none:10,uniform:5",
                 budget_eq: int = 100, item: str = "E4", per_gpu: int = 2):
    """Deploy adapters already trained by ``transfer --no-deploy`` on ``target``; missing ones are skipped."""
    tgt_rows = runner.table_rows(target)
    plan = runner.transfer_plan(source, target, arms, runs_root=RUNS, per_gpu=per_gpu, backend="modal", rows_source=[],
                                rows_target=tgt_rows, noise_source=0.0, noise_target=runner.noise_sd(tgt_rows),
                                budget_eq=budget_eq, item=item, train=False)
    print("results in", driver.remote(plan))
