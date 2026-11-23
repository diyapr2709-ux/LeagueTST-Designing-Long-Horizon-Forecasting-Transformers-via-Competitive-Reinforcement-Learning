"""LLM architect jobs on Modal: prompted search, GRPO training, transfer deployment.

Tables are passed in as ledger rows (train and validation results only).
Every lookup is charged through leaguetst.search.replay.Oracle, as for the
CPU methods, and decoding is constrained to feasible configs. Each A10G
container runs up to ``PER_GPU`` jobs in separate processes. A remote
driver runs training and then deployment, so a sleeping laptop loses
nothing; results land on the leaguetst-runs volume under /<item>/<name>/.

    M=.venv/bin/modal; A=scripts/modal_llm.py
    MODAL_PROFILE=diyapr $M run --detach $A::prompted --dataset ILI --seeds 10
    MODAL_PROFILE=diyapr $M run --detach $A::transfer --source ETTh1 --target ILI
    MODAL_PROFILE=diyapr $M volume get leaguetst-runs /E4/<name> results/search/E4/
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

MODEL = os.environ.get("LEAGUETST_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
GPU = os.environ.get("LEAGUETST_GPU", "A10G")
PER_GPU = int(os.environ.get("LEAGUETST_PER_GPU", "3"))
HF, RUNS = "/hf", "/runs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pandas==3.1.0", "PyYAML==6.0.3", "scipy", "scikit-learn",
                 "transformers", "peft", "accelerate", "huggingface_hub")
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
    """Run up to PER_GPU jobs of one kind in parallel processes on one GPU."""
    import multiprocessing as mp

    from huggingface_hub import snapshot_download

    runs_vol.reload()
    for model in {job["model"] for job in jobs}:
        snapshot_download(model)
    hf_vol.commit()
    from leaguetst.architect.llm import run_job

    context = mp.get_context("spawn")
    with context.Pool(len(jobs)) as pool:
        results = pool.starmap(run_job, [(kind, job) for job in jobs])
    runs_vol.commit()
    return results


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _append(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=4096, timeout=24 * 3600)
def driver(plan: dict) -> str:
    """Training (if any), then search, with every result written to the volume as it arrives."""
    out = Path(RUNS) / plan["item"] / plan["name"]
    out.mkdir(parents=True, exist_ok=True)
    adapters: dict[str, str] = {}
    train_jobs = plan.get("train_jobs", [])
    if train_jobs:
        batches = _chunks(train_jobs, PER_GPU)
        for batch in llm_batch.starmap([("train", b) for b in batches], return_exceptions=True):
            if isinstance(batch, Exception):
                _append(out / "errors.jsonl", [{"phase": "train", "error": repr(batch)}])
                continue
            for result in batch:
                if "error" in result:
                    _append(out / "errors.jsonl", [result])
                    continue
                _append(out / "training.jsonl", [result])
                adapters[result["job"]["run_name"]] = result["adapter"]
            runs_vol.commit()
    search_jobs = []
    for job in plan["search_jobs"]:
        if job.get("needs_adapter"):
            if job["needs_adapter"] not in adapters:
                continue
            job = {**job, "adapter": adapters[job["needs_adapter"]]}
        search_jobs.append(job)
    for batch in llm_batch.starmap([("search", b) for b in _chunks(search_jobs, PER_GPU)], return_exceptions=True):
        if isinstance(batch, Exception):
            _append(out / "errors.jsonl", [{"phase": "search", "error": repr(batch)}])
            continue
        for result in batch:
            if isinstance(result, dict) and "error" in result:
                _append(out / "errors.jsonl", [result])
            else:
                for record in result:
                    _append(out / f"{record['method']}_runs.jsonl", [record])
        runs_vol.commit()
    return str(out)


HORIZON = {"ETTh1": 336, "ILI": 24}


def _rows(dataset: str) -> list:
    from leaguetst.ledger import read_rows

    keep = ("config_id", "seed", "val_curve", "train_curve", "best_val_mse", "epochs_run", "model_config")
    return [{k: r[k] for k in keep} for r in read_rows(ROOT / "results" / "ledger.jsonl")
            if r["item"] == "E1" and r["dataset"] == dataset and int(r["horizon"]) == HORIZON[dataset]]


def _noise(rows: list) -> float:
    from leaguetst.search.replay import Table

    sys.path.insert(0, str(ROOT / "scripts"))
    from replay import observation_noise

    return observation_noise(Table(rows))


@app.local_entrypoint()
def prompted(dataset: str = "ILI", seeds: int = 10, budget_eq: int = 100, anonymous: bool = False,
             constrained: bool = True, item: str = "E3", model: str = MODEL):
    """Qwen without RL, one search per seed; constrained=False runs the repair ablation."""
    rows = _rows(dataset)
    method = "qwen_prompted" + ("_anon" if anonymous else "") + ("" if constrained else "_repair")
    jobs = [{"kind": "search", "rows": rows, "dataset": dataset, "horizon": HORIZON[dataset], "budget": budget_eq * 30,
             "method": method, "seeds": [s], "anonymous": anonymous, "constrained": constrained, "model": model,
             "noise_sd": _noise(rows)} for s in range(seeds)]
    plan = {"item": item, "name": f"{dataset}_T{HORIZON[dataset]}", "search_jobs": jobs}
    print("results in", driver.remote(plan))


@app.local_entrypoint()
def transfer(source: str = "ETTh1", target: str = "ILI", arms: str = "league:10,none:10,uniform:5", steps: int = 60,
             budget_eq: int = 100, lr: float = 1e-4, beta: float = 0.04, item: str = "E4", model: str = MODEL):
    """Train GRPO policies on ``source`` (one per arm and seed), then deploy each frozen on ``target``."""
    src_rows, tgt_rows = _rows(source), _rows(target)
    src_noise, tgt_noise = _noise(src_rows), _noise(tgt_rows)
    train_jobs, search_jobs = [], []
    for arm_spec in arms.split(","):
        arm, count = arm_spec.split(":")
        for s in range(int(count)):
            run_name = f"grpo_{arm}_{source}_seed{100 + s}"
            train_jobs.append({"rows": src_rows, "dataset": source, "horizon": HORIZON[source], "seed": 100 + s,
                               "steps": steps, "episode_budget": budget_eq * 30, "rivals": arm, "lr": lr, "beta": beta,
                               "model": model, "constrained": True, "noise_sd": src_noise, "run_name": run_name,
                               "out_dir": f"{RUNS}/{item}/adapters/{run_name}"})
            search_jobs.append({"rows": tgt_rows, "dataset": target, "horizon": HORIZON[target], "budget": budget_eq * 30,
                                "method": f"grpo_{arm}_from_{source}", "seeds": [s], "constrained": True, "model": model,
                                "noise_sd": tgt_noise, "needs_adapter": run_name})
    plan = {"item": item, "name": f"{target}_T{HORIZON[target]}", "train_jobs": train_jobs, "search_jobs": search_jobs}
    print(f"{len(train_jobs)} training runs on {source}, deployed on {target}")
    print("results in", driver.remote(plan))
