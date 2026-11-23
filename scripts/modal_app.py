"""LeagueTST GPU jobs on Modal: the K5 pilot and the tabular benchmark.

ETTh1 jobs run in workspace diya-027 and ILI jobs in diyapr, so each table
and everything that reads it stay in one workspace. Jobs read only the
leaguetst-data volume (data/trainval, no test rows) and write to
leaguetst-runs under /<item>/<dataset>_T<horizon>_s<seed>/.

    M=.venv/bin/modal; A=scripts/modal_app.py
    MODAL_PROFILE=diya-027 $M run $A::pilot --dataset ETTh1 --horizon 336
    MODAL_PROFILE=diya-027 $M run --detach $A::table --item E1 --dataset ETTh1 --horizon 336 --seed 2021
    MODAL_PROFILE=diya-027 $M run $A::status --item E1 --dataset ETTh1 --horizon 336 --seed 2021
    MODAL_PROFILE=diya-027 $M volume get leaguetst-runs /E1/ETTh1_T336_s2021/records results/runs/E1/ETTh1_T336_s2021/

A record that exists on the volume is never trained again, so an
interrupted table resumes when the same command is run again.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA, RUNS = "/data", "/runs"
TRAINVAL = f"{DATA}/trainval"
# L4 had no capacity on 23 Nov (an L4 probe waited minutes while A10G and T4
# started in seconds), so the GPU type is chosen per launch.
GPU = os.environ.get("LEAGUETST_GPU", "A10G")
MAX_CONTAINERS = int(os.environ.get("LEAGUETST_MAX_CONTAINERS", "10"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pandas==3.1.0", "PyYAML==6.0.3")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("leaguetst")
    .add_local_dir(ROOT / "configs", "/root/configs")
)

app = modal.App("leaguetst")
data_vol = modal.Volume.from_name("leaguetst-data")
runs_vol = modal.Volume.from_name("leaguetst-runs", create_if_missing=True)
VOLUMES = {DATA: data_vol, RUNS: runs_vol}


def _run_dir(item: str, dataset: str, horizon: int, seed: int) -> Path:
    return Path(RUNS) / item / f"{dataset}_T{horizon}_s{seed}"


@app.function(image=image, volumes=VOLUMES, gpu=GPU, cpu=4, memory=16384, timeout=8 * 3600,
              max_containers=MAX_CONTAINERS, retries=modal.Retries(max_retries=1, initial_delay=30))
def train_chunk(job: dict) -> dict:
    """Train a list of configs on one GPU with several worker processes."""
    import multiprocessing as mp

    import torch

    from leaguetst.table import worker_main

    runs_vol.reload()
    out_dir = _run_dir(job["item"], job["dataset"], int(job["horizon"]), int(job["seed"]))
    (out_dir / "records").mkdir(parents=True, exist_ok=True)
    job = {**job, "trainval_dir": TRAINVAL}
    entries = [tuple(e) for e in job["entries"]]
    workers = max(1, int(job.get("workers", 4)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(60):
            try:
                runs_vol.commit()
            except Exception as error:
                print("commit failed:", error, flush=True)

    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    start = time.time()
    context = mp.get_context("spawn")
    with context.Pool(workers) as pool:
        parts = pool.starmap(worker_main, [(job, entries[w::workers], str(out_dir), device) for w in range(workers)])
    stop.set()
    thread.join()
    runs_vol.commit()
    totals = {k: sum(p[k] for p in parts) for k in ("done", "skipped", "failed")}
    totals["seconds"] = round(time.time() - start, 1)
    totals["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    totals["workers"] = workers
    return totals


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=2048, timeout=24 * 3600)
def driver(jobs: list) -> list:
    """Run every chunk in parallel and keep a summary on the volume."""
    results = []
    for result in train_chunk.map(jobs, return_exceptions=True):
        results.append(result if isinstance(result, dict) else {"error": repr(result)})
        print(results[-1], flush=True)
    first = jobs[0]
    summary_path = _run_dir(first["item"], first["dataset"], int(first["horizon"]), int(first["seed"])) / "driver_summary.json"
    runs_vol.reload()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    runs_vol.commit()
    return results


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=2048, timeout=600)
def count_records(item: str, dataset: str, horizon: int, seed: int) -> dict:
    runs_vol.reload()
    out_dir = _run_dir(item, dataset, horizon, seed)
    records = sorted((out_dir / "records").glob("*.json")) if out_dir.exists() else []
    failed = sorted((out_dir / "failed").glob("*.json")) if (out_dir / "failed").exists() else []
    walls = []
    for path in records[-200:]:
        try:
            walls.append(json.loads(path.read_text())["wall_seconds"])
        except Exception:
            pass
    return {"records": len(records), "failed": len(failed),
            "mean_wall_seconds_last_200": round(sum(walls) / len(walls), 1) if walls else None,
            "summary": (out_dir / "driver_summary.json").read_text() if (out_dir / "driver_summary.json").exists() else ""}


def _git_hash() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "leaguetst", "configs"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def _chunks(entries: list, n_chunks: int) -> list[list]:
    """Spread entries over chunks round-robin, so every chunk mixes small and large configs."""
    return [entries[i::n_chunks] for i in range(n_chunks) if entries[i::n_chunks]]


def _job(item, dataset, horizon, seed, entries, workers, epochs, patience) -> dict:
    from leaguetst.table import recipe

    return {"item": item, "dataset": dataset, "horizon": horizon, "seed": seed, "entries": entries,
            "training": recipe(dataset, epochs=epochs, patience=patience), "git_hash": _git_hash(),
            "workers": workers, "threads_per_worker": 1}


@app.local_entrypoint()
def pilot(dataset: str = "ETTh1", horizon: int = 336, n_configs: int = 32, workers: int = 4):
    """K5: PatchTST/42 at three seeds, plus a timed batch of random feasible configs."""
    from leaguetst.search import configs_by_id
    from leaguetst.table import TABLE_EPOCHS, TABLE_PATIENCE, patchtst42_config

    item = f"K5_{GPU}"
    reference = [("patchtst42", patchtst42_config(dataset))]
    jobs = [_job(item, dataset, horizon, seed, reference, 1, TABLE_EPOCHS, TABLE_PATIENCE) for seed in (2021, 2022, 2023)]
    table = configs_by_id()
    picked = random.Random(5).sample(sorted(table), n_configs)
    entries = [(key, table[key].to_dict()) for key in picked]
    jobs.append(_job(item, dataset, horizon, 2021, entries, workers, TABLE_EPOCHS, TABLE_PATIENCE))
    jobs.append(_job(item + "w1", dataset, horizon, 2021, entries[:8], 1, TABLE_EPOCHS, TABLE_PATIENCE))
    for job, result in zip(jobs, train_chunk.map(jobs, return_exceptions=True)):
        print(job["item"], job["seed"], len(job["entries"]), "configs,", job["workers"], "workers:", result)


@app.local_entrypoint()
def table(item: str = "E1", dataset: str = "ETTh1", horizon: int = 336, seed: int = 2021,
          n_chunks: int = 10, workers: int = 4, ids_file: str = "", epochs: int = 30, patience: int = 10):
    """Launch the table (all 2,368 configs, or those listed in ids_file) through a remote driver."""
    from leaguetst.search import configs_by_id

    table_ = configs_by_id()
    keys = sorted(table_)
    if ids_file:
        keys = [k.strip() for k in Path(ids_file).read_text().split() if k.strip()]
        unknown = [k for k in keys if k not in table_]
        if unknown:
            raise SystemExit(f"unknown config ids: {unknown[:5]}")
    # A fixed shuffle spreads large and small configs evenly over the chunks.
    random.Random(seed).shuffle(keys)
    entries = [(k, table_[k].to_dict()) for k in keys]
    jobs = [_job(item, dataset, horizon, seed, chunk, workers, epochs, patience) for chunk in _chunks(entries, n_chunks)]
    print(f"{item} {dataset} T={horizon} seed {seed}: {len(entries)} configs in {len(jobs)} chunks, git {jobs[0]['git_hash']}")
    for result in driver.remote(jobs):
        print(result)


@app.local_entrypoint()
def status(item: str = "E1", dataset: str = "ETTh1", horizon: int = 336, seed: int = 2021):
    print(json.dumps(count_records.remote(item, dataset, horizon, seed), indent=2))
