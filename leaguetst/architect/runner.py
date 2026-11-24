"""Plans and the driver loop for LLM architect jobs, shared by Modal and a local GPU.

A plan lists training jobs (GRPO policies) and search jobs (prompted or
deployed policies). ``run_plan`` runs it with any executor that maps batches
of jobs to results, so scripts/modal_llm.py (A10G containers) and
scripts/llm_local.py (one local CUDA GPU, such as an RTX card) share every
rule: arms are interleaved over GPUs, a relaunch keeps finished adapters and
deployed (method, seed) pairs, a run stopped by the on-policy check is never
retrained, and every error names its runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# The snapshot every LLM run so far used (leaguetst-hf volume on diyapr).
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
HORIZON = {"ETTh1": 336, "ILI": 24}
ON_POLICY_STOP = "log-probabilities differ"

Executor = Callable[[str, list[list[dict[str, Any]]]], Iterable[Any]]


def git_hash() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "leaguetst", "scripts"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def table_rows(dataset: str) -> list[dict[str, Any]]:
    """E1 rows of one table from the committed ledger (train and validation results only)."""
    from leaguetst.ledger import read_rows

    keep = ("config_id", "seed", "val_curve", "train_curve", "best_val_mse", "epochs_run", "model_config")
    return [{k: r[k] for k in keep} for r in read_rows(ROOT / "results" / "ledger.jsonl")
            if r["item"] == "E1" and r["dataset"] == dataset and int(r["horizon"]) == HORIZON[dataset]]


def noise_sd(rows: list[dict[str, Any]]) -> float:
    from leaguetst.search.replay import Table

    sys.path.insert(0, str(ROOT / "scripts"))
    from replay import observation_noise

    return observation_noise(Table(rows))


def chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def parse_arms(arms: str) -> list[tuple[str, int]]:
    return [(arm, int(count)) for arm, count in (spec.split(":") for spec in arms.split(","))]


def transfer_plan(source: str, target: str, arms: str, *, runs_root: str, per_gpu: int, backend: str,
                  rows_source: list, rows_target: list, noise_source: float, noise_target: float,
                  steps: int = 60, budget_eq: int = 100, lr: float = 1e-4, beta: float = 0.04, item: str = "E4",
                  model: str = MODEL, revision: str = REVISION, deploy: bool = True, train: bool = True) -> dict:
    """GRPO policies on ``source`` (one per arm and seed), each deployed frozen on ``target``.

    Jobs are ordered seed by seed across arms, so the arms share GPUs evenly.
    With ``train=False`` only adapters that already exist are deployed.
    """
    common = {"model": model, "revision": revision, "git_hash": git_hash(), "per_gpu": per_gpu, "backend": backend}
    counts = parse_arms(arms)
    train_jobs, search_jobs = [], []
    for s in range(max(count for _, count in counts)):
        for arm, count in counts:
            if s >= count:
                continue
            run_name = f"grpo_{arm}_{source}_seed{100 + s}"
            train_jobs.append({**common, "rows": rows_source, "dataset": source, "horizon": HORIZON[source],
                               "seed": 100 + s, "steps": steps, "episode_budget": budget_eq * 30, "rivals": arm,
                               "lr": lr, "beta": beta, "constrained": True, "noise_sd": noise_source,
                               "run_name": run_name, "out_dir": f"{runs_root}/{item}/adapters/{run_name}"})
            if deploy:
                search_jobs.append({**common, "rows": rows_target, "dataset": target, "horizon": HORIZON[target],
                                    "budget": budget_eq * 30, "method": f"grpo_{arm}_from_{source}", "seeds": [s],
                                    "constrained": True, "noise_sd": noise_target, "needs_adapter": run_name})
    name = f"{target}_T{HORIZON[target]}" if deploy else f"train_{source}"
    return {"item": item, "name": name, "per_gpu": per_gpu, "train": train,
            "train_jobs": train_jobs, "search_jobs": search_jobs}


def prompted_plan(dataset: str, seeds: int, *, per_gpu: int, backend: str, rows: list, noise: float,
                  budget_eq: int = 100, anonymous: bool = False, constrained: bool = True, item: str = "E3",
                  model: str = MODEL, revision: str = REVISION, suffix: str = "") -> dict:
    """Qwen without RL, one search per seed; constrained=False runs the repair ablation.

    ``suffix`` names runs made on another backend (A26: "_local").
    """
    method = "qwen_prompted" + ("_anon" if anonymous else "") + ("" if constrained else "_repair") + suffix
    jobs = [{"model": model, "revision": revision, "git_hash": git_hash(), "per_gpu": per_gpu, "backend": backend,
             "rows": rows, "dataset": dataset, "horizon": HORIZON[dataset], "budget": budget_eq * 30, "method": method,
             "seeds": [s], "anonymous": anonymous, "constrained": constrained, "noise_sd": noise} for s in range(seeds)]
    return {"item": item, "name": f"{dataset}_T{HORIZON[dataset]}", "per_gpu": per_gpu, "train": False,
            "train_jobs": [], "search_jobs": jobs}


def _append(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stopped_runs(out: Path) -> set[str]:
    """Runs that the on-policy check stopped: reported as failed, never retrained (A22)."""
    return {name for e in _read(out / "errors.jsonl") if ON_POLICY_STOP in e.get("error", "")
            for name in e.get("run_names", [])}


def deployed(out: Path) -> set[tuple[str, int]]:
    return {(r["method"], int(r["seed"])) for path in out.glob("*_runs.jsonl") for r in _read(path)}


def run_plan(plan: dict, out: Path, execute: Executor, commit: Callable[[], None] = lambda: None) -> Path:
    """Training (if any), then search, with every result appended under ``out`` as it arrives."""
    out.mkdir(parents=True, exist_ok=True)
    errors = out / "errors.jsonl"
    stopped = stopped_runs(out)
    adapters: dict[str, str] = {}
    pending = []
    for job in plan.get("train_jobs", []):
        saved = Path(job["out_dir"]) / "adapter"
        if (saved / "adapter_config.json").exists():
            adapters[job["run_name"]] = str(saved)
        elif plan.get("train", True) and job["run_name"] not in stopped:
            pending.append(job)
    batches = chunks(pending, plan["per_gpu"])
    for batch, result in zip(batches, execute("train", batches)):
        if isinstance(result, BaseException):
            _append(errors, [{"phase": "train", "error": repr(result), "run_names": [j["run_name"] for j in batch]}])
            continue
        for record in result:
            if "error" in record:
                _append(errors, [{**record, "phase": "train", "run_names": [record["job"]["run_name"]]}])
            else:
                _append(out / "training.jsonl", [record])
                adapters[record["job"]["run_name"]] = record["adapter"]
        commit()

    done = deployed(out)
    search_jobs = []
    for job in plan["search_jobs"]:
        if (job["method"], int(job["seeds"][0])) in done:
            continue
        if job.get("needs_adapter"):
            if job["needs_adapter"] not in adapters:
                continue
            job = {**job, "adapter": adapters[job["needs_adapter"]]}
        search_jobs.append(job)
    batches = chunks(search_jobs, plan["per_gpu"])
    for batch, result in zip(batches, execute("search", batches)):
        names = [f"{j['method']}:{j['seeds'][0]}" for j in batch]
        if isinstance(result, BaseException):
            _append(errors, [{"phase": "search", "error": repr(result), "run_names": names}])
            continue
        for name, records in zip(names, result):
            if isinstance(records, dict) and "error" in records:
                _append(errors, [{**records, "phase": "search", "run_names": [name]}])
            else:
                for record in records:
                    _append(out / f"{record['method']}_runs.jsonl", [record])
        commit()
    return out
