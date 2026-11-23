"""LLM architect jobs on Modal: prompted search, GRPO training, deployment.

Tables are passed in as ledger rows (train and validation results only),
so a policy trained on one workspace's table can be deployed on the other.
Every lookup is charged through leaguetst.search.replay.Oracle, exactly as
for the CPU methods.

    M=.venv/bin/modal; A=scripts/modal_llm.py
    MODAL_PROFILE=diyapr $M run $A::prompted --dataset ILI --horizon 24 --seeds 10
    MODAL_PROFILE=diyapr $M run --detach $A::grpo --source ILI --target ETTh1 --seeds 10 --rivals league
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = os.environ.get("LEAGUETST_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
GPU = os.environ.get("LEAGUETST_GPU", "A10G")
HF, RUNS = "/hf", "/runs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pandas==3.1.0", "PyYAML==6.0.3", "scipy", "scikit-learn",
                 "transformers", "peft", "accelerate")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("leaguetst")
    .add_local_dir(ROOT / "configs", "/root/configs")
)
app = modal.App("leaguetst-llm")
hf_vol = modal.Volume.from_name("leaguetst-hf", create_if_missing=True)
runs_vol = modal.Volume.from_name("leaguetst-runs", create_if_missing=True)
VOLUMES = {HF: hf_vol, RUNS: runs_vol}


def _load(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    hf_vol.commit()
    return tok, model


def _generate(tok, model, messages, n: int, temperature: float, max_new_tokens: int = 112):
    import torch

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok([text] * n, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id)
    completions = out[:, enc["input_ids"].shape[1]:]
    return enc["input_ids"][0], completions, tok.batch_decode(completions, skip_special_tokens=True)


def _search_record(method, seed, oracle, log, budgets_epochs):
    return {"method": method, "seed": seed, "spent": oracle.spent, "lookups": len(oracle.charged), "trace": oracle.trace,
            "charges": [list(c) for c in oracle.charges], "steps": log.steps, "proposals": log.proposals,
            "invalid": log.invalid, "repaired": log.repaired, "repeats": log.repeats, "reasons": log.reasons}


@app.function(image=image, volumes=VOLUMES, gpu=GPU, cpu=4, memory=32768, timeout=6 * 3600,
              max_containers=10, retries=modal.Retries(max_retries=1, initial_delay=30))
def llm_search(job: dict) -> list:
    """Search episodes on a table with the base model or a saved adapter."""
    import torch

    from leaguetst.architect import run_episode
    from leaguetst.search.replay import Oracle, Space, Table

    torch.manual_seed(int(job["seeds"][0]))
    tok, model = _load(job.get("model", MODEL))
    if job.get("adapter"):
        from peft import PeftModel

        runs_vol.reload()
        model = PeftModel.from_pretrained(model, job["adapter"])
    model.eval()
    table = Table(job["rows"])
    space = Space(table.ids)
    temperature = float(job.get("temperature", 1.0))
    results = []
    for seed in job["seeds"]:
        torch.manual_seed(seed)
        oracle = Oracle(table, int(job["budget"]), int(seed))
        start = time.time()
        log = run_episode(oracle, space, job["dataset"], int(job["horizon"]),
                          lambda messages, n: _generate(tok, model, messages, n, temperature)[2],
                          group=int(job.get("group", 8)), anonymous=bool(job.get("anonymous", False)))
        record = _search_record(job["method"], seed, oracle, log, None)
        record["seconds"] = time.time() - start
        record["gpu"] = torch.cuda.get_device_name(0)
        results.append(record)
        print(job["method"], seed, "spent", oracle.spent, "lookups", len(oracle.charged), "invalid", log.invalid,
              "of", log.proposals, f"{record['seconds']:.0f}s", flush=True)
    return results


@app.function(image=image, volumes=VOLUMES, gpu=GPU, cpu=4, memory=32768, timeout=6 * 3600,
              max_containers=10, retries=modal.Retries(max_retries=1, initial_delay=30))
def grpo_train(job: dict) -> dict:
    """GRPO with LoRA on one table; rivals from past snapshots enter the baseline only."""
    import copy
    import math

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    from leaguetst.architect.episode import EpisodeLog, fill_wasted, score_group
    from leaguetst.architect.prompt import build_messages
    from leaguetst.search.replay import BudgetExhausted, Oracle, Space, Table

    seed = int(job["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, base = _load(job.get("model", MODEL))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    base.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lora)
    if job.get("rivals") == "league":
        model.add_adapter("rival", lora)
        model.set_adapter("default")
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad and "rival" not in n],
                                  lr=float(job.get("lr", 5e-5)))
    table = Table(job["rows"])
    space = Space(table.ids)
    dataset, horizon = job["dataset"], int(job["horizon"])
    group, n_rivals, beta = int(job.get("group", 8)), int(job.get("n_rivals", 4)), float(job.get("beta", 0.04))
    anonymous = bool(job.get("anonymous", False))
    episode_seed = seed * 1000
    oracle = Oracle(table, int(job["episode_budget"]), episode_seed)
    history: dict[str, float] = {}
    snapshots = [copy.deepcopy({k: v.detach().cpu() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()})]
    out_dir = Path(job["out_dir"])
    log_rows, training_epochs, episodes = [], 0, 1
    for step in range(1, int(job["steps"]) + 1):
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        messages = build_messages(dataset, horizon, space, list(history.items()), anonymous)
        model.eval()
        prompt_ids, completions, texts = _generate(tok, model, messages, group, 1.0)
        rival_texts: list[str] = []
        rival_ids: list[str] = []
        if job.get("rivals") == "league":
            state = snapshots[int(rng.integers(len(snapshots)))]
            set_peft_model_state_dict(model, state, adapter_name="rival")
            model.set_adapter("rival")
            rival_texts = _generate(tok, model, messages, n_rivals, 1.0)[2]
            model.set_adapter("default")
        elif job.get("rivals") == "random":
            rival_ids = [table.ids[int(i)] for i in rng.integers(0, len(table.ids), n_rivals)]
        log = EpisodeLog()
        spent_before = oracle.spent
        try:
            rewards, ids = score_group(oracle, space, texts, history, anonymous, log, rng)
            rival_rewards, _ = score_group(oracle, space, rival_texts, history, anonymous, EpisodeLog(), rng)
            rival_rewards = [r for r in rival_rewards if r is not None]
            rival_rewards += [-math.log(oracle.full(c)) for c in rival_ids if c not in history]
        except BudgetExhausted:
            training_epochs += oracle.spent - spent_before
            episodes += 1
            oracle = Oracle(table, int(job["episode_budget"]), episode_seed + episodes)
            history = {}
            continue
        training_epochs += oracle.spent - spent_before
        for reward, cid in zip(rewards, ids):
            if reward is not None and cid is not None:
                history[cid] = math.exp(-reward)
        filled = fill_wasted(rewards + rival_rewards)
        own = np.array(filled[:group])
        pool = np.array(filled)
        advantages = torch.tensor((own - pool.mean()) / (pool.std() + 1e-6), dtype=torch.float32, device="cuda")

        model.train()
        full = torch.cat([prompt_ids.unsqueeze(0).repeat(group, 1), completions], dim=1)
        mask = (completions != tok.pad_token_id).float()
        attention = torch.cat([torch.ones(group, prompt_ids.shape[0], device="cuda"), mask], dim=1)
        keep = completions.shape[1] + 1  # logits only for the positions that predict completion tokens

        def token_logp(net) -> torch.Tensor:
            logits = net(input_ids=full, attention_mask=attention, logits_to_keep=keep).logits[:, :-1].float()
            chosen = logits.gather(-1, completions.unsqueeze(-1)).squeeze(-1)
            return chosen - torch.logsumexp(logits, dim=-1)

        logp = token_logp(model)
        with torch.no_grad(), model.disable_adapter():
            ref_logp = token_logp(model)
        delta = ref_logp - logp
        kl = torch.exp(delta) - delta - 1.0
        per_token = -(advantages.unsqueeze(1) * logp) + beta * kl
        loss = ((per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        valid_ids = [c for c, r in zip(ids, rewards) if r is not None]
        row = {"step": step, "episode": episodes, "loss": float(loss), "kl": float((kl * mask).sum() / mask.sum()),
               "reward_mean": float(np.mean([r for r in rewards if r is not None])) if valid_ids else None,
               "invalid_rate": log.invalid / group, "repaired_rate": log.repaired / group, "repeat_rate": log.repeats / group,
               "reasons": log.reasons, "sample": texts[0][:200] if step in (1, 10) else None,
               "unique_rate": len(set(c for c in ids if c is not None)) / group,
               "best_in_history": min(history.values()) if history else None, "seconds": time.time() - start,
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "training_epochs": training_epochs}
        log_rows.append(row)
        print(json.dumps(row), flush=True)
        if step % int(job.get("snapshot_every", 5)) == 0:
            snapshots.append(copy.deepcopy({k: v.detach().cpu() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}))
    out_dir.mkdir(parents=True, exist_ok=True)
    if job.get("rivals") == "league":
        model.delete_adapter("rival")
    model.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "train_log.json").write_text(json.dumps({"job": {k: v for k, v in job.items() if k != "rows"}, "log": log_rows}))
    runs_vol.commit()
    return {"adapter": str(out_dir / "adapter"), "steps": len(log_rows), "training_epochs": training_epochs,
            "episodes": episodes, "gpu": torch.cuda.get_device_name(0),
            "seconds": sum(r["seconds"] for r in log_rows), "last": log_rows[-1] if log_rows else None}


def _rows(dataset: str, horizon: int, item: str = "E1") -> list:
    from leaguetst.ledger import read_rows

    keep = ("config_id", "seed", "val_curve", "train_curve", "best_val_mse", "epochs_run", "model_config")
    return [{k: r[k] for k in keep} for r in read_rows(ROOT / "results" / "ledger.jsonl")
            if r["item"] == item and r["dataset"] == dataset and int(r["horizon"]) == horizon]


HORIZON = {"ETTh1": 336, "ILI": 24}


def _append(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=4096, timeout=24 * 3600)
def driver(plan: dict) -> str:
    """Run training (if any) then search jobs remotely, so a sleeping laptop loses nothing.

    Results go to /runs/<item>/<name>/ on the volume: training.jsonl and
    <method>_runs.jsonl, collected later with ``modal volume get``.
    """
    out = Path(RUNS) / plan["item"] / plan["name"]
    out.mkdir(parents=True, exist_ok=True)
    adapters = []
    if plan.get("train_jobs"):
        for job, result in zip(plan["train_jobs"], grpo_train.map(plan["train_jobs"], return_exceptions=True)):
            if isinstance(result, Exception):
                _append(out / "training.jsonl", [{"out_dir": job["out_dir"], "error": repr(result)}])
                continue
            _append(out / "training.jsonl", [{"out_dir": job["out_dir"], **result}])
            adapters.append(result["adapter"])
            runs_vol.commit()
    search_jobs = plan["search_jobs"]
    if plan.get("train_jobs"):
        search_jobs = [{**job, "adapter": adapter} for job, adapter in zip(search_jobs, adapters)]
    for result in llm_search.map(search_jobs, return_exceptions=True):
        if isinstance(result, Exception):
            _append(out / "errors.jsonl", [{"error": repr(result)}])
        else:
            _append(out / f"{plan['method']}_runs.jsonl", result)
        runs_vol.commit()
    return str(out)


@app.local_entrypoint()
def prompted(dataset: str = "ILI", seeds: int = 10, budget_eq: int = 100, anonymous: bool = False, item: str = "E3",
             model: str = MODEL, per_container: int = 2):
    """Prompted Qwen without RL, ``seeds`` search seeds spread over containers."""
    rows = _rows(dataset, HORIZON[dataset])
    method = "qwen_prompted" + ("_anon" if anonymous else "")
    jobs = [{"rows": rows, "dataset": dataset, "horizon": HORIZON[dataset], "budget": budget_eq * 30, "method": method,
             "seeds": list(range(s, min(s + per_container, seeds))), "anonymous": anonymous, "model": model}
            for s in range(0, seeds, per_container)]
    plan = {"item": item, "name": f"{dataset}_T{HORIZON[dataset]}", "method": method, "search_jobs": jobs}
    print("results in", driver.remote(plan))


@app.local_entrypoint()
def grpo(source: str = "ILI", target: str = "", seeds: int = 10, rivals: str = "league", steps: int = 120,
         budget_eq: int = 100, item: str = "E4", model: str = MODEL, lr: float = 5e-5, anonymous: bool = False):
    """Train one policy per seed on ``source``; deploy each on ``target`` (source when empty)."""
    target = target or source
    tag = f"grpo_{rivals}{'_anon' if anonymous else ''}_{source}"
    train_jobs = [{"rows": _rows(source, HORIZON[source]), "dataset": source, "horizon": HORIZON[source], "seed": 100 + s,
                   "steps": steps, "episode_budget": budget_eq * 30, "rivals": rivals, "lr": lr, "model": model,
                   "anonymous": anonymous, "out_dir": f"{RUNS}/{item}/{tag}/seed{100 + s}"} for s in range(seeds)]
    method = f"{tag}_to_{target}"
    target_rows = _rows(target, HORIZON[target])
    search_jobs = [{"rows": target_rows, "dataset": target, "horizon": HORIZON[target], "budget": budget_eq * 30,
                    "method": method, "seeds": [s], "anonymous": anonymous, "model": model} for s in range(seeds)]
    plan = {"item": item, "name": f"{target}_T{HORIZON[target]}", "method": method, "train_jobs": train_jobs,
            "search_jobs": search_jobs}
    print("results in", driver.remote(plan))
