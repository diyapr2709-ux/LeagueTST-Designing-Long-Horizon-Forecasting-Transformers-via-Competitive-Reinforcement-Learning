"""GPU side of the architect: constrained generation, search episodes, GRPO.

Runs inside a Modal container (transformers and peft are imported lazily).
``run_job`` is the single entry point, so several jobs can share one GPU in
separate processes.

GRPO here is a short custom loop: one on-policy update per group, so the
importance ratio is one and no clipping is needed. Log-probabilities are
taken under the same masked distribution the completions were sampled
from. Rivals enter the group baseline only and cost nothing:

* "league": among the 32 configs evaluated most recently in the episode,
  drawn with weights proportional to their probability under a random past
  snapshot of the policy (scored on the current prompt);
* "uniform": among the same 32 configs, drawn uniformly;
* "none": no rivals.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from leaguetst.architect.constrained import ConfigTrie, canonical_text
from leaguetst.architect.episode import EpisodeLog, fill_wasted, run_episode, score_group
from leaguetst.architect.prompt import build_messages
from leaguetst.search.replay import BudgetExhausted, Oracle, Space, Table

END_TOKEN = "<|im_end|>"
RIVAL_POOL = 32


def load(model_name: str, revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16).to("cuda")
    return tok, model


def make_trie(tok, space: Space, anonymous: bool) -> ConfigTrie:
    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    return ConfigTrie(lambda text: tok.encode(text, add_special_tokens=False), space, end_id, anonymous)


def prompt_ids(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")["input_ids"][0].to("cuda")


def sampling_config(tok, temperature: float, max_new_tokens: int):
    """Plain ancestral sampling: no top-k, top-p or repetition penalty.

    Built from scratch so the model's own generation_config.json (which for
    Qwen2.5-Instruct sets repetition_penalty 1.1, top_k 20, top_p 0.8) cannot
    change the distribution GRPO scores (amendment A20).
    """
    from transformers import GenerationConfig

    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    return GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, repetition_penalty=1.0,
                            max_new_tokens=max_new_tokens, eos_token_id=[end_id], pad_token_id=tok.pad_token_id)


def generate(tok, model, messages, n: int, temperature: float, trie: ConfigTrie | None, max_new_tokens: int = 112,
             return_scores: bool = False):
    import torch

    ids = prompt_ids(tok, messages)
    length = ids.shape[0]
    kwargs = {}
    if trie is not None:
        kwargs["prefix_allowed_tokens_fn"] = lambda batch_id, sent: trie.allowed(sent[length:].tolist())
    with torch.no_grad():
        out = model.generate(input_ids=ids.unsqueeze(0).repeat(n, 1), attention_mask=torch.ones(n, length, device="cuda", dtype=torch.long),
                             generation_config=sampling_config(tok, temperature, max_new_tokens),
                             output_scores=return_scores, return_dict_in_generate=True, **kwargs)
    completions = out.sequences[:, length:]
    texts = tok.batch_decode(completions, skip_special_tokens=True)
    if not return_scores:
        return ids, completions, texts
    # Log-probability of each sampled token under the processed scores the sampler drew from.
    steps = torch.stack(out.scores, dim=1).float().log_softmax(dim=-1)
    sampled = steps.gather(-1, completions[:, : steps.shape[1]].unsqueeze(-1)).squeeze(-1)
    return ids, completions, texts, sampled


def sequence_logp(net, tok, ids, completions, trie: ConfigTrie | None, with_grad: bool):
    """Per-token log-probabilities of ``completions`` after prompt ``ids`` under the sampling distribution."""
    import torch

    group, length = completions.shape[0], ids.shape[0]
    full = torch.cat([ids.unsqueeze(0).repeat(group, 1), completions], dim=1)
    mask = (completions != tok.pad_token_id).float()
    attention = torch.cat([torch.ones(group, length, device="cuda"), mask], dim=1)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        logits = net(input_ids=full, attention_mask=attention, logits_to_keep=completions.shape[1] + 1).logits[:, :-1].float()
        if trie is not None:
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            rows = completions.tolist()
            for i, row in enumerate(rows):
                for t in range(len(row)):
                    allowed[i, t, trie.allowed(row[:t])] = True
            logits = logits.masked_fill(~allowed, float("-inf"))
        chosen = logits.gather(-1, completions.unsqueeze(-1)).squeeze(-1)
        logp = chosen - torch.logsumexp(logits, dim=-1)
        logp = torch.where(mask > 0, logp, torch.zeros_like(logp))
    return logp, mask


def search(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Search episodes with the base model or a saved adapter, one per seed."""
    import torch

    tok, model = load(job["model"], job.get("revision"))
    if job.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, job["adapter"])
    model.eval()
    table = Table(job["rows"])
    space = Space(table.ids)
    anonymous = bool(job.get("anonymous", False))
    trie = make_trie(tok, space, anonymous) if job.get("constrained", True) else None
    results = []
    for seed in job["seeds"]:
        torch.manual_seed(seed)
        oracle = Oracle(table, int(job["budget"]), int(seed), float(job.get("noise_sd", 0.0)))
        start = time.time()
        log = run_episode(oracle, space, job["dataset"], int(job["horizon"]),
                          lambda messages, n: generate(tok, model, messages, n, float(job.get("temperature", 1.0)), trie)[2],
                          group=int(job.get("group", 8)), anonymous=anonymous)
        results.append({"method": job["method"], "seed": seed, "spent": oracle.spent, "lookups": len(oracle.charged),
                        "trace": oracle.trace, "charges": [list(c) for c in oracle.charges], "steps": log.steps,
                        "proposals": log.proposals, "invalid": log.invalid, "repaired": log.repaired,
                        "repeats": log.repeats, "reasons": log.reasons, "seconds": time.time() - start,
                        "gpu": torch.cuda.get_device_name(0), "noise_sd": float(job.get("noise_sd", 0.0))})
    return results


def train(job: dict[str, Any]) -> dict[str, Any]:
    """GRPO with LoRA on one table, in search episodes of ``episode_budget`` epochs."""
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    seed = int(job["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, base = load(job["model"], job.get("revision"))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    base.config.use_cache = False
    # No LoRA dropout: the update must score the same network that sampled (A20).
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lora)
    rivals = job.get("rivals", "none")
    if rivals == "league":
        model.add_adapter("rival", lora)
        model.set_adapter("default")
    params = [p for n, p in model.named_parameters() if p.requires_grad and ".rival." not in n]
    optimizer = torch.optim.AdamW(params, lr=float(job.get("lr", 1e-4)))
    table = Table(job["rows"])
    space = Space(table.ids)
    anonymous = bool(job.get("anonymous", False))
    trie = make_trie(tok, space, anonymous) if job.get("constrained", True) else None
    dataset, horizon = job["dataset"], int(job["horizon"])
    group, n_rivals, beta = int(job.get("group", 8)), int(job.get("n_rivals", 4)), float(job.get("beta", 0.04))
    noise = float(job.get("noise_sd", 0.0))
    episodes = 1
    oracle = Oracle(table, int(job["episode_budget"]), seed * 1000 + episodes, noise)
    history: dict[str, float] = {}
    snapshot = lambda: {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}
    snapshots = [snapshot()]
    rows, training_epochs = [], 0
    for step in range(1, int(job["steps"]) + 1):
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        messages = build_messages(dataset, horizon, space, list(history.items()), anonymous)
        model.eval()
        ids, completions, texts, sampled_logp = generate(tok, model, messages, group, 1.0, trie, return_scores=True)
        log = EpisodeLog()
        spent_before = oracle.spent
        try:
            rewards, cids = score_group(oracle, space, texts, history, anonymous, log, rng)
        except BudgetExhausted:
            training_epochs += oracle.spent - spent_before
            episodes += 1
            oracle = Oracle(table, int(job["episode_budget"]), seed * 1000 + episodes, noise)
            history = {}
            continue
        training_epochs += oracle.spent - spent_before
        rival_rewards: list[float] = []
        pool = list(history)[-RIVAL_POOL:]  # the most recently evaluated configs (amendment A19)
        if rivals in ("league", "uniform") and pool:
            if rivals == "league":
                set_peft_model_state_dict(model, snapshots[int(rng.integers(len(snapshots)))], adapter_name="rival")
                model.set_adapter("rival")
                scores = []
                try:
                    for first in range(0, len(pool), 16):
                        chunk = pool[first:first + 16]
                        encoded = [tok.encode(canonical_text(space.choices[c], anonymous), add_special_tokens=False)
                                   + [tok.convert_tokens_to_ids(END_TOKEN)] for c in chunk]
                        width = max(len(e) for e in encoded)
                        padded = torch.tensor([e + [tok.pad_token_id] * (width - len(e)) for e in encoded], device="cuda")
                        rival_logp, rival_mask = sequence_logp(model, tok, ids, padded, trie, with_grad=False)
                        scores += (rival_logp * rival_mask).sum(dim=1).tolist()
                finally:
                    model.set_adapter("default")
                weights = np.exp(np.array(scores) - max(scores))
                # Keep every pool member drawable so sampling without replacement cannot fail.
                weights = (weights + 1e-12) / (weights + 1e-12).sum()
            else:
                weights = None
            picks = rng.choice(len(pool), size=min(n_rivals, len(pool)), replace=False, p=weights)
            rival_rewards = [-math.log(history[pool[i]]) for i in picks]
        for reward, cid in zip(rewards, cids):
            if reward is not None and cid is not None:
                history[cid] = math.exp(-reward)
        filled = fill_wasted(rewards + rival_rewards)
        own, everyone = np.array(filled[:group]), np.array(filled)
        advantages = torch.tensor((own - everyone.mean()) / (everyone.std() + 1e-6), dtype=torch.float32, device="cuda")

        model.train()
        logp, mask = sequence_logp(model, tok, ids, completions, trie, with_grad=True)
        # On-policy check: the scored log-probabilities must match the sampler's.
        width = min(sampled_logp.shape[1], logp.shape[1])
        diff = (logp[:, :width].detach() - sampled_logp[:, :width]).abs()
        # Positions after the end-of-turn token are padding with -inf scores; leave them out.
        diff = torch.where(mask[:, :width] > 0, diff, torch.zeros_like(diff))
        onpolicy_gap = float(diff.max())
        onpolicy_mean = float(diff.sum() / mask[:, :width].sum().clamp(min=1))
        # bf16 cached generation and a fresh forward pass differ by up to about 0.25
        # nats on single tokens; a different sampler shifts many tokens (A22).
        if onpolicy_mean > 0.05:
            raise RuntimeError(f"sampled and scored log-probabilities differ by {onpolicy_mean:.3f} per token on average")
        with model.disable_adapter():
            ref_logp, _ = sequence_logp(model, tok, ids, completions, trie, with_grad=False)
        delta = (ref_logp - logp).clamp(-20, 20)
        kl = torch.exp(delta) - delta - 1.0
        per_token = -(advantages.unsqueeze(1) * logp) + beta * kl
        loss = ((per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        valid = [r for r in rewards if r is not None]
        rows.append({"step": step, "episode": episodes, "loss": float(loss), "kl": float((kl * mask).sum() / mask.sum()),
                     "reward_mean": float(np.mean(valid)) if valid else None, "invalid_rate": log.invalid / group,
                     "repaired_rate": log.repaired / group, "repeat_rate": log.repeats / group,
                     "unique_rate": len({c for c in cids if c}) / group, "rivals": len(rival_rewards),
                     "best_in_history": min(history.values()) if history else None, "onpolicy_gap": onpolicy_gap, "onpolicy_mean": onpolicy_mean,
                     "seconds": time.time() - start,
                     "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "training_epochs": training_epochs})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        if step % int(job.get("snapshot_every", 5)) == 0:
            snapshots.append(snapshot())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if rivals == "league":
        model.delete_adapter("rival")
    model.save_pretrained(str(out_dir / "adapter"))
    return {"adapter": str(out_dir / "adapter"), "log": rows, "training_epochs": training_epochs, "episodes": episodes,
            "seconds": float(sum(r["seconds"] for r in rows)), "gpu": torch.cuda.get_device_name(0),
            "job": {k: v for k, v in job.items() if k != "rows"}}


def run_job(kind: str, job: dict[str, Any]) -> Any:
    """Entry point for a worker process: ``kind`` is "train" or "search"."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        result = train(job) if kind == "train" else search(job)
    except Exception as error:  # reported back to the driver, which logs it
        return {"error": repr(error), "kind": kind, "job": {k: v for k, v in job.items() if k != "rows"}}
    env = environment(job)
    for record in [result] if isinstance(result, dict) else result:
        record["env"] = env
    return result


PACKAGES = ("torch", "transformers", "peft", "accelerate", "huggingface_hub", "numpy", "scipy", "scikit-learn")


def environment(job: dict[str, Any]) -> dict[str, Any]:
    """Code, packages, model snapshot and GPU sharing behind a record (CLAUDE.md rule 7)."""
    from importlib.metadata import version

    import torch

    return {"git_hash": job.get("git_hash"), "model_revision": job.get("revision"), "per_gpu": job.get("per_gpu"),
            "backend": job.get("backend"), "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "versions": {name: version(name) for name in PACKAGES}}
