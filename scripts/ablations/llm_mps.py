"""LLM ablation arms on an Apple-silicon GPU (torch MPS backend). Amendment AB3.

The E4 code (leaguetst/architect/llm.py) is frozen and places every tensor on
CUDA, so it cannot run on a Mac. This file is a separate implementation of the
same search episode and the same GRPO step. It imports every rule that decides
a result from the frozen code (prompt, trie, episode scoring, Oracle, table
rows, observation noise) and replaces only the tensor work:

* the group shares one prompt, so the prompt is run once and its key-value
  cache is repeated over the group, in sampling, in rival scoring and in the
  update (the gradient flows through the repeated cache). The frozen code runs
  the prompt once per group member; both compute the same quantity;
* sampling is plain ancestral sampling at temperature 1.0 on the CPU from a
  seeded generator (A20: no top-k, no top-p, no repetition penalty);
* no gradient checkpointing (48 GB of shared memory).

Because backend and code differ from the registered arms, everything this file
writes goes to items named ``ABL_*_mps`` under results/ablations/mps/, with its
own references on the same Mac, and is labelled exploratory (AB1: an AB-G arm
run under other digests is exploratory).

A 120-step policy passes through its own 30- and 60-step states, so the step
trend trains each policy once and saves the adapter at 30, 60 and 120 steps.

    .venv-llm/bin/python scripts/ablations/llm_mps.py prompted --dataset ILI --seeds 30 --item ABL_P_ref_mps
    .venv-llm/bin/python scripts/ablations/llm_mps.py steps --source ETTh1 --target ILI --policies 10 --item ABL_G_steps_mps

A relaunch keeps finished searches and finished policies, and never retrains a
policy that the on-policy check stopped (A22).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaguetst.architect import runner  # noqa: E402
from leaguetst.architect.constrained import ConfigTrie, canonical_text  # noqa: E402
from leaguetst.architect.episode import EpisodeLog, fill_wasted, run_episode, score_group  # noqa: E402
from leaguetst.architect.llm import END_TOKEN, PACKAGES, RIVAL_POOL, make_trie  # noqa: E402
from leaguetst.architect.prompt import build_messages  # noqa: E402
from leaguetst.search.replay import BudgetExhausted, Oracle, Space, Table  # noqa: E402

DEVICE = "mps"
OUT = ROOT / "results" / "ablations" / "mps"
MAX_NEW_TOKENS = 112
CHECKPOINTS = (30, 60, 120)
ON_POLICY_LIMIT = 0.05  # A22


# ---------------------------------------------------------------- model work

def load(model_name: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.bfloat16).to(DEVICE)
    return tok, model


def prompt_ids(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")["input_ids"][0].to(DEVICE)


def prefill(net, ids, n: int):
    """Logits after the prompt (one row) and its key-value cache repeated ``n`` times."""
    out = net(input_ids=ids.unsqueeze(0), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    cache.batch_repeat_interleave(n)
    return out.logits[0, -1].float(), cache


def _draw(logits, allowed: list[int] | None, generator) -> tuple[int, float]:
    """Sample one token from the (masked) softmax of CPU ``logits``; return it and its log-probability."""
    import torch

    if allowed is not None:
        index = torch.tensor(allowed)
        logp = logits[index].log_softmax(dim=-1)
        i = int(torch.multinomial(logp.exp(), 1, generator=generator))
        return int(index[i]), float(logp[i])
    logp = logits.log_softmax(dim=-1)
    i = int(torch.multinomial(logp.exp(), 1, generator=generator))
    return i, float(logp[i])


def generate(tok, model, messages, n: int, trie: ConfigTrie | None, generator, max_new_tokens: int = MAX_NEW_TOKENS):
    """``n`` completions for one prompt: ids, padded completions, texts, per-token log-probabilities."""
    import torch

    ids = prompt_ids(tok, messages)
    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    rows: list[list[int]] = [[] for _ in range(n)]
    logps: list[list[float]] = [[] for _ in range(n)]
    done = [False] * n
    with torch.no_grad():
        first, cache = prefill(model, ids, n)
        step_logits = first.cpu().unsqueeze(0).repeat(n, 1)
        for _ in range(max_new_tokens):
            for r in range(n):
                if done[r]:
                    continue
                token, lp = _draw(step_logits[r], trie.allowed(rows[r]) if trie is not None else None, generator)
                rows[r].append(token)
                logps[r].append(lp)
                done[r] = token == end_id
            if all(done):
                break
            nxt = torch.tensor([[tok.pad_token_id if done[r] else rows[r][-1]] for r in range(n)], device=DEVICE)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            step_logits = out.logits[:, -1].float().cpu()
    width = max(len(r) for r in rows)
    completions = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r in rows], device=DEVICE)
    sampled = torch.tensor([lp + [0.0] * (width - len(lp)) for lp in logps])
    texts = [tok.decode(r, skip_special_tokens=True) for r in rows]
    return ids, completions, texts, sampled


def _parts(net):
    """The decoder (hidden states) and the output embedding of a plain or LoRA-wrapped causal LM."""
    lm = net.get_base_model() if hasattr(net, "get_base_model") else net
    return lm.model, lm.lm_head


def sequence_logp(net, tok, ids, completions, trie: ConfigTrie | None, with_grad: bool):
    """Per-token log-probabilities of ``completions`` after prompt ``ids`` under the sampling distribution.

    Under a trie only the allowed tokens enter the softmax, so only their logits are computed.
    """
    import torch

    group, width = completions.shape
    mask = (completions != tok.pad_token_id).float()
    decoder, head = _parts(net)
    with torch.enable_grad() if with_grad else torch.no_grad():
        out = decoder(input_ids=ids.unsqueeze(0), use_cache=True)
        cache = out.past_key_values
        cache.batch_repeat_interleave(group)
        hidden = out.last_hidden_state[:, -1:].expand(group, 1, -1)
        if width > 1:
            rest = decoder(input_ids=completions[:, :-1], past_key_values=cache, use_cache=True).last_hidden_state
            hidden = torch.cat([hidden, rest], dim=1)
        if trie is None:
            logits, target = head(hidden).float(), completions
        else:
            options = [[trie.allowed(row[:t]) for t in range(width)] for row in completions.tolist()]
            tokens = sorted({a for row in options for allowed in row for a in allowed} | set(completions.flatten().tolist()))
            column = {t: i for i, t in enumerate(tokens)}
            logits = (hidden @ head.weight[torch.tensor(tokens, device=DEVICE)].T).float()
            allowed_mask = torch.zeros(logits.shape, dtype=torch.bool)
            for i, row in enumerate(options):
                for t, allowed in enumerate(row):
                    allowed_mask[i, t, [column[a] for a in allowed]] = True
            logits = logits.masked_fill(~allowed_mask.to(DEVICE), float("-inf"))
            target = torch.tensor([[column[t] for t in row] for row in completions.tolist()], device=DEVICE)
        chosen = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        logp = chosen - torch.logsumexp(logits, dim=-1)
        logp = torch.where(mask > 0, logp, torch.zeros_like(logp))
    return logp, mask


# ------------------------------------------------------------ search and GRPO

def search(job: dict[str, Any], tok=None, model=None) -> dict[str, Any]:
    """One search episode with the base model or an adapter already loaded."""
    import torch

    model.eval()
    table = Table(job["rows"])
    space = Space(table.ids)
    anonymous = bool(job.get("anonymous", False))
    trie = make_trie(tok, space, anonymous) if job.get("constrained", True) else None
    seed = int(job["seed"])
    generator = torch.Generator().manual_seed(seed)
    oracle = Oracle(table, int(job["budget"]), seed, float(job["noise_sd"]))
    start = time.time()
    marks: list[int] = []  # charges made before each step, so a step's rewards can be recovered

    def propose(messages, n):
        marks.append(len(oracle.charges))
        return generate(tok, model, messages, n, trie, generator)[2]

    log = run_episode(oracle, space, job["dataset"], int(job["horizon"]), propose, group=8, anonymous=anonymous)
    return {"method": job["method"], "seed": seed, "spent": oracle.spent, "lookups": len(oracle.charged),
            "trace": oracle.trace, "charges": [list(c) for c in oracle.charges], "steps": log.steps,
            "proposals": log.proposals, "invalid": log.invalid, "repaired": log.repaired, "repeats": log.repeats,
            "reasons": log.reasons, "step_marks": marks, "seconds": time.time() - start, "noise_sd": float(job["noise_sd"])}


def train(job: dict[str, Any], save) -> dict[str, Any]:
    """GRPO with LoRA on one table, the E4 recipe; ``save(model, step)`` is called at every checkpoint."""
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    seed = int(job["seed"])
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, base = load(job["model"], job["revision"])
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
    trie = make_trie(tok, space, False)
    dataset, horizon = job["dataset"], int(job["horizon"])
    group, n_rivals, beta = 8, 4, float(job.get("beta", 0.04))
    noise = float(job["noise_sd"])
    episodes = 1
    oracle = Oracle(table, int(job["episode_budget"]), seed * 1000 + episodes, noise)
    history: dict[str, float] = {}

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}

    snapshots = [snapshot()]
    rows, training_epochs = [], 0
    for step in range(1, int(job["steps"]) + 1):
        start = time.time()
        messages = build_messages(dataset, horizon, space, list(history.items()), False)
        model.eval()
        ids, completions, texts, sampled_logp = generate(tok, model, messages, group, trie, generator)
        log = EpisodeLog()
        spent_before = oracle.spent
        exhausted = False
        try:
            rewards, cids = score_group(oracle, space, texts, history, False, log, rng)
        except BudgetExhausted:
            exhausted = True
        training_epochs += oracle.spent - spent_before
        if exhausted:
            episodes += 1
            oracle = Oracle(table, int(job["episode_budget"]), seed * 1000 + episodes, noise)
            history = {}
        else:
            rival_rewards: list[float] = []
            pool = list(history)[-RIVAL_POOL:]
            if rivals in ("league", "uniform") and pool:
                if rivals == "league":
                    set_peft_model_state_dict(model, snapshots[int(rng.integers(len(snapshots)))], adapter_name="rival")
                    model.set_adapter("rival")
                    try:
                        encoded = [tok.encode(canonical_text(space.choices[c], False), add_special_tokens=False)
                                   + [tok.convert_tokens_to_ids(END_TOKEN)] for c in pool]
                        width = max(len(e) for e in encoded)
                        padded = torch.tensor([e + [tok.pad_token_id] * (width - len(e)) for e in encoded], device=DEVICE)
                        rival_logp, rival_mask = sequence_logp(model, tok, ids, padded, trie, with_grad=False)
                        scores = (rival_logp * rival_mask).sum(dim=1).tolist()
                    finally:
                        model.set_adapter("default")
                    weights = np.exp(np.array(scores) - max(scores))
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
            advantages = torch.tensor((own - everyone.mean()) / (everyone.std() + 1e-6), dtype=torch.float32, device=DEVICE)

            model.train()
            logp, mask = sequence_logp(model, tok, ids, completions, trie, with_grad=True)
            diff = (logp.detach().cpu() - sampled_logp).abs()
            mask_cpu = mask.cpu()
            diff = torch.where(mask_cpu > 0, diff, torch.zeros_like(diff))
            onpolicy_gap = float(diff.max())
            onpolicy_mean = float(diff.sum() / mask_cpu.sum().clamp(min=1))
            if onpolicy_mean > ON_POLICY_LIMIT:
                raise RuntimeError(f"sampled and scored {runner.ON_POLICY_STOP} by {onpolicy_mean:.3f} per token on average")
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
            rows.append({"step": step, "episode": episodes, "loss": float(loss.detach()),
                         "kl": float((kl * mask).sum().detach() / mask.sum()),
                         "reward_mean": float(np.mean(valid)) if valid else None, "invalid_rate": log.invalid / group,
                         "repaired_rate": log.repaired / group, "repeat_rate": log.repeats / group,
                         "unique_rate": len({c for c in cids if c}) / group, "rivals": len(rival_rewards),
                         "best_in_history": min(history.values()) if history else None, "onpolicy_gap": onpolicy_gap,
                         "onpolicy_mean": onpolicy_mean, "seconds": time.time() - start,
                         "training_epochs": training_epochs,
                         "driver_gb": torch.mps.driver_allocated_memory() / 2**30})
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            if step % 5 == 0:
                snapshots.append(snapshot())
        # A checkpoint is taken after the step's update, or after the step that exhausted an episode.
        if step in CHECKPOINTS:
            save(model, step)
        torch.mps.empty_cache()
    return {"log": rows, "training_epochs": training_epochs, "episodes": episodes,
            "seconds": float(sum(r["seconds"] for r in rows)), "job": {k: v for k, v in job.items() if k != "rows"}}


# -------------------------------------------------------------- plans, records

def environment(job: dict[str, Any]) -> dict[str, Any]:
    import platform
    from importlib.metadata import version

    chip = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
    return {"git_hash": job["git_hash"], "model_revision": job["revision"], "backend": "mps", "per_gpu": 1,
            "gpu_name": f"{chip} (MPS)", "macos": platform.mac_ver()[0],
            "versions": {name: version(name) for name in PACKAGES}}


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def base_job(dataset: str) -> dict[str, Any]:
    rows = runner.table_rows(dataset)
    return {"rows": rows, "dataset": dataset, "horizon": runner.HORIZON[dataset], "noise_sd": runner.noise_sd(rows),
            "model": runner.MODEL, "revision": runner.REVISION, "git_hash": runner.git_hash()}


def prompted(args) -> None:
    job = {**base_job(args.dataset), "budget": args.budget_eq * 30, "anonymous": args.anonymous,
           "constrained": args.constrained}
    if args.noise_free:  # AB-Q: lookups are 3-seed means
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plateau import noise_free_rows

        job.update(rows=noise_free_rows(job["rows"]), noise_sd=0.0)
    method = "qwen_prompted" + ("_anon" if args.anonymous else "") + ("" if args.constrained else "_repair") + "_mps"
    out = OUT / args.item / f"{args.dataset}_T{job['horizon']}"
    done = runner.deployed(out)
    todo = [s for s in range(args.seeds) if (method, s) not in done]
    if not todo:
        return
    tok, model = load(job["model"], job["revision"])
    for s in todo:
        record = {**search({**job, "method": method, "seed": s}, tok, model), "env": environment(job)}
        append(out / f"{method}_runs.jsonl", record)
        print(f"{args.item} {args.dataset} seed {s}: {record['steps']} steps, {record['lookups']} lookups, "
              f"{record['seconds']:.0f}s", flush=True)


def steps(args) -> None:
    """Train each policy once to 120 steps, keep the adapter at 30, 60 and 120, deploy each on the target."""
    from peft import PeftModel

    source, target = base_job(args.source), base_job(args.target)
    out = OUT / args.item / f"{args.target}_T{target['horizon']}"
    adapters = ROOT / "results" / "runs" / args.item / "adapters"
    stopped = runner.stopped_runs(out)
    finished = {r["job"]["run_name"] for r in runner._read(out / "training.jsonl")}
    for p in range(args.policies):
        run_name = f"grpo_league_{args.source}_seed{100 + p}"
        job = {**source, "seed": 100 + p, "steps": max(CHECKPOINTS), "episode_budget": args.budget_eq * 30,
               "rivals": "league", "lr": 1e-4, "beta": 0.04, "run_name": run_name}
        if run_name not in finished and run_name not in stopped:
            def save(model, step, run_name=run_name):
                model.save_pretrained(str(adapters / f"{run_name}_steps{step}"), selected_adapters=["default"])

            try:
                record = {**train(job, save), "env": environment(job)}
            except Exception as error:
                append(out / "errors.jsonl", {"phase": "train", "error": repr(error), "run_names": [run_name]})
                print(f"{run_name}: {error!r}", flush=True)
                continue
            append(out / "training.jsonl", record)
        done = runner.deployed(out)
        for step in CHECKPOINTS:
            method = f"grpo_league_from_{args.source}_steps{step}_mps"
            path = adapters / f"{run_name}_steps{step}"
            if (method, p) in done or not (path / "adapter_config.json").exists():
                continue
            tok, base = load(job["model"], job["revision"])
            model = PeftModel.from_pretrained(base, str(path))
            record = {**search({**target, "budget": args.budget_eq * 30, "method": method, "seed": p}, tok, model),
                      "env": environment(job)}
            append(out / f"{method}_runs.jsonl", record)
            print(f"{method} seed {p}: {record['steps']} steps, {record['seconds']:.0f}s", flush=True)
            del model, base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prompted")
    a.add_argument("--dataset", required=True, choices=sorted(runner.HORIZON))
    a.add_argument("--seeds", type=int, default=30)
    a.add_argument("--anonymous", action="store_true")
    a.add_argument("--no-constrained", dest="constrained", action="store_false")
    a.add_argument("--noise-free", action="store_true", help="AB-Q: search the table of 3-seed means")
    a.add_argument("--item", required=True)
    a.add_argument("--budget-eq", type=int, default=100)
    b = sub.add_parser("steps")
    b.add_argument("--source", default="ETTh1")
    b.add_argument("--target", default="ILI")
    b.add_argument("--policies", type=int, default=10)
    b.add_argument("--item", default="ABL_G_steps_mps")
    b.add_argument("--budget-eq", type=int, default=100)
    args = parser.parse_args()
    if not args.item.endswith("_mps"):
        parser.error("items run on this backend end in _mps (AB3)")
    if getattr(args, "noise_free", False) and not args.item.startswith("ABL_Q_"):
        parser.error("--noise-free runs belong to ABL_Q items only")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prompted(args) if args.command == "prompted" else steps(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
