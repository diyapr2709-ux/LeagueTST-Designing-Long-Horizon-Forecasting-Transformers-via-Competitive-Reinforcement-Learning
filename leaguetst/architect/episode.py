"""A search episode driven by an LLM, charged through the table oracle.

Each step builds the prompt from the episode's history, draws a group of
proposals, and looks up every valid new config (charged once, like any
other method). A proposal over the parameter cap is replaced by the nearest
feasible config not yet in the history, exactly as TPE's proposals are, and
counted as repaired. Unparseable or out-of-space proposals, and proposals
already in the history, score the group minimum minus one and cost nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from leaguetst.architect.prompt import build_messages, parse_choice
from leaguetst.search.replay import BudgetExhausted, Oracle, Space, nearest_unseen

Proposer = Callable[[list[dict[str, str]], int], list[str]]


@dataclass
class EpisodeLog:
    steps: int = 0
    proposals: int = 0
    invalid: int = 0
    repaired: int = 0
    repeats: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


def score_group(oracle: Oracle, space: Space, texts: list[str], history: dict[str, float], anonymous: bool,
                log: EpisodeLog, rng: np.random.Generator) -> tuple[list[float | None], list[str | None]]:
    """Rewards (-log val MSE, or None for a wasted proposal) and config ids for one group."""
    rewards: list[float | None] = []
    ids: list[str | None] = []
    for text in texts:
        choice, reason = parse_choice(text, space, anonymous)
        log.proposals += 1
        cid = space.to_id(choice) if choice is not None else None
        if choice is not None and cid is None:
            reason = "over the cap, repaired"
            cid = nearest_unseen(choice, space, set(history), rng)
            log.repaired += 1
        log.reasons[reason] = log.reasons.get(reason, 0) + 1
        if cid is None:
            log.invalid += 1
            rewards.append(None)
            ids.append(None)
            continue
        if cid in history:
            log.repeats += 1
            rewards.append(None)
            ids.append(cid)
            continue
        value = oracle.full(cid)
        rewards.append(-math.log(value))
        ids.append(cid)
    return rewards, ids


def fill_wasted(rewards: list[float | None]) -> list[float]:
    valid = [r for r in rewards if r is not None]
    floor = (min(valid) if valid else 0.0) - 1.0
    return [floor if r is None else r for r in rewards]


def run_episode(oracle: Oracle, space: Space, dataset: str, horizon: int, propose: Proposer, group: int = 8,
                anonymous: bool = False, max_idle_steps: int = 20) -> EpisodeLog:
    """Search until the budget is spent, or until ``max_idle_steps`` steps add nothing new."""
    log = EpisodeLog()
    history: dict[str, float] = {}
    rng = np.random.default_rng(oracle.seed + 7_000_001)
    idle = 0
    try:
        while idle < max_idle_steps:
            messages = build_messages(dataset, horizon, space, list(history.items()), anonymous)
            texts = propose(messages, group)
            before = len(history)
            rewards, ids = score_group(oracle, space, texts, history, anonymous, log, rng)
            for reward, cid in zip(rewards, ids):
                if reward is not None and cid is not None:
                    history[cid] = math.exp(-reward)
            log.steps += 1
            idle = idle + 1 if len(history) == before else 0
    except BudgetExhausted:
        pass
    return log
