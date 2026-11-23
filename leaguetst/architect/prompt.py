"""Prompts for the architect and strict parsing of its JSON proposals.

The prompt states the task, the parameter cap, every field with its allowed
values, and the configs already evaluated in this search with their
validation MSE (best first). The model answers with one JSON object. With
``anonymous=True`` the field names are replaced by neutral letters, which
tests whether the model's prior knowledge of the names matters.
"""

from __future__ import annotations

import json
import re
from typing import Any

from leaguetst.search.feasible import BUDGETS, INPUT_LEN
from leaguetst.search.replay import MUTABLE_FIELDS, Space

TASKS = {
    "ETTh1": "hourly readings of an electricity transformer (7 variables, oil temperature is the target)",
    "ILI": "weekly influenza-like-illness statistics from the US CDC (7 variables)",
}
DESCRIPTIONS = {
    "patch_len": "time steps per patch",
    "stride": "step between patch starts: half or full patch length",
    "n_layers": "encoder layers",
    "n_heads": "attention heads",
    "d_model": "token embedding width",
    "ff_ratio": "feed-forward width as a multiple of d_model",
    "dropout": "dropout rate",
    "instance_norm": "reversible instance normalisation on or off",
}
ANONYMOUS = {name: f"field_{chr(ord('a') + i)}" for i, name in enumerate(MUTABLE_FIELDS)}
HISTORY_LINES = 24


def _value_text(value: Any) -> str:
    return json.dumps(value)


def build_messages(dataset: str, horizon: int, space: Space, history: list[tuple[str, float]],
                   anonymous: bool = False) -> list[dict[str, str]]:
    """Chat messages for one proposal. ``history`` holds (config_id, val MSE) pairs."""
    names = ANONYMOUS if anonymous else {f: f for f in MUTABLE_FIELDS}
    fields = []
    for f in MUTABLE_FIELDS:
        allowed = ", ".join(_value_text(v) for v in space.domains[f])
        meaning = "" if anonymous else f" ({DESCRIPTIONS[f]})"
        fields.append(f"- {names[f]}{meaning}: one of [{allowed}]")
    task = (
        f"Design a patch-based transformer forecaster for {TASKS[dataset]}. "
        f"It reads the last {INPUT_LEN[dataset]} steps and forecasts the next {horizon}. "
        f"It is trained with a fixed recipe and scored by validation MSE (lower is better). "
        f"Its parameter count must stay at or below {BUDGETS[(dataset, horizon)]:,}; "
        f"a configuration over the cap is replaced by the nearest one that fits before it is trained."
    )
    lines = [task, "", "Choose one value for every field:", *fields, ""]
    if history:
        ranked = sorted(history, key=lambda item: item[1])[:HISTORY_LINES]
        lines.append(f"Configurations already evaluated in this search ({len(history)} so far, best {len(ranked)} shown):")
        for cid, value in ranked:
            choice = space.choices[cid]
            config = {names[f]: choice[f] for f in MUTABLE_FIELDS}
            lines.append(f"{json.dumps(config)} -> {value:.4f}")
        lines.append("")
        lines.append("Propose one new configuration, different from those above, that you expect to score lower.")
    else:
        lines.append("Nothing has been evaluated yet. Propose the configuration you expect to score lowest.")
    order = ", ".join(names[f] for f in MUTABLE_FIELDS)
    lines.append(f"Answer with a single JSON object with the fields in this order: {order}. Nothing else.")
    return [
        {"role": "system", "content": "You are an expert in deep learning for time-series forecasting."},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_choice(text: str, space: Space, anonymous: bool = False) -> tuple[dict[str, Any] | None, str]:
    """The field values of the first JSON object in ``text``, or None with the reason."""
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return None, "no json"
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "bad json"
    if not isinstance(raw, dict):
        return None, "bad json"
    names = ANONYMOUS if anonymous else {f: f for f in MUTABLE_FIELDS}
    choice = {}
    for field in MUTABLE_FIELDS:
        key = names[field]
        if key not in raw:
            return None, f"missing {key}"
        value = raw[key]
        allowed = space.domains[field]
        if isinstance(value, str) and field != "stride":
            lowered = value.strip().lower()
            value = {"true": True, "false": False}.get(lowered, value)
            try:
                value = type(allowed[0])(value) if not isinstance(value, bool) else value
            except (TypeError, ValueError):
                return None, f"bad value for {key}"
        if field == "instance_norm" and not isinstance(value, bool):
            return None, f"bad value for {key}"
        if field != "instance_norm" and isinstance(value, bool):
            return None, f"bad value for {key}"
        if value not in allowed:
            return None, f"{key} out of range"
        choice[field] = value
    return choice, "ok"


def parse_proposal(text: str, space: Space, anonymous: bool = False) -> tuple[str | None, str]:
    """Config id of the first JSON object in ``text``, or None with the reason."""
    choice, reason = parse_choice(text, space, anonymous)
    if choice is None:
        return None, reason
    cid = space.to_id(choice)
    if cid is None:
        return None, "over the cap"
    return cid, "ok"
