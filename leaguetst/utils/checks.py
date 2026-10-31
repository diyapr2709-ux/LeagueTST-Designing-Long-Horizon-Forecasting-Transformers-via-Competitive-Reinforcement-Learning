"""Type and range checks for values that arrive from YAML, JSON or a model.

Configs will later be written by a language model, so a value is checked
for its type as well as its range: a string ``"false"`` must not pass as a
boolean, and ``True`` or ``16.5`` must not pass as an integer.
"""

from __future__ import annotations

import math
import numbers
from typing import Any


def require_int(name: str, value: Any, low: int | None = None, high: int | None = None) -> int:
    """Return ``value`` as an int, or raise ValueError if it is not one in range."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must lie in [{low}, {high}], got {value}")
    return value


def require_real(name: str, value: Any, low: float | None = None, high: float | None = None, high_open: bool = False) -> float:
    """Return ``value`` as a finite float in range, or raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    too_high = high is not None and (value >= high if high_open else value > high)
    if (low is not None and value < low) or too_high:
        bracket = ")" if high_open else "]"
        raise ValueError(f"{name} must lie in [{low}, {high}{bracket}, got {value}")
    return value


def require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value
