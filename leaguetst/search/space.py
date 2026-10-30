"""The 30,720-configuration design space of the forecaster.

Every field of :class:`ForecasterConfig` takes one of a few values. Stride is
expressed relative to the patch length, as half or full, and resolved to an
integer when a config is built.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Iterator

from leaguetst.models.config import ForecasterConfig

SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "patch_len": (8, 12, 16, 24),
    "stride": ("half", "full"),
    "n_layers": (1, 2, 3, 4),
    "n_heads": (2, 4, 8, 16),
    "d_model": (16, 32, 64, 128, 256),
    "ff_ratio": (1, 2, 4),
    "dropout": (0.0, 0.1, 0.2, 0.3),
    "instance_norm": (True, False),
    "channel_mode": ("independent", "mixing"),
}


def space_size() -> int:
    """Number of configurations in the full grid."""
    return math.prod(len(values) for values in SEARCH_SPACE.values())


def resolve_choice(choice: dict[str, Any]) -> ForecasterConfig:
    """Turn a mapping of search-space values into a config.

    ``stride`` may be ``"half"``, ``"full"`` or an integer.
    """
    values = dict(choice)
    stride = values["stride"]
    if stride == "half":
        values["stride"] = values["patch_len"] // 2
    elif stride == "full":
        values["stride"] = values["patch_len"]
    return ForecasterConfig.from_dict(values)


def enumerate_configs() -> Iterator[ForecasterConfig]:
    """Yield every configuration of the grid in a fixed order."""
    names = list(SEARCH_SPACE)
    for combination in itertools.product(*(SEARCH_SPACE[n] for n in names)):
        yield resolve_choice(dict(zip(names, combination)))


def sample_config(rng: random.Random) -> ForecasterConfig:
    """Draw one configuration uniformly from the grid."""
    choice = {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}
    return resolve_choice(choice)
