"""The wide design space: the current grid plus lookback and head type.

Every field of the 30,720-config grid keeps its values, and two fields are
added: the head (flatten, pooled, low-rank at three ranks, shared linear)
and the lookback, whose values depend on the dataset. Feasibility uses the
same caps as :mod:`leaguetst.search.feasible`.

The grid has 737,280 configs on ETTh1 and 552,960 on ILI, too many to build
one object each just to count them. Parameter counts depend only on the
shape fields (everything except heads, dropout and instance norm), so the
counts are computed in closed form with NumPy over the shape grid and
multiplied by the number of training settings per shape. With the flatten
head and each dataset's default lookback the counts equal those of
:func:`leaguetst.search.feasible.feasible_table`.
"""

from __future__ import annotations

import itertools
import math
import random
from functools import lru_cache
from typing import Any, Iterator, Sequence

import numpy as np

from leaguetst.models.wide import HEAD_TYPES, WideConfig
from leaguetst.search.feasible import BUDGETS, INPUT_LEN, N_VARS, SETTINGS, config_id
from leaguetst.search.space import SEARCH_SPACE, resolve_choice

HEAD_CHOICES: tuple[tuple[str, int], ...] = (
    ("flatten", 0),
    ("pooled", 0),
    ("lowrank", 4),
    ("lowrank", 8),
    ("lowrank", 16),
    ("shared_linear", 0),
)
LOOKBACKS: dict[str, tuple[int, ...]] = {"ETTh1": (96, 192, 336, 512), "ILI": (52, 104, 156)}
DATASETS = tuple(LOOKBACKS)
HORIZONS: dict[str, tuple[int, ...]] = {ds: tuple(t for d, t in SETTINGS if d == ds) for ds in DATASETS}
HEAD_CODES = {"flatten": "F", "pooled": "P", "lowrank": "R", "shared_linear": "S"}


def head_label(head_type: str, head_rank: int) -> str:
    return f"lowrank r={head_rank}" if head_type == "lowrank" else head_type


def wide_space(dataset: str) -> dict[str, tuple[Any, ...]]:
    """Field values of the wide grid for ``dataset``."""
    return {**SEARCH_SPACE, "head": HEAD_CHOICES, "input_len": LOOKBACKS[dataset]}


def wide_space_size(dataset: str) -> int:
    return math.prod(len(values) for values in wide_space(dataset).values())


def resolve_wide_choice(choice: dict[str, Any]) -> WideConfig:
    """Turn a mapping of wide-grid values into a config.

    ``head`` is a ``(head_type, head_rank)`` pair and ``stride`` may be
    ``"half"``, ``"full"`` or an integer.
    """
    values = dict(choice)
    head_type, head_rank = values.pop("head")
    input_len = values.pop("input_len")
    backbone = resolve_choice(values)
    return WideConfig(**backbone.to_dict(), input_len=input_len, head_type=head_type, head_rank=head_rank)


def sample_wide_config(dataset: str, rng: random.Random) -> WideConfig:
    """Draw one configuration uniformly from the wide grid."""
    return resolve_wide_choice({name: rng.choice(values) for name, values in wide_space(dataset).items()})


def wide_config_id(config: WideConfig) -> str:
    """The base ``config_id`` followed by lookback and head, e.g. ``p16s8l3h4d16f8r3n1iL336R8``."""
    rank = str(config.head_rank) if config.head_type == "lowrank" else ""
    return f"{config_id(config.backbone())}L{config.input_len}{HEAD_CODES[config.head_type]}{rank}"


def parameter_parts(
    patch_len, stride, n_layers, d_model, ff_ratio, mixing, head_code, head_rank, input_len, horizon, n_vars=N_VARS
):
    """Backbone and head parameter counts in closed form.

    Works on plain integers and on NumPy integer arrays alike; ``head_code``
    indexes ``HEAD_TYPES``. The backbone is the patch projection, the
    positional table and the encoder layers (four d x d projections with
    bias, two batch norms, two feed-forward linears). RevIN has no weights.
    """
    d = np.asarray(d_model, dtype=np.int64)
    d_ff = np.asarray(ff_ratio, dtype=np.int64) * d
    n_patches = (np.asarray(input_len, dtype=np.int64) - patch_len) // stride + 2
    width = np.where(mixing, n_vars, 1)
    in_features = patch_len * width
    out = horizon * width
    per_layer = 4 * (d * d + d) + 2 * (2 * d) + (d * d_ff + d_ff) + (d_ff * d + d)
    backbone = n_layers * per_layer + in_features * d + d + n_patches * d
    code = np.asarray(head_code)
    head = np.select(
        [code == HEAD_TYPES.index(name) for name in ("flatten", "pooled", "lowrank", "shared_linear")],
        [
            d * n_patches * out + out,
            d * out + out,
            head_rank * (d * n_patches + out) + out,
            d * out + out + n_patches * out,
        ],
        default=-1,
    )
    if np.any(head < 0):
        raise ValueError("unknown head code")
    return backbone, head


def wide_parameters(config: WideConfig, horizon: int, n_vars: int = N_VARS) -> int:
    """Trainable parameters of ``WideTST(config, n_vars, horizon)``."""
    backbone, head = parameter_parts(
        config.patch_len, config.stride, config.n_layers, config.d_model, config.ff_ratio,
        config.channel_mode == "mixing", HEAD_TYPES.index(config.head_type), config.head_rank,
        config.input_len, horizon, n_vars,
    )
    return int(backbone + head)


def wide_fits(config: WideConfig, dataset: str, horizon: int) -> bool:
    """Whether ``config`` is within the parameter cap at one setting."""
    return wide_parameters(config, horizon) <= BUDGETS[(dataset, horizon)]


def settings_per_width(d_model: int) -> int:
    """Training settings (heads, dropout, instance norm) valid at one width."""
    heads = sum(d_model % h == 0 for h in SEARCH_SPACE["n_heads"])
    return heads * len(SEARCH_SPACE["dropout"]) * len(SEARCH_SPACE["instance_norm"])


@lru_cache(maxsize=None)
def shape_grid(dataset: str) -> dict[str, np.ndarray]:
    """Every shape of the wide grid as read-only columns.

    A shape fixes every field except heads, dropout and instance norm, so it
    fixes the parameter count. ``settings`` is the number of configs that
    share the shape. The lookback is the last field, so the shapes of one
    lookback come in the same order on both datasets.
    """
    axes = [SEARCH_SPACE[n] for n in ("patch_len", "stride", "n_layers", "d_model", "ff_ratio", "channel_mode")]
    axes += [range(len(HEAD_CHOICES)), LOOKBACKS[dataset]]
    rows = np.array(
        [
            (p, p // 2 if s == "half" else p, layers, d, f, mode == "mixing", head, length)
            for p, s, layers, d, f, mode, head, length in itertools.product(*axes)
        ],
        dtype=np.int64,
    )
    columns = {
        "patch_len": rows[:, 0],
        "stride": rows[:, 1],
        "n_layers": rows[:, 2],
        "d_model": rows[:, 3],
        "ff_ratio": rows[:, 4],
        "mixing": rows[:, 5].astype(bool),
        "head": rows[:, 6],
        "head_code": np.array([HEAD_TYPES.index(HEAD_CHOICES[h][0]) for h in rows[:, 6]], dtype=np.int64),
        "head_rank": np.array([HEAD_CHOICES[h][1] for h in rows[:, 6]], dtype=np.int64),
        "input_len": rows[:, 7],
    }
    columns["settings"] = np.array([settings_per_width(int(d)) for d in columns["d_model"]], dtype=np.int64)
    for column in columns.values():
        column.flags.writeable = False
    return columns


def grid_parameter_parts(dataset: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Backbone and head parameter counts of every shape at one horizon."""
    g = shape_grid(dataset)
    return parameter_parts(
        g["patch_len"], g["stride"], g["n_layers"], g["d_model"], g["ff_ratio"], g["mixing"],
        g["head_code"], g["head_rank"], g["input_len"], horizon,
    )


def grid_parameters(dataset: str, horizon: int) -> np.ndarray:
    backbone, head = grid_parameter_parts(dataset, horizon)
    return backbone + head


def feasible_mask(dataset: str, horizons: Sequence[int] | None = None) -> np.ndarray:
    """Shapes of ``dataset`` within the cap at every horizon in ``horizons``.

    ``None`` means all four horizons of the dataset, the set a winner must
    fit to be retrained at every horizon.
    """
    horizons = HORIZONS[dataset] if horizons is None else tuple(horizons)
    mask = np.ones(len(shape_grid(dataset)["d_model"]), dtype=bool)
    for horizon in horizons:
        mask &= grid_parameters(dataset, horizon) <= BUDGETS[(dataset, horizon)]
    return mask


def shared_mask(dataset: str = DATASETS[0]) -> np.ndarray:
    """Shapes at each dataset's default lookback that fit all eight settings.

    The mask indexes the shapes of ``dataset`` at its default lookback (336
    or 104). With the flatten head it selects the 74 architectures of
    today's ``feasible_all``.
    """
    masks = []
    reference = None
    for name in DATASETS:
        g = shape_grid(name)
        rows = g["input_len"] == INPUT_LEN[name]
        fields = np.stack([g[n][rows] for n in ("patch_len", "stride", "n_layers", "d_model", "ff_ratio", "mixing", "head")])
        if reference is None:
            reference = fields
        elif not np.array_equal(reference, fields):
            raise AssertionError("shape grids of the two datasets are not aligned")
        masks.append(feasible_mask(name)[rows])
    g = shape_grid(dataset)
    out = np.zeros(len(g["d_model"]), dtype=bool)
    out[g["input_len"] == INPUT_LEN[dataset]] = np.logical_and.reduce(masks)
    return out


def _shape_config(g: dict[str, np.ndarray], i: int, n_heads: int, dropout: float, instance_norm: bool) -> WideConfig:
    head_type, head_rank = HEAD_CHOICES[int(g["head"][i])]
    return WideConfig(
        patch_len=int(g["patch_len"][i]),
        stride=int(g["stride"][i]),
        n_layers=int(g["n_layers"][i]),
        n_heads=n_heads,
        d_model=int(g["d_model"][i]),
        ff_ratio=int(g["ff_ratio"][i]),
        dropout=dropout,
        instance_norm=instance_norm,
        channel_mode="mixing" if g["mixing"][i] else "independent",
        input_len=int(g["input_len"][i]),
        head_type=head_type,
        head_rank=head_rank,
    )


def _settings(d_model: int) -> list[tuple[int, float, bool]]:
    return [
        (h, r, n)
        for h, r, n in itertools.product(SEARCH_SPACE["n_heads"], SEARCH_SPACE["dropout"], SEARCH_SPACE["instance_norm"])
        if d_model % h == 0
    ]


def iter_feasible_configs(dataset: str, mask: np.ndarray | None = None) -> Iterator[WideConfig]:
    """Yield the configs of every shape selected by ``mask`` (default: all four horizons)."""
    g = shape_grid(dataset)
    mask = feasible_mask(dataset) if mask is None else mask
    for i in np.flatnonzero(mask):
        for setting in _settings(int(g["d_model"][i])):
            yield _shape_config(g, int(i), *setting)


def summarize(dataset: str, mask: np.ndarray) -> dict[str, Any]:
    """Config and architecture counts of the shapes selected by ``mask``.

    An architecture is a shape: patch length, stride, layers, width, ff
    ratio, channel mode, head and lookback. Heads, dropout and instance norm
    are training settings, as in docs/feasible_set.md.
    """
    g = shape_grid(dataset)
    settings = g["settings"]

    def counts(column: np.ndarray, labels: dict[int, str]) -> dict[str, int]:
        return {label: int(settings[mask & (column == key)].sum()) for key, label in labels.items()}

    head_labels = {i: head_label(*choice) for i, choice in enumerate(HEAD_CHOICES)}
    return {
        "configs": int(settings[mask].sum()),
        "architectures": int(mask.sum()),
        "by_head": counts(g["head"], head_labels),
        "by_width": counts(g["d_model"], {d: str(d) for d in SEARCH_SPACE["d_model"]}),
        "by_channel": counts(g["mixing"], {0: "independent", 1: "mixing"}),
        "by_lookback": counts(g["input_len"], {length: str(length) for length in LOOKBACKS[dataset]}),
    }


def wide_feasible_table() -> dict[str, Any]:
    """Feasible counts per setting, per dataset (all four horizons) and shared.

    ``flatten_default`` repeats the counts for the flatten head at the
    default lookbacks, which equal today's feasible set.
    """
    per_setting, flatten_default = {}, {}
    for dataset, horizon in SETTINGS:
        g = shape_grid(dataset)
        mask = feasible_mask(dataset, (horizon,))
        per_setting[f"{dataset}_T{horizon}"] = {
            "budget": BUDGETS[(dataset, horizon)],
            "grid": wide_space_size(dataset),
            **summarize(dataset, mask),
        }
        today = mask & (g["head_code"] == 0) & (g["input_len"] == INPUT_LEN[dataset])
        flatten_default[f"{dataset}_T{horizon}"] = int(g["settings"][today].sum())
    per_dataset = {ds: {"grid": wide_space_size(ds), **summarize(ds, feasible_mask(ds))} for ds in DATASETS}
    shared = shared_mask()
    g = shape_grid(DATASETS[0])
    flatten_default["all_settings"] = int(g["settings"][shared & (g["head_code"] == 0)].sum())
    return {
        "per_setting": per_setting,
        "per_dataset": per_dataset,
        "shared": summarize(DATASETS[0], shared),
        "flatten_default": flatten_default,
    }
