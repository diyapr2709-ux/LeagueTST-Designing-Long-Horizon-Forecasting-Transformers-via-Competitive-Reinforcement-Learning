"""Parameter counts and the feasible part of the design space.

A configuration is feasible at a setting when its PatchTST parameter count
is at most that of the PatchTST/42 reproduction at the same dataset and
horizon. The count is computed in closed form from the layer shapes so that
all 30,720 configurations can be checked in a fraction of a second; the
tests compare it with real instantiated models.

The search covers ``FEASIBLE_ALL``: the configurations that fit the cap at
all eight settings, so a winner found at one setting can be retrained at
every other within budget.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from leaguetst.models.config import ForecasterConfig
from leaguetst.search.space import enumerate_configs

N_VARS = 7

# Parameter counts of the PatchTST/42 reproduction (docs/progress.md).
BUDGETS: dict[tuple[str, int], int] = {
    ("ETTh1", 96): 81_728,
    ("ETTh1", 192): 146_336,
    ("ETTh1", 336): 243_248,
    ("ETTh1", 720): 501_680,
    ("ILI", 24): 33_400,
    ("ILI", 36): 41_476,
    ("ILI", 48): 49_552,
    ("ILI", 60): 57_628,
}
INPUT_LEN = {"ETTh1": 336, "ILI": 104}
SETTINGS: tuple[tuple[str, int], ...] = tuple(BUDGETS)


def patchtst_parameters(config: ForecasterConfig, input_len: int, horizon: int, n_vars: int = N_VARS) -> int:
    """Trainable parameters of ``PatchTST(config, n_vars, input_len, horizon)``.

    Per encoder layer: four d x d projections with bias, two batch norms
    (weight and bias), and the two feed-forward linears. Around the encoder:
    the patch projection, the positional table and the flatten head.
    RevIN has no parameters.
    """
    d = config.d_model
    d_ff = config.d_ff
    n_patches = config.n_patches(input_len)
    mixing = config.channel_mode == "mixing"
    in_features = config.patch_len * (n_vars if mixing else 1)
    out_features = horizon * (n_vars if mixing else 1)
    per_layer = 4 * (d * d + d) + 2 * (2 * d) + (d * d_ff + d_ff) + (d_ff * d + d)
    embedding = in_features * d + d
    positional = n_patches * d
    head = d * n_patches * out_features + out_features
    return config.n_layers * per_layer + embedding + positional + head


def config_id(config: ForecasterConfig) -> str:
    """Short readable key that names one configuration uniquely.

    Example: ``p16s8l3h4d16f8r3n1i`` is patch 16, stride 8, 3 layers,
    4 heads, width 16, ff ratio 8, dropout 0.3, instance norm on,
    channel-independent.
    """
    dropout_tenths = round(config.dropout * 10)
    if abs(config.dropout * 10 - dropout_tenths) > 1e-9:
        raise ValueError(f"dropout {config.dropout} is not a multiple of 0.1")
    mode = "i" if config.channel_mode == "independent" else "m"
    return (
        f"p{config.patch_len}s{config.stride}l{config.n_layers}h{config.n_heads}"
        f"d{config.d_model}f{config.ff_ratio}r{dropout_tenths}n{int(config.instance_norm)}{mode}"
    )


def fits(config: ForecasterConfig, dataset: str, horizon: int) -> bool:
    """Whether ``config`` is within the parameter cap at one setting."""
    budget = BUDGETS[(dataset, horizon)]
    return patchtst_parameters(config, INPUT_LEN[dataset], horizon) <= budget


@lru_cache(maxsize=1)
def _grid() -> tuple[ForecasterConfig, ...]:
    return tuple(enumerate_configs())


def feasible_configs(dataset: str, horizon: int) -> list[ForecasterConfig]:
    """Grid configurations within the cap at one setting, in grid order."""
    return [c for c in _grid() if fits(c, dataset, horizon)]


@lru_cache(maxsize=1)
def feasible_all() -> tuple[ForecasterConfig, ...]:
    """Grid configurations within the cap at all eight settings, in grid order."""
    return tuple(c for c in _grid() if all(fits(c, ds, t) for ds, t in SETTINGS))


def feasible_table() -> dict[str, Any]:
    """Counts per setting, split by channel mode, for the report."""
    rows = {}
    for dataset, horizon in SETTINGS:
        configs = feasible_configs(dataset, horizon)
        rows[f"{dataset}_T{horizon}"] = {
            "budget": BUDGETS[(dataset, horizon)],
            "feasible": len(configs),
            "independent": sum(c.channel_mode == "independent" for c in configs),
            "mixing": sum(c.channel_mode == "mixing" for c in configs),
        }
    return {"per_setting": rows, "all_settings": len(feasible_all()), "grid": len(_grid())}


def configs_by_id() -> dict[str, ForecasterConfig]:
    """``config_id`` to config for every configuration in ``feasible_all()``."""
    table = {config_id(c): c for c in feasible_all()}
    if len(table) != len(feasible_all()):
        raise AssertionError("config_id is not unique over the feasible set")
    return table
