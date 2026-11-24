"""A PatchTST forecaster whose lookback and forecasting head are design choices.

PatchTST's flatten head maps every patch embedding straight to the
forecast, so it holds ``d_model * n_patches * horizon`` weights, most of
the model at long horizons (docs/wide_space.md). Under the PatchTST/42
parameter cap that leaves only widths 16 and 32 and channel-independent
configs.
:class:`WideConfig` adds the lookback and a head type, so that wider
encoders and channel mixing can trade head weights for encoder weights:

* ``"flatten"``: the PatchTST head, one linear map from all patches.
* ``"pooled"``: mean over patches, then a linear map from ``d_model``.
* ``"lowrank"``: the flatten map factored through ``head_rank`` units.
* ``"shared_linear"``: one linear map applied to every patch, then a
  learned mix over patches for each output step. Its weight is the
  flatten weight restricted to an outer product per output step.

With the flatten head, :class:`WideTST` builds the same modules in the same
order as :class:`PatchTST`, so under the same seed both models hold the same
weights and give the same forecasts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn as nn

from leaguetst.models.config import ForecasterConfig
from leaguetst.models.patchtst import EncoderLayer, PatchEmbedding, PatchTST
from leaguetst.models.revin import RevIN
from leaguetst.utils.checks import require_int

HEAD_TYPES = ("flatten", "pooled", "lowrank", "shared_linear")
BACKBONE_FIELDS = tuple(f.name for f in fields(ForecasterConfig))
MAX_INPUT_LEN = 4096
MAX_HEAD_RANK = 1024


@dataclass(frozen=True)
class WideConfig:
    """The nine fields of :class:`ForecasterConfig` plus lookback and head.

    Attributes:
        input_len: lookback, the number of input steps.
        head_type: one of ``HEAD_TYPES``.
        head_rank: inner width of the ``"lowrank"`` head; 0 for every
            other head.
    """

    patch_len: int = 16
    stride: int = 8
    n_layers: int = 3
    n_heads: int = 4
    d_model: int = 16
    ff_ratio: int = 8
    dropout: float = 0.3
    instance_norm: bool = True
    channel_mode: str = "independent"
    input_len: int = 336
    head_type: str = "flatten"
    head_rank: int = 0

    def __post_init__(self) -> None:
        # ForecasterConfig checks and normalises the nine shared fields.
        backbone = self.backbone()
        for name in BACKBONE_FIELDS:
            object.__setattr__(self, name, getattr(backbone, name))
        object.__setattr__(self, "input_len", require_int("input_len", self.input_len, 1, MAX_INPUT_LEN))
        if self.input_len < self.patch_len:
            raise ValueError("input shorter than one patch")
        if not isinstance(self.head_type, str) or self.head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}")
        if self.head_type == "lowrank":
            object.__setattr__(self, "head_rank", require_int("head_rank", self.head_rank, 1, MAX_HEAD_RANK))
        elif require_int("head_rank", self.head_rank) != 0:
            raise ValueError("head_rank is used by the lowrank head only and must be 0 otherwise")

    def backbone(self) -> ForecasterConfig:
        """The nine shared fields as a :class:`ForecasterConfig`."""
        return ForecasterConfig(**{name: getattr(self, name) for name in BACKBONE_FIELDS})

    @property
    def d_ff(self) -> int:
        return self.ff_ratio * self.d_model

    @property
    def n_patches(self) -> int:
        """Patch count of the lookback, with PatchTST's end padding."""
        return (self.input_len - self.patch_len) // self.stride + 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "WideConfig":
        """Build a config from a mapping, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_forecaster(cls, config: ForecasterConfig, input_len: int) -> "WideConfig":
        """The wide config equal to ``PatchTST(config, ..., input_len, ...)``."""
        return cls(**config.to_dict(), input_len=input_len)


class PooledHead(nn.Module):
    """Mean over patches, then a linear map from ``d_model`` to the outputs."""

    def __init__(self, n_patches: int, d_model: int, out_features: int):
        super().__init__()
        self.n_patches = n_patches
        self.d_model = d_model
        self.linear = nn.Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.unflatten(-1, (self.n_patches, self.d_model)).mean(dim=-2))


class LowRankHead(nn.Module):
    """The flatten head factored as (outputs x rank) times (rank x inputs)."""

    def __init__(self, in_features: int, rank: int, out_features: int):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SharedLinearHead(nn.Module):
    """One linear map per patch, then a learned mix over patches per output.

    Output j is ``sum_n mix[n, j] * (W_j . h_n + b_j)``. The mix starts at
    ``1 / n_patches``, so at initialisation the head equals the pooled head.
    """

    def __init__(self, n_patches: int, d_model: int, out_features: int):
        super().__init__()
        self.n_patches = n_patches
        self.d_model = d_model
        self.per_patch = nn.Linear(d_model, out_features)
        self.mix = nn.Parameter(torch.full((n_patches, out_features), 1.0 / n_patches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        per_patch = self.per_patch(x.unflatten(-1, (self.n_patches, self.d_model)))
        return (per_patch * self.mix).sum(dim=-2)


def build_head(config: WideConfig, out_features: int) -> nn.Module:
    """The head of ``config``; every head maps (..., n_patches * d_model) to (..., out_features)."""
    n_patches, d_model = config.n_patches, config.d_model
    if config.head_type == "flatten":
        return nn.Linear(d_model * n_patches, out_features)
    if config.head_type == "pooled":
        return PooledHead(n_patches, d_model, out_features)
    if config.head_type == "lowrank":
        return LowRankHead(d_model * n_patches, config.head_rank, out_features)
    return SharedLinearHead(n_patches, d_model, out_features)


class WideTST(PatchTST):
    """PatchTST with the lookback and head read from a :class:`WideConfig`.

    ``encode`` and ``forward`` are PatchTST's own. The constructor does not
    call ``PatchTST.__init__``, which always allocates the flatten head; for a
    wide encoder that head alone can exceed the whole parameter cap. It builds
    the same modules in the same order instead, with the configured head.
    """

    def __init__(self, config: WideConfig, n_vars: int, horizon: int):
        nn.Module.__init__(self)
        self.config = config
        self.n_vars = n_vars
        self.input_len = config.input_len
        self.horizon = horizon
        self.n_patches = config.n_patches
        self.store_attention = False
        self.attention_maps: list[torch.Tensor] = []

        self.revin = RevIN() if config.instance_norm else None

        patch_features = config.patch_len
        head_outputs = horizon
        if config.channel_mode == "mixing":
            patch_features = config.patch_len * n_vars
            head_outputs = horizon * n_vars

        self.embedding = PatchEmbedding(config.patch_len, config.stride, patch_features, config.d_model)
        self.positional = nn.Parameter(torch.empty(self.n_patches, config.d_model))
        nn.init.uniform_(self.positional, -0.02, 0.02)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        )
        self.head = build_head(config, head_outputs)
