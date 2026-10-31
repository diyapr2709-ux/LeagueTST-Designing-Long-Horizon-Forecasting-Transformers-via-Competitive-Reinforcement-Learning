"""Configuration of the forecasting transformer.

The nine fields of :class:`ForecasterConfig` are the design choices that the
search methods may set. Everything else about the model (positional encoding,
batch normalisation, GELU, residual attention) is fixed to the PatchTST
recipe so that the search compares like with like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch.nn as nn

from leaguetst.utils.checks import require_bool, require_int, require_real

CHANNEL_MODES = ("independent", "mixing")

# Upper bounds far above the search grid and the baselines. They exist so that
# a runaway proposal fails fast instead of allocating an enormous model.
LIMITS = {"patch_len": 512, "stride": 512, "n_layers": 12, "n_heads": 64, "d_model": 1024, "ff_ratio": 16}


@dataclass(frozen=True)
class ForecasterConfig:
    """The nine design choices of a forecaster.

    Attributes:
        patch_len: number of time steps per patch.
        stride: step between the starts of consecutive patches.
        n_layers: number of encoder layers.
        n_heads: number of attention heads; must divide ``d_model``.
        d_model: width of the token embedding.
        ff_ratio: feed-forward width as a multiple of ``d_model``.
        dropout: dropout applied to embeddings, attention output and the
            feed-forward block.
        instance_norm: whether to apply reversible instance normalisation.
        channel_mode: ``"independent"`` treats every variable as its own
            sequence; ``"mixing"`` embeds all variables of a patch together.
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

    def __post_init__(self) -> None:
        # The dataclass is frozen, so normalised values are written back with
        # object.__setattr__ (a NumPy integer becomes a plain int, for example).
        for name, limit in LIMITS.items():
            object.__setattr__(self, name, require_int(name, getattr(self, name), 1, limit))
        object.__setattr__(self, "dropout", require_real("dropout", self.dropout, 0.0, 1.0, high_open=True))
        require_bool("instance_norm", self.instance_norm)
        if self.stride > self.patch_len:
            raise ValueError("stride larger than patch_len would skip time steps")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}"
            )
        if not isinstance(self.channel_mode, str) or self.channel_mode not in CHANNEL_MODES:
            raise ValueError(f"channel_mode must be one of {CHANNEL_MODES}")

    @property
    def d_ff(self) -> int:
        return self.ff_ratio * self.d_model

    def n_patches(self, input_len: int) -> int:
        """Patch count for an input of ``input_len`` steps with end padding.

        The sequence is padded by one stride at the end, as in PatchTST, so a
        336-step input with patch 16 and stride 8 yields 42 patches.
        """
        if input_len < self.patch_len:
            raise ValueError("input shorter than one patch")
        return (input_len - self.patch_len) // self.stride + 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ForecasterConfig":
        """Build a config from a mapping, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**values)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
