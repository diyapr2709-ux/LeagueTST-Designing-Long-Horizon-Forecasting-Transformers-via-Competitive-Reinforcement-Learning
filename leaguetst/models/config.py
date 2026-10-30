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

CHANNEL_MODES = ("independent", "mixing")


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
        if self.patch_len <= 0 or self.stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if self.stride > self.patch_len:
            raise ValueError("stride larger than patch_len would skip time steps")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}"
            )
        if self.ff_ratio <= 0:
            raise ValueError("ff_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.channel_mode not in CHANNEL_MODES:
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
