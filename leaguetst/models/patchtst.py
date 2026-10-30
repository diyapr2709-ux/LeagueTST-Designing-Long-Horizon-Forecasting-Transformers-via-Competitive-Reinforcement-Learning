"""Configurable PatchTST forecaster.

The architecture follows the supervised PatchTST model (Nie et al., ICLR
2023): reversible instance normalisation, patching with end padding, a
learned positional embedding, a pre-activation-free encoder with batch
normalisation and residual attention scores, and a flatten-plus-linear head.

Two things differ from the reference code. The nine design choices in
:class:`ForecasterConfig` are read from one object instead of many flags,
and a ``channel_mode`` of ``"mixing"`` embeds all variables of a patch into a
single token instead of running every variable through the encoder alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from leaguetst.models.config import ForecasterConfig
from leaguetst.models.revin import RevIN


class PatchEmbedding(nn.Module):
    """Cut a series into patches and project each one to ``d_model``.

    Input has shape (batch, series, time). The time axis is padded at the end
    by one stride with its last value, so every input step belongs to at
    least one patch. Output has shape (batch, series, patches, d_model).
    """

    def __init__(self, patch_len: int, stride: int, in_features: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.project = nn.Linear(in_features, d_model)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, series, time) -> (batch, series, patches, patch_len)."""
        x = F.pad(x, (0, self.stride), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.patchify(x))


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention that also returns its scores.

    The raw scores of each layer are added to those of the next one before the
    softmax, the residual attention of PatchTST. Attention weights are kept
    so that later analysis can read them through forward hooks.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, prev_scores: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_tokens, _ = x.shape
        shape = (batch, n_tokens, self.n_heads, self.d_head)
        q = self.to_q(x).view(shape).transpose(1, 2)
        k = self.to_k(x).view(shape).transpose(1, 2)
        v = self.to_v(x).view(shape).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if prev_scores is not None:
            scores = scores + prev_scores
        weights = scores.softmax(dim=-1)

        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, n_tokens, -1)
        out = self.proj_dropout(self.to_out(out))
        return out, weights, scores


class TokenBatchNorm(nn.Module):
    """BatchNorm1d over the feature axis of (batch, tokens, features) input."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Post-norm transformer layer with batch normalisation and GELU."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = TokenBatchNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = TokenBatchNorm(d_model)

    def forward(
        self, x: torch.Tensor, prev_scores: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attended, weights, scores = self.attention(x, prev_scores)
        x = self.norm_attn(x + self.dropout_attn(attended))
        x = self.norm_ffn(x + self.dropout_ffn(self.feed_forward(x)))
        return x, weights, scores


class PatchTST(nn.Module):
    """PatchTST with the nine design choices read from a config.

    Forward input has shape (batch, input_len, n_vars) and the output has
    shape (batch, horizon, n_vars). When ``store_attention`` is set, the
    attention weights of the last forward pass are kept in ``attention_maps``,
    one tensor per layer.
    """

    def __init__(self, config: ForecasterConfig, n_vars: int, input_len: int, horizon: int):
        super().__init__()
        self.config = config
        self.n_vars = n_vars
        self.input_len = input_len
        self.horizon = horizon
        self.n_patches = config.n_patches(input_len)
        self.store_attention = False
        self.attention_maps: list[torch.Tensor] = []

        self.revin = RevIN() if config.instance_norm else None

        patch_features = config.patch_len
        head_outputs = horizon
        if config.channel_mode == "mixing":
            patch_features = config.patch_len * n_vars
            head_outputs = horizon * n_vars

        self.embedding = PatchEmbedding(
            config.patch_len, config.stride, patch_features, config.d_model
        )
        self.positional = nn.Parameter(torch.empty(self.n_patches, config.d_model))
        nn.init.uniform_(self.positional, -0.02, 0.02)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        )
        self.head = nn.Linear(config.d_model * self.n_patches, head_outputs)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the encoder over (sequences, patches, d_model) tokens."""
        x = self.embed_dropout(tokens + self.positional)
        scores = None
        maps = []
        for layer in self.layers:
            x, weights, scores = layer(x, scores)
            if self.store_attention:
                maps.append(weights.detach())
        if self.store_attention:
            self.attention_maps = maps
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.revin is not None:
            x, mean, std = self.revin.normalize(x)

        series = x.transpose(1, 2)
        if self.config.channel_mode == "independent":
            # One sequence per variable: (batch * vars, patches, d_model).
            tokens = self.embedding(series).reshape(batch * self.n_vars, self.n_patches, -1)
            encoded = self.encode(tokens)
            out = self.head(encoded.reshape(batch, self.n_vars, -1)).transpose(1, 2)
        else:
            # One sequence per window; each token stacks every variable's patch.
            patches = self.embedding.patchify(series)
            patches = patches.permute(0, 2, 1, 3).reshape(batch, self.n_patches, -1)
            tokens = self.embedding.project(patches)
            encoded = self.encode(tokens)
            out = self.head(encoded.reshape(batch, -1)).view(batch, self.horizon, self.n_vars)

        if self.revin is not None:
            out = self.revin.denormalize(out, mean, std)
        return out
