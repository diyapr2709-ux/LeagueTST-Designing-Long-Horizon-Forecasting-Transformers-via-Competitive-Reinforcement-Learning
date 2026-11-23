"""PatchTST forward pass in NumPy float64, read from a saved state dict.

Written from the layer definitions, not from the torch module, and run in
evaluation mode: dropout off, batch norm with its running statistics.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

REVIN_EPS = 1e-5
BATCHNORM_EPS = 1e-5


def _linear(x: np.ndarray, state: dict[str, np.ndarray], name: str) -> np.ndarray:
    return x @ state[f"{name}.weight"].T + state[f"{name}.bias"]


def _batchnorm(x: np.ndarray, state: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Batch norm over the last axis with running statistics."""
    mean = state[f"{name}.norm.running_mean"]
    var = state[f"{name}.norm.running_var"]
    return (x - mean) / np.sqrt(var + BATCHNORM_EPS) * state[f"{name}.norm.weight"] + state[f"{name}.norm.bias"]


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _patches(series: np.ndarray, patch_len: int, stride: int) -> np.ndarray:
    """(..., time) -> (..., patches, patch_len), end padded by one stride with the last value."""
    padded = np.concatenate([series, np.repeat(series[..., -1:], stride, axis=-1)], axis=-1)
    n_patches = (padded.shape[-1] - patch_len) // stride + 1
    starts = np.arange(n_patches) * stride
    index = starts[:, None] + np.arange(patch_len)[None, :]
    return padded[..., index]


def numpy_patchtst_forward(
    state: dict[str, np.ndarray],
    config: dict,
    inputs: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Forecasts of shape (batch, horizon, variables) for inputs (batch, time, variables).

    ``config`` must name every field that changes the computation; nothing
    is defaulted, so a config missing a field fails instead of guessing.
    """
    for key in ("patch_len", "stride", "n_heads", "instance_norm", "channel_mode"):
        if key not in config:
            raise KeyError(f"model config lacks {key}")
    x = inputs.astype(np.float64)
    batch, _, n_vars = x.shape
    if config["instance_norm"]:
        mean = x.mean(axis=1, keepdims=True)
        std = np.sqrt(x.var(axis=1, keepdims=True) + REVIN_EPS)
        x = (x - mean) / std

    series = x.transpose(0, 2, 1)  # (batch, vars, time)
    patches = _patches(series, int(config["patch_len"]), int(config["stride"]))  # (batch, vars, P, patch_len)
    n_patches = patches.shape[2]
    mixing = config["channel_mode"] == "mixing"
    if mixing:
        tokens_in = patches.transpose(0, 2, 1, 3).reshape(batch, n_patches, -1)
    else:
        tokens_in = patches.reshape(batch * n_vars, n_patches, -1)

    h = _linear(tokens_in, state, "embedding.project") + state["positional"]
    d_model = h.shape[-1]
    n_heads = int(config["n_heads"])
    d_head = d_model // n_heads
    scores_prev = None
    layer = 0
    while f"layers.{layer}.attention.to_q.weight" in state:
        prefix = f"layers.{layer}"
        seqs, tokens, _ = h.shape

        def heads(y: np.ndarray) -> np.ndarray:
            return y.reshape(seqs, tokens, n_heads, d_head).transpose(0, 2, 1, 3)

        q = heads(_linear(h, state, f"{prefix}.attention.to_q"))
        k = heads(_linear(h, state, f"{prefix}.attention.to_k"))
        v = heads(_linear(h, state, f"{prefix}.attention.to_v"))
        scores = q @ k.transpose(0, 1, 3, 2) * d_head**-0.5
        if scores_prev is not None:
            scores = scores + scores_prev
        attended = (_softmax(scores) @ v).transpose(0, 2, 1, 3).reshape(seqs, tokens, d_model)
        attended = _linear(attended, state, f"{prefix}.attention.to_out")
        h = _batchnorm(h + attended, state, f"{prefix}.norm_attn")
        ff = _linear(_gelu(_linear(h, state, f"{prefix}.feed_forward.0")), state, f"{prefix}.feed_forward.3")
        h = _batchnorm(h + ff, state, f"{prefix}.norm_ffn")
        scores_prev = scores
        layer += 1

    if mixing:
        out = _linear(h.reshape(batch, -1), state, "head").reshape(batch, horizon, n_vars)
    else:
        out = _linear(h.reshape(batch, n_vars, -1), state, "head").transpose(0, 2, 1)
    if config["instance_norm"]:
        out = out * std + mean
    return out
