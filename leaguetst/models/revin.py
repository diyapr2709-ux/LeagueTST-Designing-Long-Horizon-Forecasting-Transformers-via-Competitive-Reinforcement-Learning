"""Reversible instance normalisation (Kim et al., ICLR 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Normalise each input window per variable and undo it on the forecast.

    The statistics are computed over the time axis of each window and are
    detached from the graph, matching the PatchTST implementation with the
    affine option switched off.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale ``x`` of shape (batch, time, vars); return it with its stats."""
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / std, mean, std

    @staticmethod
    def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Restore the original scale of a forecast of shape (batch, time, vars)."""
        return x * std + mean
