"""DLinear baseline (Zeng et al., AAAI 2023).

The input window is split into a moving-average trend and the remaining
seasonal part. Each part is mapped from ``input_len`` to ``horizon`` steps by
one linear layer shared across variables, and the two forecasts are summed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeriesDecomposition(nn.Module):
    """Split (batch, time, vars) input into seasonal and trend parts."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the window is centred")
        self.pad = (kernel_size - 1) // 2
        self.average = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front = x[:, :1, :].expand(-1, self.pad, -1)
        back = x[:, -1:, :].expand(-1, self.pad, -1)
        padded = torch.cat([front, x, back], dim=1)
        trend = self.average(padded.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class DLinear(nn.Module):
    """Two linear maps over time, one for the trend and one for the rest."""

    def __init__(self, n_vars: int, input_len: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        self.n_vars = n_vars
        self.input_len = input_len
        self.horizon = horizon
        self.decompose = SeriesDecomposition(kernel_size)
        self.seasonal = nn.Linear(input_len, horizon)
        self.trend = nn.Linear(input_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decompose(x)
        seasonal_out = self.seasonal(seasonal.transpose(1, 2))
        trend_out = self.trend(trend.transpose(1, 2))
        return (seasonal_out + trend_out).transpose(1, 2)
