"""
DLinear: Are Transformers Effective for Time Series Forecasting?
Zeng et al., AAAI 2023.  https://arxiv.org/abs/2205.13504

Original decomposition-based linear model.
This implementation is kept identical to the paper — no modifications.

Input shape  : (B, seq_len, num_features)
Output shape : (B, pred_len, num_features)
"""

import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    """Centered moving average for series decomposition."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # Pad both ends to keep sequence length
        pad_left = self.kernel_size // 2
        pad_right = self.kernel_size - pad_left - 1

        # Replicate-pad along the time dimension
        x_pad = torch.cat(
            [x[:, :1, :].expand(-1, pad_left, -1),
             x,
             x[:, -1:, :].expand(-1, pad_right, -1)],
            dim=1,
        )  # (B, T + kernel_size - 1, C)

        # AvgPool1d expects (B, C, T)
        x_pad = x_pad.permute(0, 2, 1)
        trend = self.avg(x_pad)           # (B, C, T)
        trend = trend.permute(0, 2, 1)    # (B, T, C)
        return trend


class _SeriesDecomposition(nn.Module):
    """Additive decomposition: x = trend + seasonal."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear backbone.

    Two independent linear layers — one for the seasonal component
    and one for the trend component — map seq_len → pred_len
    independently for each channel (channel-independent setting).
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features

        # Decomposition kernel size (paper default: 25)
        kernel_size = getattr(config, "moving_avg", 25)
        self.decomposition = _SeriesDecomposition(kernel_size)

        # Channel-independent: one weight per feature channel
        # Weight shape: (num_features, seq_len, pred_len) → factored as num_features separate linears
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        # Initialize trend linear with small weights (helps stabilize early training)
        nn.init.xavier_uniform_(self.linear_seasonal.weight)
        nn.init.xavier_uniform_(self.linear_trend.weight)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc : (B, seq_len, C)

        Returns:
            pred  : (B, pred_len, C)
        """
        seasonal, trend = self.decomposition(x_enc)  # each (B, T, C)

        # Operate on the time dimension: transpose to (B, C, T)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        seasonal_out = self.linear_seasonal(seasonal)   # (B, C, pred_len)
        trend_out = self.linear_trend(trend)             # (B, C, pred_len)

        pred = seasonal_out + trend_out                  # (B, C, pred_len)
        pred = pred.permute(0, 2, 1)                     # (B, pred_len, C)
        return pred
