"""
Individual loss components used in the Adaptive Loss Weighting framework.

All losses return per-sample scalars of shape (B,) so that they can be
weighted independently for each sample before the batch mean is taken.

Components:
    MSELoss        — standard mean squared error
    MAELoss        — mean absolute error
    TrendLoss      — MSE on first-order temporal differences
    FrequencyLoss  — MAE on FFT magnitude spectra
"""

import torch
import torch.nn as nn
from .base_loss import BaseLoss


class MSELoss(BaseLoss):
    """
    Mean Squared Error, averaged over time steps and channels.
    Captures overall magnitude of prediction error.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, T, C)
        # Mean over T and C, keep B → (B,)
        return ((pred - target) ** 2).mean(dim=(1, 2))


class MAELoss(BaseLoss):
    """
    Mean Absolute Error, averaged over time steps and channels.
    Less sensitive to outliers than MSE; complements it in the mixture.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean(dim=(1, 2))


class TrendLoss(BaseLoss):
    """
    Trend Loss based on first-order temporal differences.

    Compares the rate-of-change (finite differences) between predicted
    and target sequences.  High weight on this loss encourages the model
    to capture the correct direction and magnitude of changes rather than
    just the absolute level.

    diff(x)[t] = x[t+1] - x[t]   →   shape (B, T-1, C)
    Loss = MSE(diff(pred), diff(target))
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # First-order finite difference along the time axis
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]      # (B, T-1, C)
        target_diff = target[:, 1:, :] - target[:, :-1, :]

        return ((pred_diff - target_diff) ** 2).mean(dim=(1, 2))


class FrequencyLoss(BaseLoss):
    """
    Frequency Domain Loss based on FFT magnitude spectra.

    Computes the MAE between the real FFT magnitude of the predicted
    and target sequences.  This penalizes errors in dominant periodicities
    and seasonal patterns that may not be captured by point-wise losses.

    Only the non-redundant half of the spectrum (rfft) is used.
    Magnitudes are normalized by sequence length so the loss scale is
    independent of pred_len.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, T, C = pred.shape

        # rfft along time dim; result shape (B, T//2+1, C)
        pred_fft = torch.fft.rfft(pred, dim=1, norm="ortho").abs()
        target_fft = torch.fft.rfft(target, dim=1, norm="ortho").abs()

        return (pred_fft - target_fft).abs().mean(dim=(1, 2))
