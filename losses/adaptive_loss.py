"""
AdaptiveLossWeighting — the proposed method.

Pipeline per forward pass:
    1. StatisticalFeatureExtractor(x_enc)  →  features       (B, C, 6)
    2. WeightGeneratorMLP(features)        →  log_var        (B, C, 3)
    3. Compute per-(sample,channel) losses →  raw_losses     (B, C, 3)
    4. Loss normalization (EMA)            →  norm_losses    (B, C, 3)  ← scale alignment
    5. Per-(sample,channel) unc. weighting →  (B, C)
    6. Batch + channel mean                →  scalar

Granularity — per (sample, channel), not per sample:
    An earlier version computed one weight per *sample*, using statistical
    features averaged across the C channels. In a multivariate window one
    channel can be smooth/trending while another is noisy at the very same
    timestep — averaging across channels collapses exactly that
    heterogeneity before the gate ever sees it, and empirically left the
    gate with too little to work with (diagnostic showed real, reproducible,
    interpretable correlations between features and weights, but too narrow
    a range to move final accuracy). Keeping the channel axis gives the gate
    genuinely more heterogeneous per-(sample,channel) inputs to condition on.

Loss Normalization (Step 4):
    MSE ≈ 0.3, MAE ≈ 0.5, Trend ≈ 0.01 — orders of magnitude apart.
    Without normalization, whichever component has the largest raw scale
    dominates the weighted sum regardless of the predicted log-variance.

    Fix: divide each loss by its exponential moving average (EMA) of the
    batch+channel mean.
        normalized_loss_i = loss_i / (ema_i + eps)

    After normalization all components hover around 1.0, so log_var_i(x)
    reflects "how (un)certain am I about component i for this (sample,
    channel)", not a correction for scale differences between components.

    EMA is updated only during training (not eval/inference) using:
        ema ← momentum * ema + (1 - momentum) * batch_mean

Per-(sample,channel) uncertainty weighting (Step 5):
    Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, 2018),
    applied per (sample, channel) and conditioned on statistical input
    features instead of being a single global learned scalar per task:

        loss = sum_i [ 0.5 * exp(-log_var_i) * norm_loss_i + 0.5 * log_var_i ]

    This replaces an earlier softmax-gate design. That design let the
    weight generator lower the reported loss for free by pushing weight
    toward zero on whichever component was hardest for a given sample —
    its gradient w.r.t. the weights was exactly the (normalized) per-sample
    losses, so "hide the hard part" was always a free win, and a hard
    min-weight floor + entropy regularization were needed to stop total
    collapse. Here, driving log_var_i toward +inf still shrinks the first
    term but the `+ 0.5 * log_var_i` term grows without bound, so ignoring a
    component is never free. For a fixed norm_loss_i, the optimum is
    log_var_i* = log(norm_loss_i) — i.e. the log-variance that best
    *predicts* how hard/noisy that component actually is for that
    (sample, channel), not the one that best hides it. No entropy term or
    weight floor is needed for this reason.
"""

import torch
import torch.nn as nn

from features.statistical_extractor import StatisticalFeatureExtractor
from weight_generator.mlp_generator import WeightGeneratorMLP

LOSS_NAMES = ["mse", "mae", "trend"]


def _mse_channel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over time only, keeping the channel axis. Returns (B, C)."""
    return ((pred - target) ** 2).mean(dim=1)


def _mae_channel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over time only, keeping the channel axis. Returns (B, C)."""
    return (pred - target).abs().mean(dim=1)


def _trend_channel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """First-order temporal difference MSE, mean over time, per channel. Returns (B, C)."""
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    return ((pred_diff - target_diff) ** 2).mean(dim=1)


class AdaptiveLossWeighting(nn.Module):
    """
    Per-(sample, channel) adaptive combination of MSE, MAE, and Trend losses
    via feature-conditioned homoscedastic uncertainty weighting.

    Args:
        num_stat_features    : dimensionality of the statistical feature vector (default: 6)
        hidden_dim           : hidden dim of the weight generator MLP (default: 64)
        dropout              : dropout in the weight generator (default: 0.1)
        max_log_var          : clamp range for predicted log-variance (default: 4.0)
        loss_norm_momentum   : EMA momentum for loss normalization (default: 0.9)
                               Higher = slower adaptation, more stable normalization.
        feature_norm_momentum: EMA momentum for statistical feature normalization (default: 0.9)
    """

    def __init__(
        self,
        num_stat_features: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        max_log_var: float = 4.0,
        loss_norm_momentum: float = 0.9,
        feature_norm_momentum: float = 0.9,
    ):
        super().__init__()

        self.loss_norm_momentum = loss_norm_momentum

        self.feature_extractor = StatisticalFeatureExtractor(norm_momentum=feature_norm_momentum)

        self.weight_generator = WeightGeneratorMLP(
            num_stat_features=num_stat_features,
            num_losses=len(LOSS_NAMES),
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_log_var=max_log_var,
        )

        # EMA running mean for each loss component — initialized to 1.0 so that
        # normalization is a no-op at the start of training.
        # register_buffer: part of module state (saved/loaded with state_dict)
        # but NOT a learnable parameter.
        self.register_buffer(
            "loss_running_mean",
            torch.ones(len(LOSS_NAMES)),
        )

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_running_mean(self, batch_mean: torch.Tensor):
        """Update EMA with the current batch+channel mean loss values (detached)."""
        self.loss_running_mean = (
            self.loss_norm_momentum * self.loss_running_mean
            + (1.0 - self.loss_norm_momentum) * batch_mean.detach()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,    # (B, pred_len, C)
        target: torch.Tensor,  # (B, pred_len, C)
        x_enc: torch.Tensor,   # (B, seq_len, C)
    ) -> torch.Tensor:

        # Step 1: Per-channel statistical feature extraction
        features = self.feature_extractor(x_enc)       # (B, C, 6)

        # Step 2: Per-(sample,channel) log-variance prediction
        # detach: features are derived from input data, not learned representations.
        # Preventing gradients from flowing back through the feature extractor
        # removes spurious gradient paths and stabilizes MLP training.
        log_var = self.weight_generator(features.detach())  # (B, C, 3)

        # Step 3: Per-(sample,channel) loss computation
        per_sample_losses = torch.stack(
            [_mse_channel(pred, target), _mae_channel(pred, target), _trend_channel(pred, target)],
            dim=2,
        )  # (B, C, 3)

        # Step 4: Loss normalization via EMA
        # Update running mean only during training
        if self.training:
            batch_mean = per_sample_losses.mean(dim=(0, 1))  # (3,)
            self._update_running_mean(batch_mean)

        # Normalize: each component now has expected magnitude ≈ 1.0
        norm_losses = per_sample_losses / (self.loss_running_mean.view(1, 1, -1) + 1e-8)

        # Step 5: Per-(sample,channel) homoscedastic uncertainty weighting.
        # precision_i = exp(-log_var_i); ignoring a component (log_var_i -> inf)
        # is penalized by the uncapped `+ 0.5 * log_var_i` term below.
        precision = torch.exp(-log_var)  # (B, C, 3)
        per_sample_total = 0.5 * (precision * norm_losses + log_var).sum(dim=2)  # (B, C)

        return per_sample_total.mean()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def get_current_weights(self, x_enc: torch.Tensor) -> dict:
        """
        Average *interpretable* weight per loss component over the batch and
        channel axes (for logging) — precision normalized to sum to 1, i.e.
        how much of the total confidence mass is allocated to each
        component. Training itself does not use this normalized form; it
        uses the raw precision and log_var directly (see forward).
        """
        with torch.no_grad():
            features = self.feature_extractor(x_enc)
            log_var = self.weight_generator(features.detach())
            precision = torch.exp(-log_var)
            weights = precision / precision.sum(dim=2, keepdim=True)
            avg = weights.mean(dim=(0, 1))
        return {name: avg[i].item() for i, name in enumerate(LOSS_NAMES)}

    def get_running_means(self) -> dict:
        """Current EMA normalization values (for logging/debugging)."""
        return {name: self.loss_running_mean[i].item() for i, name in enumerate(LOSS_NAMES)}
