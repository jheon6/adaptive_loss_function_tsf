"""
AdaptiveLossWeighting — the proposed method.

Pipeline per forward pass:
    1. StatisticalFeatureExtractor(x_enc)  →  features       (B, 6)
    2. WeightGeneratorMLP(features)        →  weights        (B, 4)
    3. Compute per-sample losses           →  raw_losses     (B, 4)
    4. Loss normalization (EMA)            →  norm_losses    (B, 4)  ← scale alignment
    5. Weighted sum per sample             →  (B,)
    6. Entropy regularization              →  penalize weight collapse
    7. Batch mean                          →  scalar

Loss Normalization (Step 4):
    MSE ≈ 0.3, MAE ≈ 0.5, Trend ≈ 0.01, Frequency ≈ 5~50 — orders of magnitude apart.
    Without normalization, Frequency Loss dominates the weighted sum regardless of weights,
    and the MLP learns to put w_freq → 0 just to survive, rather than learning
    meaningful signal-adaptive weighting.

    Fix: divide each loss by its exponential moving average (EMA) of the batch mean.
        normalized_loss_i = loss_i / (ema_i + eps)

    After normalization all four components hover around 1.0, so the MLP's weights
    directly reflect "how much should I care about component i for this sample"
    rather than compensating for scale differences.

    EMA is updated only during training (not eval/inference) using:
        ema ← momentum * ema + (1 - momentum) * batch_mean

Entropy regularization (Step 6):
    Prevents weight collapse — the MLP collapsing all weight onto the currently
    smallest loss component.  Adds -entropy_coef * H(w) to the total loss,
    where H(w) = -Σ w_j * log(w_j).
"""

import torch
import torch.nn as nn

from .components import MSELoss, MAELoss, TrendLoss
from features.statistical_extractor import StatisticalFeatureExtractor
from weight_generator.mlp_generator import WeightGeneratorMLP

LOSS_NAMES = ["mse", "mae", "trend"]


class AdaptiveLossWeighting(nn.Module):
    """
    Sample-adaptive weighted combination of MSE, MAE, Trend, and Frequency losses
    with EMA-based loss normalization and entropy regularization.

    Args:
        num_stat_features    : dimensionality of the statistical feature vector (default: 6)
        hidden_dim           : hidden dim of the weight generator MLP (default: 64)
        dropout              : dropout in the weight generator (default: 0.1)
        temperature          : softmax temperature — lower = sharper weights (default: 1.0)
        entropy_coef         : entropy regularization strength; 0.0 = disabled (default: 0.1)
        loss_norm_momentum   : EMA momentum for loss normalization (default: 0.9)
                               Higher = slower adaptation, more stable normalization.
        feature_norm_momentum: EMA momentum for statistical feature normalization (default: 0.9)
        min_weight           : hard floor on each component's weight after softmax (default: 0.05)
    """

    def __init__(
        self,
        num_stat_features: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
        entropy_coef: float = 0.1,
        loss_norm_momentum: float = 0.9,
        feature_norm_momentum: float = 0.9,
        min_weight: float = 0.05,
    ):
        super().__init__()

        self.entropy_coef = entropy_coef
        self.loss_norm_momentum = loss_norm_momentum

        self.feature_extractor = StatisticalFeatureExtractor(norm_momentum=feature_norm_momentum)

        self.weight_generator = WeightGeneratorMLP(
            num_stat_features=num_stat_features,
            num_losses=len(LOSS_NAMES),
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
            min_weight=min_weight,
        )

        self.loss_components = nn.ModuleList([
            MSELoss(),
            MAELoss(),
            TrendLoss(),
        ])

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
        """Update EMA with the current batch's mean loss values (detached)."""
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

        # Step 1: Statistical feature extraction
        features = self.feature_extractor(x_enc)       # (B, 6)

        # Step 2: Per-sample weight prediction
        # detach: features are derived from input data, not learned representations.
        # Preventing gradients from flowing back through the feature extractor
        # removes spurious gradient paths and stabilizes MLP training.
        weights = self.weight_generator(features.detach())  # (B, 3)

        # Step 3: Per-sample loss computation
        per_sample_losses = torch.stack(
            [component(pred, target) for component in self.loss_components],
            dim=1,
        )  # (B, 4)

        # Step 4: Loss normalization via EMA
        # Update running mean only during training
        if self.training:
            batch_mean = per_sample_losses.mean(dim=0)  # (4,)
            self._update_running_mean(batch_mean)

        # Normalize: each component now has expected magnitude ≈ 1.0
        norm_losses = per_sample_losses / (self.loss_running_mean.unsqueeze(0) + 1e-8)

        # Step 5: Weighted sum per sample → (B,)
        weighted = (weights * norm_losses).sum(dim=1)

        # Step 6: Entropy regularization — discourage weight collapse
        if self.entropy_coef > 0.0:
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            return weighted.mean() - self.entropy_coef * entropy

        return weighted.mean()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def get_current_weights(self, x_enc: torch.Tensor) -> dict:
        """Average weight per loss component over the batch (for logging)."""
        with torch.no_grad():
            features = self.feature_extractor(x_enc)
            weights  = self.weight_generator(features.detach())
            avg      = weights.mean(dim=0)
        return {name: avg[i].item() for i, name in enumerate(LOSS_NAMES)}

    def get_running_means(self) -> dict:
        """Current EMA normalization values (for logging/debugging)."""
        return {name: self.loss_running_mean[i].item() for i, name in enumerate(LOSS_NAMES)}
