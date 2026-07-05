"""
FixedWeightLoss — static combination of loss components for ablation studies.

Ablation configurations (from losses/__init__.py):
    mse_only              : {"mse": 1.0}
    mse_mae_fixed         : {"mse": 0.5, "mae": 0.5}
    mse_trend_fixed       : {"mse": 0.5, "trend": 0.5}
    mse_trend_freq_fixed  : {"mse": 1/3, "trend": 1/3, "frequency": 1/3}
    mse_mae_trend_fixed   : {"mse": 1/3, "mae": 1/3, "trend": 1/3}

The weights are fixed scalars (not learned) and identical for every sample.
This class is intentionally a thin wrapper around the loss components so that
fair comparison with AdaptiveLossWeighting is straightforward.
"""

import torch
import torch.nn as nn
from .components import MSELoss, MAELoss, TrendLoss, FrequencyLoss

_COMPONENT_MAP = {
    "mse": MSELoss,
    "mae": MAELoss,
    "trend": TrendLoss,
    "frequency": FrequencyLoss,
}


class FixedWeightLoss(nn.Module):
    """
    Fixed-weight linear combination of loss components.

    Args:
        weights : dict mapping component name to scalar weight.
                  Weights are automatically normalized to sum to 1.
                  Example: {"mse": 0.5, "trend": 0.5}
    """

    def __init__(self, weights: dict):
        super().__init__()

        assert len(weights) > 0, "At least one loss component must be specified."
        for key in weights:
            assert key in _COMPONENT_MAP, (
                f"Unknown loss component '{key}'. "
                f"Available: {list(_COMPONENT_MAP.keys())}"
            )

        # Normalize weights so they sum to 1
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

        # Instantiate each active loss component
        self.components = nn.ModuleDict({
            name: _COMPONENT_MAP[name]()
            for name in self.weights
        })

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_enc: torch.Tensor = None,  # unused; kept for API compatibility with AdaptiveLoss
    ) -> torch.Tensor:
        """
        Args:
            pred   : (B, pred_len, C)
            target : (B, pred_len, C)
            x_enc  : ignored (for interface compatibility)

        Returns:
            scalar loss (batch mean)
        """
        total = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)

        for name, component in self.components.items():
            per_sample = component(pred, target)   # (B,)
            total = total + self.weights[name] * per_sample

        return total.mean()

    def get_weight_dict(self) -> dict:
        """Return the normalized weight dictionary (for logging)."""
        return dict(self.weights)
