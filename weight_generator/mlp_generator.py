"""
WeightGeneratorMLP — learns to predict per-sample loss weights.

Architecture:
    features (B, num_stat_features)
    → Linear(num_stat_features, hidden_dim)
    → LayerNorm + ReLU
    → Dropout
    → Linear(hidden_dim, num_losses)
    → Softmax(dim=-1)

Output: weights (B, num_losses) where weights.sum(dim=-1) == 1.

The MLP is trained end-to-end with the backbone via the total loss gradient.
LayerNorm is added before the activation to stabilize training when statistical
features have very different scales (e.g. variance vs. normalized entropy).
"""

import torch
import torch.nn as nn


class WeightGeneratorMLP(nn.Module):
    """
    Lightweight MLP that maps statistical features to loss weights.

    Args:
        num_stat_features : dimensionality of the input feature vector (default: 6)
        num_losses        : number of loss components to weight (default: 4)
        hidden_dim        : width of the single hidden layer (default: 64)
        dropout           : dropout probability for regularization (default: 0.1)
        temperature       : softmax temperature — lower → sharper distribution (default: 1.0)
        min_weight        : hard floor applied to every component after softmax,
                             so no component can ever be driven fully to zero
                             (entropy regularization alone only discourages
                             collapse, it doesn't prevent it). Must satisfy
                             min_weight * num_losses < 1. (default: 0.05)
    """

    def __init__(
        self,
        num_stat_features: int = 6,
        num_losses: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
        min_weight: float = 0.05,
    ):
        super().__init__()
        assert min_weight * num_losses < 1.0, (
            f"min_weight={min_weight} with num_losses={num_losses} leaves no room "
            f"for adaptivity (min_weight * num_losses must be < 1)."
        )
        self.temperature = temperature
        self.num_losses = num_losses
        self.min_weight = min_weight

        self.net = nn.Sequential(
            nn.Linear(num_stat_features, hidden_dim),
            nn.LayerNorm(hidden_dim),          # stabilize across feature scales
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_losses),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize the final linear layer near zero so that the initial
        weight distribution starts close to uniform (= equal weighting)."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias of the last layer: set to 0 so softmax starts uniform
        last_linear = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, num_stat_features)

        Returns:
            weights  : (B, num_losses)  — sums to 1 along dim=-1
        """
        logits = self.net(features)                        # (B, num_losses)
        weights = torch.softmax(logits / self.temperature, dim=-1)

        # Hard floor: rescale the softmax output into [min_weight, 1] so every
        # component keeps at least min_weight, while still summing to 1.
        if self.min_weight > 0.0:
            weights = weights * (1.0 - self.num_losses * self.min_weight) + self.min_weight

        return weights
