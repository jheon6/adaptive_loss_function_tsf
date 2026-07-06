"""
WeightGeneratorMLP — predicts per-sample loss uncertainty (log-variance).

Architecture:
    features (B, num_stat_features)
    → Linear(num_stat_features, hidden_dim)
    → LayerNorm + ReLU
    → Dropout
    → Linear(hidden_dim, num_losses)
    → clamp to [-max_log_var, max_log_var]

Output: log_var (B, num_losses), UNCONSTRAINED (no softmax).

Why not softmax:
    A softmax gate is trained through the same objective it modulates, so its
    gradient w.r.t. each weight is exactly that component's (normalized) loss.
    The MLP can lower the reported total loss for free by pushing weight
    toward zero on whatever component is currently hardest for a sample —
    it never has to pay for hiding a hard target, so `min_weight` floors and
    entropy regularization were needed as after-the-fact patches.

    Instead we predict a per-sample log-variance s_i(x) and combine losses as
    homoscedastic uncertainty weighting (Kendall et al., 2018), applied
    per-sample instead of as a single global scalar per task:

        L = sum_i [ 0.5 * exp(-s_i) * loss_i  +  0.5 * s_i ]

    Driving s_i -> +inf (i.e. "ignore this component") still lowers the first
    term but the `+0.5 * s_i` term grows without bound, so it is never free.
    Solving dL/ds_i = 0 for fixed loss_i gives s_i* = log(loss_i): the
    optimum log-variance is the one that actually reflects how hard/noisy
    that component is for that sample, not the one that best hides it.
"""

import torch
import torch.nn as nn


class WeightGeneratorMLP(nn.Module):
    """
    Lightweight MLP that maps statistical features to per-loss log-variances.

    Args:
        num_stat_features : dimensionality of the input feature vector (default: 6)
        num_losses        : number of loss components to weight (default: 3)
        hidden_dim        : width of the single hidden layer (default: 64)
        dropout           : dropout probability for regularization (default: 0.1)
        max_log_var       : clamp range for the predicted log-variance, so
                             exp(-log_var) can't explode/vanish numerically
                             (default: 4.0 → precision ranges over roughly
                             [exp(-4), exp(4)] ≈ [0.018, 54.6])
    """

    def __init__(
        self,
        num_stat_features: int = 6,
        num_losses: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        max_log_var: float = 4.0,
    ):
        super().__init__()
        self.num_losses = num_losses
        self.max_log_var = max_log_var

        self.net = nn.Sequential(
            nn.Linear(num_stat_features, hidden_dim),
            nn.LayerNorm(hidden_dim),          # stabilize across feature scales
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_losses),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize the final linear layer to output zero, so every sample
        starts with log_var = 0 (i.e. precision = 1, equal treatment)."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        last_linear = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, num_stat_features)

        Returns:
            log_var  : (B, num_losses) — unconstrained, clamped to
                       [-max_log_var, max_log_var]
        """
        log_var = self.net(features)  # (B, num_losses)
        return log_var.clamp(min=-self.max_log_var, max=self.max_log_var)
