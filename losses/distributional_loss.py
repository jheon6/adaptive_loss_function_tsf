"""
SkewNormalNLLLoss — distributional alternative to AdaptiveLossWeighting.

Pipeline per forward pass:
    1. StatisticalFeatureExtractor(x_enc)  →  features       (B, C, 6)
    2. WeightGeneratorMLP(features)        →  (log_sigma, skew_raw)  (B, C, 2)
    3. sigma = exp(log_sigma); alpha = max_skew * tanh(skew_raw)
    4. Skew-normal NLL of target given location=pred, scale=sigma, shape=alpha
    5. Mean over time, then batch + channel mean → scalar

Why this exists (vs. weighting MSE/MAE/Trend by log-variance):
    AdaptiveLossWeighting conditions its weight generator on 6 statistical
    features (variance, skewness, kurtosis, ACF-lag1, spectral entropy, trend
    strength), but only combines MSE/MAE/Trend — none of which can reflect
    skewness or kurtosis (MSE sees the 2nd moment only, MAE ignores shape
    entirely). The feature richness and the loss's ability to *use* that
    richness were mismatched.

    Modeling the residual as skew-normal instead closes that gap for
    skewness specifically: skew becomes a distribution *shape* parameter
    the model is scored on directly (via NLL), not a scalar that reweights
    two shape-blind losses. Kurtosis/ACF/spectral-entropy are still
    unaddressed by this class — see project memory for scope notes.

Point prediction vs. distribution mean:
    `pred` (the backbone's output) is used as the skew-normal's location
    parameter (xi), not its mean. For a skew-normal, mean = xi + sigma *
    delta * sqrt(2/pi), delta = alpha / sqrt(1 + alpha^2) — these coincide
    only when alpha = 0. Training must use raw `pred` as xi (that is what
    the NLL is actually parameterized by), but reporting MSE/MAE against
    raw `pred` would silently bias skewed channels. `point_forecast()`
    below applies the correction and is the value callers should use
    wherever a deterministic forecast is needed for evaluation.
"""

import math

import torch
import torch.nn as nn

from features.statistical_extractor import StatisticalFeatureExtractor
from weight_generator.mlp_generator import WeightGeneratorMLP

_LOG_2PI = math.log(2 * math.pi)
_SQRT_2_OVER_PI = math.sqrt(2 / math.pi)


class SkewNormalNLLLoss(nn.Module):
    """
    Per-(sample, channel) skew-normal negative log-likelihood, with
    location = backbone prediction and (scale, shape) conditioned on
    statistical input features.

    Args:
        num_stat_features     : dimensionality of the statistical feature vector (default: 6)
        hidden_dim             : hidden dim of the weight generator MLP (default: 64)
        dropout                : dropout in the weight generator (default: 0.1)
        max_log_var             : clamp range for predicted log-sigma (default: 4.0)
        max_skew                : bound on the skew-normal shape parameter alpha (default: 5.0)
        feature_norm_momentum   : EMA momentum for statistical feature normalization (default: 0.9)
    """

    def __init__(
        self,
        num_stat_features: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        max_log_var: float = 4.0,
        max_skew: float = 5.0,
        feature_norm_momentum: float = 0.9,
    ):
        super().__init__()

        self.max_skew = max_skew

        self.feature_extractor = StatisticalFeatureExtractor(norm_momentum=feature_norm_momentum)

        # Reused as a generic features -> 2 params MLP; outputs are
        # reinterpreted below as (log_sigma, skew_raw) rather than
        # per-loss log-variance.
        self.weight_generator = WeightGeneratorMLP(
            num_stat_features=num_stat_features,
            num_losses=2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_log_var=max_log_var,
        )

    # ------------------------------------------------------------------
    # Shared param computation
    # ------------------------------------------------------------------

    def _dist_params(self, x_enc: torch.Tensor):
        """Returns (log_sigma, sigma, alpha), each shaped (B, C)."""
        features = self.feature_extractor(x_enc)              # (B, C, 6)
        raw = self.weight_generator(features.detach())        # (B, C, 2)
        log_sigma, skew_raw = raw[..., 0], raw[..., 1]         # (B, C) each
        sigma = torch.exp(log_sigma)
        alpha = self.max_skew * torch.tanh(skew_raw)
        return log_sigma, sigma, alpha

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,    # (B, pred_len, C)
        target: torch.Tensor,  # (B, pred_len, C)
        x_enc: torch.Tensor,   # (B, seq_len, C)
    ) -> torch.Tensor:
        log_sigma, sigma, alpha = self._dist_params(x_enc)    # (B, C) each

        sigma_b = sigma.unsqueeze(1)     # (B, 1, C) — broadcast over pred_len
        alpha_b = alpha.unsqueeze(1)     # (B, 1, C)
        log_sigma_b = log_sigma.unsqueeze(1)

        z = (target - pred) / sigma_b    # (B, pred_len, C)

        # NLL of skew-normal(location=pred, scale=sigma, shape=alpha):
        #   pdf = (2/sigma) * phi(z) * Phi(alpha*z)
        #   nll = log(sigma) - log(2) - log_phi(z) - log_Phi(alpha*z)
        #       = log_sigma + 0.5*z^2 + 0.5*log(2*pi) - log(2) - log_ndtr(alpha*z)
        log_Phi = torch.special.log_ndtr(alpha_b * z)
        nll = log_sigma_b + 0.5 * z.pow(2) + 0.5 * _LOG_2PI - math.log(2) - log_Phi

        return nll.mean(dim=1).mean()

    # ------------------------------------------------------------------
    # Point forecast (mean-corrected, for MSE/MAE evaluation)
    # ------------------------------------------------------------------

    def point_forecast(self, pred: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Skew-normal mean, added to the raw location prediction:
            mean = xi + sigma * delta * sqrt(2/pi),  delta = alpha / sqrt(1 + alpha^2)

        Use this (not raw `pred`) whenever a deterministic forecast is
        needed for MSE/MAE-style evaluation — see class docstring.
        """
        with torch.no_grad():
            _, sigma, alpha = self._dist_params(x_enc)               # (B, C)
            delta = alpha / torch.sqrt(1.0 + alpha.pow(2))
            mean_shift = (sigma * delta * _SQRT_2_OVER_PI).unsqueeze(1)  # (B, 1, C)
            return pred + mean_shift

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def get_current_weights(self, x_enc: torch.Tensor) -> dict:
        """Average (sigma, skew) over batch and channel, for per-epoch logging."""
        with torch.no_grad():
            _, sigma, alpha = self._dist_params(x_enc)
        return {"sigma": sigma.mean().item(), "skew": alpha.mean().item()}
