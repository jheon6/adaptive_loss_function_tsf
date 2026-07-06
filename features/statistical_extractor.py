"""
Statistical Feature Extractor for Adaptive Loss Weighting.

Computes per-(sample, channel) statistical descriptors from the encoder
input x_enc. All operations are implemented using PyTorch so that gradients
flow through the feature vector back to the WeightGeneratorMLP during
training.

Features (6 total, computed per sample per channel):
    1. Variance          — spread of the signal
    2. Skewness          — asymmetry of the amplitude distribution
    3. Kurtosis          — tail heaviness of the amplitude distribution
    4. ACF (lag-1)       — first-order autocorrelation
    5. Spectral Entropy  — concentration of energy in frequency domain
    6. Trend Strength    — ratio of variance explained by a linear trend

Output shape: (B, C, 6)
"""

import torch
import torch.nn as nn


class StatisticalFeatureExtractor(nn.Module):
    """
    Differentiable extractor of statistical time-series features.

    Input : x_enc  (B, T, C)
    Output: features (B, C, num_features=6)

    Kept per-channel (not averaged across C): in a multivariate window, one
    channel can be smooth/trending while another is noisy at the same
    timestep, and averaging across channels collapses exactly that
    heterogeneity before the gate ever sees it.

    Raw features have very different scales (variance and kurtosis are
    unbounded, acf/spectral_entropy/trend_strength are bounded in [-1, 1]
    or [0, 1]). Without normalization the MLP's first Linear layer sees
    wildly different input magnitudes across samples, which can make its
    logits — and therefore the predicted weights — swing erratically.
    An EMA-based per-feature z-score (same pattern as the loss normalization
    in AdaptiveLossWeighting) keeps every feature at roughly unit scale.
    """

    NUM_FEATURES = 6
    FEATURE_NAMES = [
        "variance",
        "skewness",
        "kurtosis",
        "acf_lag1",
        "spectral_entropy",
        "trend_strength",
    ]

    def __init__(self, eps: float = 1e-8, norm_momentum: float = 0.9):
        super().__init__()
        self.eps = eps  # numerical stability guard
        self.norm_momentum = norm_momentum

        # EMA running stats for per-feature normalization.
        # Initialized to mean=0, std=1 so normalization is a no-op at the start.
        self.register_buffer("running_mean", torch.zeros(self.NUM_FEATURES))
        self.register_buffer("running_std", torch.ones(self.NUM_FEATURES))

    def _update_running_stats(self, features: torch.Tensor):
        """
        Update EMA mean/std with the current batch's stats (detached).
        features: (B, C, num_features) — reduce over both batch and channel
        so running_mean/std stay per-feature-type (shape (num_features,)),
        not per-channel-position (channel identity isn't meaningful to
        normalize against; it's just "which variate", not an ordered axis).
        """
        batch_mean = features.mean(dim=(0, 1)).detach()
        batch_std = features.std(dim=(0, 1), unbiased=False).detach()
        self.running_mean = self.norm_momentum * self.running_mean + (1.0 - self.norm_momentum) * batch_mean
        self.running_std = self.norm_momentum * self.running_std + (1.0 - self.norm_momentum) * batch_std

    # ------------------------------------------------------------------
    # Per-feature helpers  (all operate on a (B, T, C) tensor)
    # ------------------------------------------------------------------

    def _variance(self, x: torch.Tensor) -> torch.Tensor:
        """Unbiased variance along the time axis. Returns (B, C)."""
        return x.var(dim=1, unbiased=True)

    def _skewness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pearson's moment coefficient of skewness.
        skew = E[(x - mu)^3] / std^3
        Returns (B, C).
        """
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=True).clamp(min=self.eps)
        z = (x - mu) / std
        return z.pow(3).mean(dim=1)

    def _kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Excess kurtosis (Fisher's definition, subtract 3).
        kurt = E[(x - mu)^4] / std^4 - 3
        Returns (B, C).
        """
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=True).clamp(min=self.eps)
        z = (x - mu) / std
        return z.pow(4).mean(dim=1) - 3.0

    def _acf_lag1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lag-1 autocorrelation coefficient.
        acf(1) = cov(x_t, x_{t-1}) / var(x)
        Returns (B, C).
        """
        mu = x.mean(dim=1, keepdim=True)
        x_centered = x - mu                          # (B, T, C)
        # Numerator: covariance between consecutive time steps
        cov = (x_centered[:, 1:, :] * x_centered[:, :-1, :]).mean(dim=1)
        var = x.var(dim=1, unbiased=True).clamp(min=self.eps)
        return cov / var

    def _spectral_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalized spectral entropy based on FFT magnitude spectrum.
        H = -sum(p * log(p)) / log(N_freq)
        where p is the normalized power spectral density.
        Returns (B, C).
        """
        # rfft returns only the non-redundant half of the spectrum
        fft_mag = torch.fft.rfft(x, dim=1).abs()    # (B, N_freq, C)
        power = fft_mag.pow(2)
        power_sum = power.sum(dim=1, keepdim=True).clamp(min=self.eps)
        p = power / power_sum                        # normalized PSD

        n_freq = p.shape[1]
        # Shannon entropy (clamp p to avoid log(0))
        entropy = -(p * torch.log(p.clamp(min=self.eps))).sum(dim=1)
        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(float(n_freq), device=x.device))
        return entropy / max_entropy.clamp(min=self.eps)

    def _trend_strength(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trend strength measured as R² of a least-squares linear fit.
        R² = 1 - SS_res / SS_tot

        A value close to 1 means the series is strongly trended;
        close to 0 means no linear trend.
        Returns (B, C).
        """
        B, T, C = x.shape
        device = x.device

        # Build the design matrix [1, t] for each time step
        t = torch.arange(T, dtype=x.dtype, device=device)
        t_norm = (t - t.mean()) / (t.std() + self.eps)          # (T,)
        ones = torch.ones(T, dtype=x.dtype, device=device)
        # A: (T, 2)
        A = torch.stack([ones, t_norm], dim=1)

        # Closed-form OLS: beta = (A^T A)^{-1} A^T x
        # A^T A is (2, 2); A^T x is (2, B*C)
        x_flat = x.reshape(B * C, T).T           # (T, B*C)
        AtA = A.T @ A                             # (2, 2)
        Atx = A.T @ x_flat                        # (2, B*C)

        # Solve using lstsq for numerical stability
        beta = torch.linalg.lstsq(AtA, Atx).solution  # (2, B*C)

        # Fitted values and residuals
        x_hat = (A @ beta).T.reshape(B, T, C)    # (B, T, C)
        ss_res = ((x - x_hat) ** 2).sum(dim=1)
        ss_tot = ((x - x.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=self.eps)
        return r2.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc : (B, T, C)

        Returns:
            features : (B, C, 6) — per-channel, NOT averaged across channels.
                       Averaging across channels was discarding exactly the
                       cross-channel heterogeneity a per-channel gate needs
                       (e.g. a noisy channel vs. a smooth trending channel in
                       the same multivariate window look identical once
                       averaged). Downstream (WeightGeneratorMLP) operates on
                       the last dim only, so it broadcasts over (B, C) as-is.
        """
        feats = [
            self._variance(x_enc),          # (B, C)
            self._skewness(x_enc),           # (B, C)
            self._kurtosis(x_enc),           # (B, C)
            self._acf_lag1(x_enc),           # (B, C)
            self._spectral_entropy(x_enc),   # (B, C)
            self._trend_strength(x_enc),     # (B, C)
        ]

        features = torch.stack(feats, dim=2)  # (B, C, 6)

        if self.training:
            self._update_running_stats(features)

        normalized = (features - self.running_mean) / (self.running_std + self.eps)
        return normalized
