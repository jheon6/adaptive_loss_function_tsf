"""
PatchTST — A Time Series is Worth 64 Words: Long-Term Forecasting with
Transformers. Nie et al., ICLR 2023. https://arxiv.org/abs/2211.14730

Channel-independent variant: each channel is treated as its own univariate
series (folded into the batch dimension), instance-normalized, split into
overlapping patches, embedded, and passed through a standard Transformer
encoder. A single linear head flattens the encoder output into the forecast
horizon. No cross-channel attention — every channel shares the same weights,
same as DLinear's channel-independent design.

Input shape  : (B, seq_len, num_features)
Output shape : (B, pred_len, num_features)
"""

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Learnable additive positional embedding, one vector per patch position."""

    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class PatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features

        self.patch_len = getattr(config, "patch_len", 16)
        self.stride = getattr(config, "patch_stride", 8)
        d_model = getattr(config, "d_model", 128)
        n_heads = getattr(config, "n_heads", 8)
        e_layers = getattr(config, "e_layers", 2)
        d_ff = getattr(config, "d_ff", 256)
        dropout = getattr(config, "patchtst_dropout", 0.1)

        # Replication-pad the tail by `stride` steps (standard PatchTST trick)
        # so the last real observation is covered by its own patch.
        self.padding = self.stride
        self.num_patches = (self.seq_len + self.padding - self.patch_len) // self.stride + 1

        self.pad_layer = nn.ReplicationPad1d((0, self.padding))
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_encoding = _PositionalEncoding(self.num_patches, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.head = nn.Linear(self.num_patches * d_model, self.pred_len)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc : (B, seq_len, C)

        Returns:
            pred  : (B, pred_len, C)
        """
        B, _, C = x_enc.shape

        # Instance normalization (RevIN-style): per-sample, per-channel.
        # Critical for transformer-based TSF backbones — removes the
        # sample's own level/scale before attention, added back at the end.
        mean = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-5)
        x = (x_enc - mean) / std

        # Channel-independent: fold channel into the batch dimension.
        x = x.permute(0, 2, 1)                   # (B, C, T)
        x = self.pad_layer(x)                    # (B, C, T + stride)
        x = x.unfold(-1, self.patch_len, self.stride)   # (B, C, num_patches, patch_len)
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        x = self.patch_embed(x)                  # (B*C, num_patches, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x)                      # (B*C, num_patches, d_model)

        x = x.reshape(B * C, -1)                 # (B*C, num_patches * d_model)
        out = self.head(x)                       # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        # Reverse instance normalization.
        out = out * std + mean
        return out
