"""
Abstract base class for all loss components.

Every individual loss must:
  - Accept (pred, target) with shape (B, T, C)
  - Return a per-sample scalar loss of shape (B,)

Returning per-sample scalars (not a single mean) is critical:
the AdaptiveLossWeighting module weights each loss per sample before
reducing to a batch mean.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLoss(ABC, nn.Module):
    """
    Interface for individual loss components.

    Subclasses implement `forward` to return a (B,) tensor of
    unreduced per-sample losses.
    """

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,    # (B, T, C)
        target: torch.Tensor,  # (B, T, C)
    ) -> torch.Tensor:         # (B,)
        """Compute per-sample loss. Must NOT reduce across the batch."""
        ...
