"""
PatchTST stub — placeholder for future integration.
Nie et al., ICLR 2023.  https://arxiv.org/abs/2211.14730

To integrate a full PatchTST implementation:
1. Implement or import the PatchTST backbone here.
2. Register it in models/__init__.py:
       from .patchtst import PatchTST
       _MODEL_REGISTRY["PatchTST"] = PatchTST
3. Add a corresponding experiment YAML under configs/experiments/.

The Adaptive Loss Weighting framework is backbone-agnostic:
no changes to losses/, features/, or weight_generator/ are needed.
"""

import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(
            "PatchTST is not yet implemented. "
            "See the docstring in this file for integration instructions."
        )

    def forward(self, x_enc):
        raise NotImplementedError
