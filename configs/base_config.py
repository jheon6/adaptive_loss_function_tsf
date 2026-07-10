"""
Base experiment configuration using Python dataclasses.

Dataclasses provide type annotations and default values while remaining
plain Python objects — no external config library required.

A config can be constructed:
    1. Directly in Python:  cfg = ExperimentConfig(loss_type="adaptive")
    2. From a YAML file  :  cfg = ExperimentConfig.from_yaml("configs/experiments/adaptive_etth1.yaml")
    3. From CLI args     :  cfg = ExperimentConfig.from_args()
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------ #
    # Experiment identity
    # ------------------------------------------------------------------ #
    exp_id: str = "exp"               # unique run identifier
    save_dir: str = "outputs"         # root dir for checkpoints / logs
    seed: int = 42                    # RNG seed for reproducibility

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    data_name: str = "ETTh1"
    data_path: str = "data"           # directory containing the CSV files
    features: str = "M"               # "M" = multivariate, "S" = univariate
    target: str = "OT"                # target column (used only when features="S")
    seq_len: int = 336                # encoder input length
    pred_len: int = 96                # forecast horizon
    num_features: int = 7             # number of variates (ETTh1 = 7)

    # ------------------------------------------------------------------ #
    # Model (backbone)
    # ------------------------------------------------------------------ #
    model_name: str = "DLinear"
    moving_avg: int = 25              # DLinear decomposition kernel size

    # PatchTST-only hyperparameters
    patch_len: int = 16
    patch_stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    patchtst_dropout: float = 0.1

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    loss_type: str = "adaptive"
    # adaptive-only hyper-parameters
    num_stat_features: int = 6
    weight_gen_hidden_dim: int = 64
    weight_gen_dropout: float = 0.1
    weight_gen_max_log_var: float = 4.0  # clamp range for predicted log-variance
    weight_gen_lr_scale: float = 0.5     # weight generator LR = learning_rate * this scale
    loss_norm_momentum: float = 0.9 # EMA momentum for per-loss scale normalization
    feature_norm_momentum: float = 0.9   # EMA momentum for statistical feature normalization

    # ------------------------------------------------------------------ #
    # Bilevel (meta-learn the weight generator against val loss)
    # ------------------------------------------------------------------ #
    bilevel: bool = False              # if True, use BilevelTrainer instead of Trainer
    bilevel_inner_lr: Optional[float] = None  # virtual backbone step size; None -> learning_rate

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 7                 # early stopping patience; 0 = disabled (run full epochs)
    num_workers: int = 0

    # ------------------------------------------------------------------ #
    # Hardware
    # ------------------------------------------------------------------ #
    device: str = "cuda"              # "cuda" or "cpu"; auto-detected at runtime

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from a YAML file, overriding only the specified fields."""
        with open(path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        cfg = cls()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key in YAML: '{key}'")
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Build from an argparse Namespace, ignoring None values."""
        cfg = cls()
        for key, value in vars(args).items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_yaml(self, path: str):
        """Serialize config to YAML for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def __post_init__(self):
        """Auto-detect device if not explicitly set to 'cpu'."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
