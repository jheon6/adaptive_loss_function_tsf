from .components import MSELoss, MAELoss, TrendLoss, FrequencyLoss
from .adaptive_loss import AdaptiveLossWeighting
from .fixed_loss import FixedWeightLoss
from .distributional_loss import SkewNormalNLLLoss
from .base_loss import BaseLoss


def build_loss(config):
    """
    Instantiate the loss function specified in config.loss_type.

    loss_type options:
        "mse_only"              → FixedWeightLoss(mse=1.0)
        "mse_mae_fixed"         → FixedWeightLoss(mse=0.5, mae=0.5)
        "mse_trend_fixed"       → FixedWeightLoss(mse=0.5, trend=0.5)
        "mse_trend_freq_fixed"  → FixedWeightLoss(mse=0.34, trend=0.33, frequency=0.33)
        "mse_mae_trend_fixed"   → FixedWeightLoss(mse=0.33, mae=0.33, trend=0.33)
        "adaptive"              → AdaptiveLossWeighting(...)
        "skew_normal_nll"       → SkewNormalNLLLoss(...)

    Note: AdaptiveLossWeighting only weights {mse, mae, trend} (no frequency).
    "mse_mae_trend_fixed" mirrors that same 3-component set with static equal
    weights, so it is the fair fixed-weight counterpart to compare against.
    """
    lt = config.loss_type

    if lt == "mse_only":
        return FixedWeightLoss(weights={"mse": 1.0})

    elif lt == "mse_mae_fixed":
        return FixedWeightLoss(weights={"mse": 0.5, "mae": 0.5})

    elif lt == "mse_trend_fixed":
        return FixedWeightLoss(weights={"mse": 0.5, "trend": 0.5})

    elif lt == "mse_trend_freq_fixed":
        return FixedWeightLoss(weights={"mse": 1/3, "trend": 1/3, "frequency": 1/3})

    elif lt == "mse_mae_trend_fixed":
        return FixedWeightLoss(weights={"mse": 1/3, "mae": 1/3, "trend": 1/3})

    elif lt == "adaptive":
        return AdaptiveLossWeighting(
            num_stat_features=getattr(config, "num_stat_features", 6),
            hidden_dim=getattr(config, "weight_gen_hidden_dim", 64),
            dropout=getattr(config, "weight_gen_dropout", 0.1),
            max_log_var=getattr(config, "weight_gen_max_log_var", 4.0),
            loss_norm_momentum=getattr(config, "loss_norm_momentum", 0.9),
            feature_norm_momentum=getattr(config, "feature_norm_momentum", 0.9),
        )

    elif lt == "skew_normal_nll":
        return SkewNormalNLLLoss(
            num_stat_features=getattr(config, "num_stat_features", 6),
            hidden_dim=getattr(config, "weight_gen_hidden_dim", 64),
            dropout=getattr(config, "weight_gen_dropout", 0.1),
            max_log_var=getattr(config, "weight_gen_max_log_var", 4.0),
            max_skew=getattr(config, "max_skew", 5.0),
            feature_norm_momentum=getattr(config, "feature_norm_momentum", 0.9),
        )

    else:
        raise ValueError(
            f"Unknown loss_type '{lt}'. "
            "Choose from: mse_only, mse_mae_fixed, mse_trend_fixed, "
            "mse_trend_freq_fixed, mse_mae_trend_fixed, adaptive, skew_normal_nll"
        )
