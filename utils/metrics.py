"""
Evaluation metrics for time series forecasting.

All functions operate on numpy arrays of shape (N, T, C) or (N*T*C,).
Inverse-transform (un-normalize) before calling these if you want
metrics on the original scale.
"""

import numpy as np


def _flatten(pred: np.ndarray, target: np.ndarray):
    """Flatten to 1D for scalar metric computation."""
    return pred.flatten(), target.flatten()


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _flatten(pred, target)
    return float(np.mean((p - t) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _flatten(pred, target)
    return float(np.mean(np.abs(p - t)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, target)))


def mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    p, t = _flatten(pred, target)
    return float(np.mean(np.abs((p - t) / (np.abs(t) + eps))) * 100)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute all standard TSF metrics.

    Args:
        pred   : (N, T, C) or (N*T*C,)
        target : same shape as pred

    Returns:
        dict with keys: mse, mae, rmse, mape
    """
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "mape": mape(pred, target),
    }
