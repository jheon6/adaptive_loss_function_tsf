"""
ETT (Electricity Transformer Temperature) Dataset Loader.

Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (minutely).
Performs standard train/val/test split and z-score normalization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base_dataset import WindowedTimeSeriesDataset


# Official ETT split boundaries (number of time steps)
ETT_SPLIT = {
    "ETTh1": {"train": 12 * 30 * 24, "val": 4 * 30 * 24, "test": 4 * 30 * 24},
    "ETTh2": {"train": 12 * 30 * 24, "val": 4 * 30 * 24, "test": 4 * 30 * 24},
    "ETTm1": {"train": 12 * 30 * 24 * 4, "val": 4 * 30 * 24 * 4, "test": 4 * 30 * 24 * 4},
    "ETTm2": {"train": 12 * 30 * 24 * 4, "val": 4 * 30 * 24 * 4, "test": 4 * 30 * 24 * 4},
}


class ETTDataset(WindowedTimeSeriesDataset):
    """
    Sliding-window dataset for ETT benchmark.

    Each sample is a tuple (x_enc, x_dec, y) where:
        x_enc : (seq_len, num_features)   — encoder input
        y     : (pred_len, num_features)  — prediction target

    x_dec is included for compatibility with encoder-decoder backbones
    (e.g. Informer, Autoformer). DLinear only uses x_enc.
    """

    def __init__(
        self,
        root_path: str,
        data_name: str = "ETTh1",
        split: str = "train",
        seq_len: int = 336,
        pred_len: int = 96,
        features: str = "M",          # "M" = multivariate, "S" = univariate (OT only)
        target: str = "OT",
        scale: bool = True,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert data_name in ETT_SPLIT, f"Unknown dataset: {data_name}"
        assert features in ("M", "S"), f"features must be 'M' or 'S'"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale

        self._load_and_split(root_path, data_name, split)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_split(self, root_path: str, data_name: str, split: str):
        csv_path = os.path.join(root_path, f"{data_name}.csv")
        df = pd.read_csv(csv_path)

        # Drop the date column (first column)
        df = df.drop(columns=[df.columns[0]])

        # Feature selection
        if self.features == "S":
            df = df[[self.target]]

        # Compute split indices
        borders = ETT_SPLIT[data_name]
        n_train = borders["train"]
        n_val = borders["val"]

        if split == "train":
            start, end = 0, n_train
        elif split == "val":
            start, end = n_train - self.seq_len, n_train + n_val
        else:  # test
            start, end = n_train + n_val - self.seq_len, len(df)

        # Fit scaler on train portion only
        if self.scale:
            train_data = df.iloc[:n_train].values.astype(np.float32)
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            data = self.scaler.transform(df.values.astype(np.float32))
        else:
            self.scaler = None
            data = df.values.astype(np.float32)

        self.data = data[start:end]
