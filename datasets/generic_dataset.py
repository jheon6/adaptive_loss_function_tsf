"""
Generic multivariate CSV dataset — for benchmarks that aren't in the ETT
family. Covers two raw formats:

    has_header=False (default): no header, no date column — just a comma-
        separated matrix (Electricity, Traffic, Exchange; source:
        laiguokun/multivariate-time-series-data).
    has_header=True: a normal header row with a leading date/timestamp
        column to drop (Weather; source: Jena Climate weather station).

Unlike ETTDataset, the split is always by ratio (0.7 / 0.1 / 0.2) rather than
fixed month boundaries, following the standard Autoformer/Informer benchmark
convention for these datasets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base_dataset import WindowedTimeSeriesDataset

SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


class GenericCSVDataset(WindowedTimeSeriesDataset):
    """
    Sliding-window dataset for header-less multivariate CSV benchmarks.

    Each sample is a tuple (x_enc, y) where:
        x_enc : (seq_len, num_features)   — encoder input
        y     : (pred_len, num_features)  — prediction target
    """

    def __init__(
        self,
        root_path: str,
        data_name: str,
        split: str = "train",
        seq_len: int = 336,
        pred_len: int = 96,
        features: str = "M",          # "M" = multivariate, "S" = univariate (last column only)
        target: str = None,
        scale: bool = True,
        has_header: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert features in ("M", "S"), f"features must be 'M' or 'S'"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.has_header = has_header

        self._load_and_split(root_path, data_name, split)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_split(self, root_path: str, data_name: str, split: str):
        csv_path = os.path.join(root_path, f"{data_name}.csv")
        if self.has_header:
            df = pd.read_csv(csv_path)
            df = df.drop(columns=[df.columns[0]])  # drop the leading date/timestamp column
        else:
            df = pd.read_csv(csv_path, header=None)

        # Feature selection — no named "OT" column here, so "S" mode
        # defaults to the last column unless a target index/name is given.
        if self.features == "S":
            target_col = self.target if self.target is not None else df.columns[-1]
            df = df[[target_col]]

        # Ratio-based split
        n = len(df)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        if split == "train":
            start, end = 0, n_train
        elif split == "val":
            start, end = n_train - self.seq_len, n_train + n_val
        else:  # test
            start, end = n_train + n_val - self.seq_len, n

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
