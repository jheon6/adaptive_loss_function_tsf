"""
Shared sliding-window Dataset base class.

Every dataset in this project (ETT, Electricity, Traffic, ...) reduces to the
same thing once loaded into a (T, C) numpy array: slide a window of length
seq_len + pred_len across it with stride 1. Only how the raw file is read and
split into train/val/test differs between datasets, so subclasses only need
to implement `_load_and_split` and set `self.seq_len`, `self.pred_len`,
`self.data`, `self.scaler`.
"""

import torch
from torch.utils.data import Dataset


class WindowedTimeSeriesDataset(Dataset):
    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        s = idx
        enc_end = s + self.seq_len
        pred_end = enc_end + self.pred_len

        x_enc = torch.tensor(self.data[s:enc_end], dtype=torch.float32)
        y = torch.tensor(self.data[enc_end:pred_end], dtype=torch.float32)

        return x_enc, y

    @property
    def num_features(self) -> int:
        return self.data.shape[1]

    def inverse_transform(self, data):
        """Undo z-score normalization (for final evaluation)."""
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)
