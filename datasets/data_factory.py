"""
DataLoader factory.

Centralizes DataLoader construction so that Trainer code
remains dataset-agnostic.
"""

from torch.utils.data import DataLoader
from .ett_dataset import ETTDataset
from .generic_dataset import GenericCSVDataset


# Registry: dataset name -> (dataset class, extra constructor kwargs).
# Add new datasets here as needed.
_DATASET_REGISTRY = {
    "ETTh1": (ETTDataset, {}),
    "ETTh2": (ETTDataset, {}),
    "ETTm1": (ETTDataset, {}),
    "ETTm2": (ETTDataset, {}),
    "Electricity": (GenericCSVDataset, {}),
    "Traffic": (GenericCSVDataset, {}),
    "Exchange": (GenericCSVDataset, {}),
    "Weather": (GenericCSVDataset, {}),
}


def build_dataloader(config, split: str) -> DataLoader:
    """
    Build a DataLoader for the given split using the dataset specified in config.

    Args:
        config : experiment config object (must expose data_name, data_path,
                 seq_len, pred_len, features, target, batch_size, num_workers)
        split  : one of "train", "val", "test"

    Returns:
        DataLoader
    """
    data_name = config.data_name
    assert data_name in _DATASET_REGISTRY, (
        f"Dataset '{data_name}' is not registered. "
        f"Available: {list(_DATASET_REGISTRY.keys())}"
    )

    dataset_cls, extra_kwargs = _DATASET_REGISTRY[data_name]
    dataset = dataset_cls(
        root_path=config.data_path,
        data_name=data_name,
        split=split,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        features=config.features,
        target=config.target,
        scale=True,
        **extra_kwargs,
    )

    shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader
