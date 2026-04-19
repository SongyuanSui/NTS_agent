from .dataset_registry import get_dataset_loader, list_dataset_loaders, register_dataset_loader
from .loaders.classification_loader import (
    DEFAULT_UCR2015_DIR,
    UCR2015ClassificationLoader,
    list_ucr2015_datasets,
    load_ucr2015_as_samples,
    load_ucr2015_local,
)
from .transforms import remap_labels_zero_based
from .adapters.univariate_adapter import array_split_to_samples as to_time_series_samples

__all__ = [
    "DEFAULT_UCR2015_DIR",
    "UCR2015ClassificationLoader",
    "get_dataset_loader",
    "list_dataset_loaders",
    "register_dataset_loader",
    "list_ucr2015_datasets",
    "load_ucr2015_as_samples",
    "load_ucr2015_local",
    "remap_labels_zero_based",
    "to_time_series_samples",
]
