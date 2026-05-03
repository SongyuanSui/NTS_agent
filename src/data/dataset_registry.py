from __future__ import annotations

from data.dataset_base import DatasetLoaderBase
from data.loaders.anomaly_loader import (
    SKABAnomalySequenceLoader,
    SKABAnomalyWindowLoader,
)
from data.loaders.classification_multivariate_loader import (
    UEAMultivariateClassificationLoader,
)
from data.loaders.classification_univariate_loader import UCR2015ClassificationLoader


_DATASET_REGISTRY: dict[str, DatasetLoaderBase] = {
    "ucr2015": UCR2015ClassificationLoader(),
    "uea": UEAMultivariateClassificationLoader(),
    "skab_sequence": SKABAnomalySequenceLoader(),
    "skab_window": SKABAnomalyWindowLoader(),
}


def register_dataset_loader(name: str, loader: DatasetLoaderBase) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Loader name must be non-empty")
    if not isinstance(loader, DatasetLoaderBase):
        raise TypeError("loader must be an instance of DatasetLoaderBase")
    _DATASET_REGISTRY[key] = loader


def get_dataset_loader(name: str) -> DatasetLoaderBase:
    key = name.strip().lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset loader '{name}'. Available: {available}")
    return _DATASET_REGISTRY[key]


def list_dataset_loaders() -> list[str]:
    return sorted(_DATASET_REGISTRY.keys())
