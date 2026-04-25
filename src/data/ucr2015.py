from __future__ import annotations

# Compatibility module kept for existing imports.
from data.adapters.univariate_adapter import array_split_to_samples as to_time_series_samples
from data.loaders.classification_univariate_loader import (
    DEFAULT_UCR2015_DIR,
    list_ucr2015_datasets,
    load_ucr2015_as_samples,
    load_ucr2015_local,
)
from data.transforms import remap_labels_zero_based

__all__ = [
    "DEFAULT_UCR2015_DIR",
    "list_ucr2015_datasets",
    "load_ucr2015_as_samples",
    "load_ucr2015_local",
    "remap_labels_zero_based",
    "to_time_series_samples",
]
