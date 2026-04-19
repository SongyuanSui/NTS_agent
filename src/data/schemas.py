from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.schemas import TimeSeriesSample


@dataclass(slots=True)
class DatasetSplit:
    """Container for one dataset split (e.g., train/test)."""

    samples: list[TimeSeriesSample]
    split_name: str


@dataclass(slots=True)
class ClassificationDatasetBundle:
    """Canonical classification dataset bundle used by data loaders."""

    dataset_name: str
    train: DatasetSplit
    test: DatasetSplit
    label_map: Optional[dict[float, int]] = None
    metadata: dict = field(default_factory=dict)
