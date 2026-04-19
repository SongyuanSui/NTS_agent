from __future__ import annotations

from pathlib import Path
from typing import Any

from core.enums import TaskType
from data.dataset_base import DatasetLoaderBase


class NotImplementedAnomalySequenceLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_SEQUENCE

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> dict:
        raise NotImplementedError("Anomaly-sequence dataset loader is not implemented yet.")


class NotImplementedAnomalyWindowLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_WINDOW

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> dict:
        raise NotImplementedError("Anomaly-window dataset loader is not implemented yet.")


# Backward-compatible alias for earlier code paths.
NotImplementedAnomalyLoader = NotImplementedAnomalyWindowLoader
