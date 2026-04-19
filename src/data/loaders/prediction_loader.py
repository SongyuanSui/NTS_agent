from __future__ import annotations

from pathlib import Path
from typing import Any

from core.enums import TaskType
from data.dataset_base import DatasetLoaderBase


class NotImplementedPredictionLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.PREDICTION

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> dict:
        raise NotImplementedError("Prediction dataset loader is not implemented yet.")
