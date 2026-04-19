from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.enums import TaskType
from core.interfaces import BaseDatasetLoaderInterface


class DatasetLoaderBase(BaseDatasetLoaderInterface, ABC):
    """Base implementation for all dataset loaders."""

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Task type this loader supports."""

    @abstractmethod
    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> Any:
        """Load one dataset and return a task-specific bundle."""
