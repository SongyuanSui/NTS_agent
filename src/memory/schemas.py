# src/memory/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from core.enums import RepresentationType, TaskType


Metadata = dict[str, Any]


@dataclass(slots=True)
class MemoryViewRecord:
    """
    One view-specific record stored inside memory.

    payload examples
    ----------------
    - TS view: np.ndarray
    - SUMMARY view: str
    - STATISTIC view: dict[str, float] or np.ndarray
    """

    representation_type: RepresentationType
    payload: Any
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.representation_type, RepresentationType):
            raise TypeError("representation_type must be a RepresentationType.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class MemoryEntry:
    """
    Canonical entry stored in memory bank.

    A single entry corresponds to one sample and one channel.
    For univariate data, channel_id should usually be 0.
    """

    entry_id: str
    sample_id: str
    task_type: TaskType
    channel_id: int
    label: Optional[Any] = None
    ts_view: Optional[MemoryViewRecord] = None
    summary_view: Optional[MemoryViewRecord] = None
    statistic_view: Optional[MemoryViewRecord] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id:
            raise ValueError("entry_id must be a non-empty string.")

        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type must be a TaskType.")

        if not isinstance(self.channel_id, int) or self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        for attr_name in ("ts_view", "summary_view", "statistic_view"):
            value = getattr(self, attr_name)
            if value is not None and not isinstance(value, MemoryViewRecord):
                raise TypeError(f"{attr_name} must be a MemoryViewRecord or None.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def get_view(self, representation_type: RepresentationType) -> Optional[MemoryViewRecord]:
        if representation_type == RepresentationType.TS:
            return self.ts_view
        if representation_type == RepresentationType.SUMMARY:
            return self.summary_view
        if representation_type == RepresentationType.STATISTIC:
            return self.statistic_view
        raise ValueError(f"Unsupported representation_type: {representation_type}")

    def has_view(self, representation_type: RepresentationType) -> bool:
        return self.get_view(representation_type) is not None