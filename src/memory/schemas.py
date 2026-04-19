from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import RepresentationType, TaskType


Metadata = dict[str, Any]


@dataclass(slots=True)
class MemoryEntry:
    """
    Minimal memory schema aligned with representation outputs.

    For current stat-feature retrieval experiments, `statistic_view` is the
    primary payload used by retrievers.
    """

    entry_id: str
    sample_id: str
    channel_id: int
    task_type: TaskType | str
    label: Optional[Any] = None
    statistic_view: Optional[dict[str, float] | list[float] | tuple[float, ...] | Any] = None
    ts_view: Optional[Any] = None
    summary_view: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id:
            raise ValueError("entry_id must be a non-empty string.")

        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        self.channel_id = int(self.channel_id)
        if self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)

        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type must be a TaskType or a valid string.")

        if self.summary_view is not None and not isinstance(self.summary_view, str):
            raise TypeError("summary_view must be a string if provided.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

        if self.ts_view is None and self.summary_view is None and self.statistic_view is None:
            raise ValueError("At least one representation view must be provided.")

    def has_view(self, representation_type: RepresentationType | str) -> bool:
        if isinstance(representation_type, str):
            representation_type = RepresentationType(representation_type)

        if representation_type == RepresentationType.TS:
            return self.ts_view is not None

        if representation_type == RepresentationType.SUMMARY:
            return self.summary_view is not None

        if representation_type == RepresentationType.STATISTIC:
            return self.statistic_view is not None

        return False