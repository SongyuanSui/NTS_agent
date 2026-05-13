from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ts_logging.event_log import EventLogger


@dataclass
class ExecutionContext:
    """
    Mutable runtime context shared across pipeline stages.

    The pipeline and agent APIs in this project accept plain dictionaries.
    This wrapper keeps that compatibility while offering a typed place for
    common fields such as logger, event_logger, run_id, and artifact paths.
    """

    pipeline_name: Optional[str] = None
    run_id: Optional[str] = None
    stage: Optional[str] = None
    sample_id: Optional[str] = None
    query_id: Optional[str] = None
    task_type: Optional[str] = None
    logger: Optional[logging.Logger] = None
    event_logger: Optional[EventLogger] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifacts is None:
            self.artifacts = {}
        if self.state is None:
            self.state = {}
        if not isinstance(self.artifacts, dict):
            self.artifacts = dict(self.artifacts)
        if not isinstance(self.state, dict):
            self.state = dict(self.state)
        if self.logger is not None and not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger or None.")
        if self.event_logger is not None and not isinstance(self.event_logger, EventLogger):
            raise TypeError("event_logger must be an EventLogger or None.")

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> ExecutionContext:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError("ExecutionContext.from_dict expects a dict or None.")

        known = {
            "pipeline_name",
            "run_id",
            "stage",
            "sample_id",
            "query_id",
            "task_type",
            "logger",
            "event_logger",
            "artifacts",
        }
        state = dict(data.get("state", {}))
        for key, value in data.items():
            if key not in known and key != "state":
                state[key] = value

        return cls(
            pipeline_name=data.get("pipeline_name"),
            run_id=data.get("run_id"),
            stage=data.get("stage"),
            sample_id=data.get("sample_id"),
            query_id=data.get("query_id"),
            task_type=data.get("task_type"),
            logger=data.get("logger"),
            event_logger=data.get("event_logger"),
            artifacts=dict(data.get("artifacts", {})),
            state=state,
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.state)
        data.update(
            {
                "pipeline_name": self.pipeline_name,
                "run_id": self.run_id,
                "stage": self.stage,
                "sample_id": self.sample_id,
                "query_id": self.query_id,
                "task_type": self.task_type,
                "artifacts": self.artifacts,
            }
        )
        if self.logger is not None:
            data["logger"] = self.logger
        if self.event_logger is not None:
            data["event_logger"] = self.event_logger
        return data

    def child(self, stage: str, **updates: Any) -> ExecutionContext:
        data = self.to_dict()
        data["stage"] = stage
        data.update(updates)
        return ExecutionContext.from_dict(data)

    def set_artifact(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("artifact key must be a non-empty string.")
        self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
