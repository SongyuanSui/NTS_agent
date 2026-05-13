from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    """Small structured config object for experiment runners."""

    name: str
    task: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ExperimentConfig.name must be a non-empty string.")
        for attr in ("task", "data", "pipeline", "agents", "retrieval", "evaluation", "metadata"):
            value = getattr(self, attr)
            if not isinstance(value, dict):
                setattr(self, attr, dict(value))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        if not isinstance(data, dict):
            raise TypeError("ExperimentConfig.from_dict expects a dict.")
        return cls(
            name=str(data.get("name", "experiment")),
            task=dict(data.get("task", {})),
            data=dict(data.get("data", {})),
            pipeline=dict(data.get("pipeline", {})),
            agents=dict(data.get("agents", {})),
            retrieval=dict(data.get("retrieval", {})),
            evaluation=dict(data.get("evaluation", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": dict(self.task),
            "data": dict(self.data),
            "pipeline": dict(self.pipeline),
            "agents": dict(self.agents),
            "retrieval": dict(self.retrieval),
            "evaluation": dict(self.evaluation),
            "metadata": dict(self.metadata),
        }
