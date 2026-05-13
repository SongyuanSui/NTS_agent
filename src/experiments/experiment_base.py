from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from experiments.experiment_config import ExperimentConfig


@dataclass(slots=True)
class ExperimentResult:
    """Standard result object for experiment runners."""

    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExperiment(ABC):
    """Base class for simple experiment orchestration."""

    def __init__(self, config: ExperimentConfig | dict[str, Any], name: Optional[str] = None) -> None:
        if isinstance(config, dict):
            config = ExperimentConfig.from_dict(config)
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be ExperimentConfig or dict.")
        self.config = config
        self.name = name or config.name

    @abstractmethod
    def run(self, context: Optional[dict[str, Any]] = None) -> ExperimentResult:
        raise NotImplementedError
