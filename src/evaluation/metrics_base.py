"""Base classes and utilities for evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Metadata = dict[str, Any]


@dataclass(slots=True)
class MetricResult:
    """Lightweight metric container for evaluation results."""

    name: str
    value: float
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

        self.value = float(self.value)
        if not np.isfinite(self.value):
            raise ValueError("value must be finite")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)
