from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.enums import RepresentationType
from core.schemas import RepresentationRecord, TimeSeriesSample


Metadata = dict[str, Any]


@dataclass(slots=True)
class RepresentationInput:
    """Input schema for one representation computation request."""

    samples: list[TimeSeriesSample] = field(default_factory=list)
    channel_id: int = 0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.samples, list):
            self.samples = list(self.samples)

        for sample in self.samples:
            if not isinstance(sample, TimeSeriesSample):
                raise TypeError("samples must contain TimeSeriesSample objects only.")

        self.channel_id = int(self.channel_id)
        if self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RepresentationOutput:
    """Output schema for one representation computation request."""

    rep_type: RepresentationType | str
    records: list[RepresentationRecord] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.rep_type, str):
            self.rep_type = RepresentationType(self.rep_type)

        if not isinstance(self.rep_type, RepresentationType):
            raise TypeError("rep_type must be a RepresentationType or a valid string.")

        if not isinstance(self.records, list):
            self.records = list(self.records)
        for record in self.records:
            if not isinstance(record, RepresentationRecord):
                raise TypeError("records must contain RepresentationRecord objects only.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def num_records(self) -> int:
        return len(self.records)
