# src/core/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .enums import DataMode, RepresentationType, TaskType


Metadata = dict[str, Any]
LabelSpace = list[str]


@dataclass(slots=True)
class TimeSeriesSample:
    """
    Canonical sample object used across the whole framework.

    Notes
    -----
    - `x` must be either:
        * shape (T,) for univariate time series
        * shape (T, C) for multivariate time series
    - `y` is optional because unlabeled test/query samples are valid.
    - `metadata` can store dataset-specific fields such as:
        dataset_name, split, timestamps, anomaly spans, horizon info, etc.
    """

    sample_id: str
    x: np.ndarray
    y: Optional[Any] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        self.x = np.asarray(self.x, dtype=float)

        if self.x.ndim not in (1, 2):
            raise ValueError(
                f"x must have shape (T,) or (T, C), but got ndim={self.x.ndim}."
            )
        if self.x.shape[0] <= 0:
            raise ValueError("x must contain at least one time step.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def data_mode(self) -> DataMode:
        return DataMode.UNIVARIATE if self.x.ndim == 1 else DataMode.MULTIVARIATE

    @property
    def length(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_channels(self) -> int:
        return 1 if self.x.ndim == 1 else int(self.x.shape[1])

    @property
    def is_univariate(self) -> bool:
        return self.data_mode == DataMode.UNIVARIATE

    @property
    def is_multivariate(self) -> bool:
        return self.data_mode == DataMode.MULTIVARIATE


@dataclass(slots=True)
class TaskSpec:
    """
    Task-level specification shared by task modules, agents, and pipelines.

    Recommended values
    ------------------
    task_type:
        - TaskType.CLASSIFICATION
        - TaskType.PREDICTION
        - TaskType.ANOMALY_SEQUENCE
        - TaskType.ANOMALY_WINDOW

    granularity examples:
        - "sample"
        - "sequence"
        - "window"
        - "channel"
    """

    task_type: TaskType | str
    label_space: LabelSpace = field(default_factory=list)
    granularity: str = "sample"
    description: str = ""
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)

        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type must be an instance of TaskType or a valid string.")

        if not isinstance(self.label_space, list):
            self.label_space = list(self.label_space)

        self.label_space = [str(label) for label in self.label_space]
        if len(set(self.label_space)) != len(self.label_space):
            raise ValueError("label_space contains duplicated labels.")

        if not isinstance(self.granularity, str) or not self.granularity:
            raise ValueError("granularity must be a non-empty string.")

        if not isinstance(self.description, str):
            raise TypeError("description must be a string.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def is_classification(self) -> bool:
        return self.task_type == TaskType.CLASSIFICATION

    @property
    def is_prediction(self) -> bool:
        return self.task_type == TaskType.PREDICTION

    @property
    def is_anomaly_sequence(self) -> bool:
        return self.task_type == TaskType.ANOMALY_SEQUENCE

    @property
    def is_anomaly_window(self) -> bool:
        return self.task_type == TaskType.ANOMALY_WINDOW


@dataclass(slots=True)
class ChannelData:
    """
    One channel extracted from a sample.

    This object is used by the channel_decomposer agent. For univariate samples,
    a single ChannelData with channel_id=0 is sufficient.
    """

    sample_id: str
    channel_id: int
    values: np.ndarray
    score: Optional[float] = None
    selected: bool = True
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        self.channel_id = int(self.channel_id)
        if self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 1:
            raise ValueError(
                f"ChannelData.values must be 1D, but got ndim={self.values.ndim}."
            )
        if self.values.shape[0] <= 0:
            raise ValueError("ChannelData.values must contain at least one element.")

        if self.score is not None:
            self.score = float(self.score)
            if not np.isfinite(self.score):
                raise ValueError("score must be finite if provided.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def length(self) -> int:
        return int(self.values.shape[0])


@dataclass(slots=True)
class QueryInstance:
    """
    Standardized query object consumed by retrieval and reasoning modules.

    Typical usage
    -------------
    - classification / prediction:
        query is built from one sample
    - anomaly window:
        query may represent one sliding window from a larger sequence
    """

    query_id: str
    sample: TimeSeriesSample
    task_spec: TaskSpec
    channels: list[ChannelData] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if not isinstance(self.sample, TimeSeriesSample):
            raise TypeError("sample must be a TimeSeriesSample.")

        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be a TaskSpec.")

        if not isinstance(self.channels, list):
            self.channels = list(self.channels)

        for channel in self.channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("channels must contain ChannelData objects only.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def has_channel_decomposition(self) -> bool:
        return len(self.channels) > 0


@dataclass(slots=True)
class RepresentationRecord:
    """
    Lightweight representation payload attached to a sample or channel.

    payload examples
    ----------------
    - TS view: np.ndarray
    - Summary view: str
    - Statistic view: dict[str, float] or np.ndarray
    """

    rep_type: RepresentationType | str
    payload: Any
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.rep_type, str):
            self.rep_type = RepresentationType(self.rep_type)

        if not isinstance(self.rep_type, RepresentationType):
            raise TypeError(
                "rep_type must be an instance of RepresentationType or a valid string."
            )

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class PredictionRecord:
    """
    Final standardized prediction object produced by the pipeline.

    prediction:
        can be a label, anomaly decision, or prediction label.
    confidence:
        optional because some baselines may not provide it.
    """

    sample_id: str
    task_type: TaskType | str
    prediction: Any
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)

        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type must be an instance of TaskType or a valid string.")

        if self.confidence is not None:
            self.confidence = float(self.confidence)
            if not np.isfinite(self.confidence):
                raise ValueError("confidence must be finite if provided.")
            if self.confidence < 0.0 or self.confidence > 1.0:
                raise ValueError("confidence must be within [0.0, 1.0].")

        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise TypeError("reasoning must be a string if provided.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class BatchPredictionRecord:
    """
    Helper container for batched pipeline outputs.
    """

    records: list[PredictionRecord]
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.records, list):
            self.records = list(self.records)

        for record in self.records:
            if not isinstance(record, PredictionRecord):
                raise TypeError("records must contain PredictionRecord objects only.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


@dataclass(slots=True)
class PipelineResult:
    """
    Standard pipeline return object.

    This keeps the top-level pipeline return stable even if later you want to
    attach intermediate outputs, timing info, or artifact paths.
    """

    prediction: PredictionRecord
    intermediates: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, PredictionRecord):
            raise TypeError("prediction must be a PredictionRecord.")

        if not isinstance(self.intermediates, dict):
            self.intermediates = dict(self.intermediates)

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)