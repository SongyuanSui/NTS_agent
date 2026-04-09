# src/core/enums.py

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """
    Small helper enum base class whose members behave like strings.

    Benefits
    --------
    - JSON / YAML / logging 更自然
    - 可以直接和已有字符串配置对接
    """

    def __str__(self) -> str:
        return self.value


class TaskType(StrEnum):
    """
    Supported task types in the framework.
    """

    CLASSIFICATION = "classification"
    PREDICTION = "prediction"
    ANOMALY_SEQUENCE = "anomaly_sequence"
    ANOMALY_WINDOW = "anomaly_window"


class DataMode(StrEnum):
    """
    Time-series data modality by channel cardinality.
    """

    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"


class RepresentationType(StrEnum):
    """
    Supported representation / memory views.
    """

    TS = "ts"
    SUMMARY = "summary"
    STATISTIC = "statistic"


class RetrievalMode(StrEnum):
    """
    Retrieval branches used in the framework.
    """

    TS = "ts"
    TEXT = "text"
    STAT = "stat"
    HYBRID = "hybrid"


class PipelineStage(StrEnum):
    """
    Common execution stages used by pipelines / agents / logging.
    """

    CHANNEL_SELECTION = "channel_selection"
    CHANNEL_DECOMPOSITION = "channel_decomposition"
    REPRESENTATION = "representation"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    AGGREGATION = "aggregation"
    MEMORY_BUILD = "memory_build"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class Granularity(StrEnum):
    """
    Prediction / annotation granularity.
    """

    SAMPLE = "sample"
    # SEQUENCE = "sequence"
    WINDOW = "window"
    CHANNEL = "channel"


class SplitName(StrEnum):
    """
    Common dataset split names.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ScoreDirection(StrEnum):
    """
    Whether a retrieval score should be interpreted as larger-is-better
    or smaller-is-better.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"