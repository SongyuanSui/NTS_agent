# src/agents/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import RepresentationType
from core.schemas import ChannelData, QueryInstance, TaskSpec, TimeSeriesSample
from retrieval.schemas import RetrievedSet


Metadata = dict[str, Any]


# ----------------------------------------------------------------------
# Generic base schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class AgentInput:
    """
    Lightweight generic base input for agent schemas.
    """

    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class AgentOutput:
    """
    Lightweight generic base output for agent schemas.
    """

    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


# ----------------------------------------------------------------------
# Channel selector schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ChannelSelectorInput(AgentInput):
    """
    Input for dataset-level channel selection.

    Notes
    -----
    - train_samples must be labeled.
    - nn_eval_samples <= 0 means using all samples as probe samples.
    """

    train_samples: list[TimeSeriesSample] = field(default_factory=list)
    task_spec: TaskSpec | None = None
    top_k: int = 1
    max_len: int | None = None
    z_norm: bool = True
    alpha: float = 0.5
    nn_eval_samples: int = 200
    diversity_threshold: float | None = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.train_samples, list) or len(self.train_samples) == 0:
            raise ValueError("train_samples must be a non-empty list of TimeSeriesSample.")

        for sample in self.train_samples:
            if not isinstance(sample, TimeSeriesSample):
                raise TypeError(
                    "train_samples must contain TimeSeriesSample objects only."
                )

        if self.task_spec is None or not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be a TaskSpec.")

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.max_len is not None:
            if not isinstance(self.max_len, int) or self.max_len <= 0:
                raise ValueError("max_len must be a positive integer or None.")

        if not isinstance(self.z_norm, bool):
            raise TypeError("z_norm must be a bool.")

        if not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be numeric.")
        self.alpha = float(self.alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be within [0.0, 1.0].")

        if not isinstance(self.nn_eval_samples, int):
            raise TypeError("nn_eval_samples must be an integer.")

        if self.diversity_threshold is not None:
            if not isinstance(self.diversity_threshold, (int, float)):
                raise TypeError("diversity_threshold must be numeric or None.")
            self.diversity_threshold = float(self.diversity_threshold)

        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer.")


@dataclass(slots=True)
class ChannelSelectorOutput(AgentOutput):
    """
    Output of dataset-level channel selection.

    selected_channel_ids:
        Top-k selected channels after optional diversity pruning.

    ranked_channel_ids:
        All channels ranked by fused score from high to low.

    channel_scores:
        Final fused score per channel.

    score_details:
        Per-channel detailed diagnostics such as B/C scores and debug stats.
    """

    selected_channel_ids: list[int] = field(default_factory=list)
    ranked_channel_ids: list[int] = field(default_factory=list)
    channel_scores: dict[int, float] = field(default_factory=dict)
    score_details: dict[int, dict[str, Any]] = field(default_factory=dict)
    selection_applied: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.selected_channel_ids, list):
            self.selected_channel_ids = list(self.selected_channel_ids)
        self.selected_channel_ids = [int(x) for x in self.selected_channel_ids]
        for channel_id in self.selected_channel_ids:
            if channel_id < 0:
                raise ValueError(
                    "selected_channel_ids must contain non-negative integers only."
                )

        if not isinstance(self.ranked_channel_ids, list):
            self.ranked_channel_ids = list(self.ranked_channel_ids)
        self.ranked_channel_ids = [int(x) for x in self.ranked_channel_ids]
        for channel_id in self.ranked_channel_ids:
            if channel_id < 0:
                raise ValueError(
                    "ranked_channel_ids must contain non-negative integers only."
                )

        if not isinstance(self.channel_scores, dict):
            raise TypeError("channel_scores must be a dict[int, float].")
        normalized_channel_scores: dict[int, float] = {}
        for key, value in self.channel_scores.items():
            key = int(key)
            if key < 0:
                raise ValueError("channel_scores keys must be non-negative integers.")
            if not isinstance(value, (int, float)):
                raise TypeError("channel_scores values must be numeric.")
            normalized_channel_scores[key] = float(value)
        self.channel_scores = normalized_channel_scores

        if not isinstance(self.score_details, dict):
            raise TypeError("score_details must be a dict[int, dict].")
        normalized_score_details: dict[int, dict[str, Any]] = {}
        for key, value in self.score_details.items():
            key = int(key)
            if key < 0:
                raise ValueError("score_details keys must be non-negative integers.")
            if not isinstance(value, dict):
                raise TypeError("score_details values must be dict objects.")
            normalized_score_details[key] = value
        self.score_details = normalized_score_details

        if not isinstance(self.selection_applied, bool):
            raise TypeError("selection_applied must be a bool.")


# ----------------------------------------------------------------------
# Channel decomposer schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ChannelDecomposerInput(AgentInput):
    """
    Input for per-sample channel decomposition.

    selected_channel_ids:
        Optional externally selected channels, usually from ChannelSelectorOutput.
        If None, the decomposer may:
        - keep all channels
        - or apply its own fallback policy
    """

    query: QueryInstance | None = None
    selected_channel_ids: list[int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None:
            raise ValueError("ChannelDecomposerInput.query must not be None.")
        if not isinstance(self.query, QueryInstance):
            raise TypeError("ChannelDecomposerInput.query must be a QueryInstance.")

        if self.selected_channel_ids is not None:
            if not isinstance(self.selected_channel_ids, list):
                self.selected_channel_ids = list(self.selected_channel_ids)
            self.selected_channel_ids = [int(x) for x in self.selected_channel_ids]
            for channel_id in self.selected_channel_ids:
                if channel_id < 0:
                    raise ValueError(
                        "selected_channel_ids must contain non-negative integers only."
                    )


@dataclass(slots=True)
class ChannelDecomposerOutput(AgentOutput):
    """
    Output of per-sample channel decomposition.

    all_channels:
        All decomposed channels from the sample/query.

    selected_channels:
        Channels actually forwarded to downstream pipeline.

    selected_channel_ids:
        Channel ids corresponding to selected_channels.
    """

    query_id: str = ""
    all_channels: list[ChannelData] = field(default_factory=list)
    selected_channels: list[ChannelData] = field(default_factory=list)
    selected_channel_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if not isinstance(self.all_channels, list):
            self.all_channels = list(self.all_channels)
        for channel in self.all_channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("all_channels must contain ChannelData objects only.")

        if not isinstance(self.selected_channels, list):
            self.selected_channels = list(self.selected_channels)
        for channel in self.selected_channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("selected_channels must contain ChannelData objects only.")

        if not isinstance(self.selected_channel_ids, list):
            self.selected_channel_ids = list(self.selected_channel_ids)
        self.selected_channel_ids = [int(x) for x in self.selected_channel_ids]
        for channel_id in self.selected_channel_ids:
            if channel_id < 0:
                raise ValueError(
                    "selected_channel_ids must contain non-negative integers only."
                )

        selected_ids_from_channels = [channel.channel_id for channel in self.selected_channels]
        if self.selected_channel_ids and self.selected_channel_ids != selected_ids_from_channels:
            raise ValueError(
                "selected_channel_ids does not match channel ids in selected_channels."
            )

        if not self.selected_channel_ids and self.selected_channels:
            self.selected_channel_ids = selected_ids_from_channels


# ----------------------------------------------------------------------
# Representation schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class RepresentationInput(AgentInput):
    """
    Input for representation agents.
    """

    query: QueryInstance | None = None
    channels: list[ChannelData] = field(default_factory=list)
    representation_type: RepresentationType | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("RepresentationInput.query must be a QueryInstance.")

        if not isinstance(self.channels, list):
            self.channels = list(self.channels)
        for channel in self.channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("channels must contain ChannelData objects only.")

        if self.representation_type is None or not isinstance(
            self.representation_type, RepresentationType
        ):
            raise TypeError("representation_type must be a RepresentationType.")


@dataclass(slots=True)
class RepresentationOutput(AgentOutput):
    """
    Output of representation agents.

    channel_payloads:
        Dict[channel_id, payload]
    """

    query_id: str = ""
    representation_type: RepresentationType | None = None
    channel_payloads: dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if self.representation_type is None or not isinstance(
            self.representation_type, RepresentationType
        ):
            raise TypeError("representation_type must be a RepresentationType.")

        if not isinstance(self.channel_payloads, dict):
            raise TypeError("channel_payloads must be a dict.")

        normalized_payloads: dict[int, Any] = {}
        for key, value in self.channel_payloads.items():
            key = int(key)
            if key < 0:
                raise ValueError("channel_payloads keys must be non-negative integers.")
            normalized_payloads[key] = value
        self.channel_payloads = normalized_payloads


# ----------------------------------------------------------------------
# Retrieval schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class RetrievalInput(AgentInput):
    """
    Input for retrieval agents.

    channels:
        The selected channels to retrieve for.
    """

    query: QueryInstance | None = None
    channels: list[ChannelData] = field(default_factory=list)
    top_k: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("RetrievalInput.query must be a QueryInstance.")

        if not isinstance(self.channels, list):
            self.channels = list(self.channels)
        for channel in self.channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("channels must contain ChannelData objects only.")

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")


@dataclass(slots=True)
class RetrievalOutput(AgentOutput):
    """
    Output of retrieval agents.

    retrieved_sets:
        Dict[channel_id, RetrievedSet]
    """

    query_id: str = ""
    retrieved_sets: dict[int, RetrievedSet] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if not isinstance(self.retrieved_sets, dict):
            raise TypeError("retrieved_sets must be a dict.")

        normalized_sets: dict[int, RetrievedSet] = {}
        for key, retrieved_set in self.retrieved_sets.items():
            key = int(key)
            if key < 0:
                raise ValueError("retrieved_sets keys must be non-negative integers.")
            if not isinstance(retrieved_set, RetrievedSet):
                raise TypeError("retrieved_sets values must be RetrievedSet objects.")
            normalized_sets[key] = retrieved_set
        self.retrieved_sets = normalized_sets


# ----------------------------------------------------------------------
# Reasoning schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ReasonerInput(AgentInput):
    """
    Input for the reasoner agent.
    """

    query: QueryInstance | None = None
    task_spec: TaskSpec | None = None
    retrieved_sets: dict[int, RetrievedSet] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("ReasonerInput.query must be a QueryInstance.")

        if self.task_spec is None or not isinstance(self.task_spec, TaskSpec):
            raise TypeError("ReasonerInput.task_spec must be a TaskSpec.")

        if not isinstance(self.retrieved_sets, dict):
            raise TypeError("retrieved_sets must be a dict.")

        normalized_sets: dict[int, RetrievedSet] = {}
        for key, retrieved_set in self.retrieved_sets.items():
            key = int(key)
            if key < 0:
                raise ValueError("retrieved_sets keys must be non-negative integers.")
            if not isinstance(retrieved_set, RetrievedSet):
                raise TypeError("retrieved_sets values must be RetrievedSet objects.")
            normalized_sets[key] = retrieved_set
        self.retrieved_sets = normalized_sets


@dataclass(slots=True)
class ReasonerChannelDecision:
    """
    One channel-level decision produced by the reasoner.
    """

    channel_id: int
    prediction: Any
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel_id = int(self.channel_id)
        if self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be numeric if provided.")
            self.confidence = float(self.confidence)
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError("confidence must be within [0.0, 1.0].")

        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise TypeError("reasoning must be a string if provided.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class ReasonerOutput(AgentOutput):
    """
    Output of the reasoner agent.
    """

    query_id: str = ""
    channel_decisions: list[ReasonerChannelDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if not isinstance(self.channel_decisions, list):
            self.channel_decisions = list(self.channel_decisions)

        for decision in self.channel_decisions:
            if not isinstance(decision, ReasonerChannelDecision):
                raise TypeError(
                    "channel_decisions must contain ReasonerChannelDecision objects only."
                )


# ----------------------------------------------------------------------
# Aggregation schemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class AggregationInput(AgentInput):
    """
    Input for the aggregator agent.
    """

    query: QueryInstance | None = None
    channel_decisions: list[ReasonerChannelDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("AggregationInput.query must be a QueryInstance.")

        if not isinstance(self.channel_decisions, list):
            self.channel_decisions = list(self.channel_decisions)

        for decision in self.channel_decisions:
            if not isinstance(decision, ReasonerChannelDecision):
                raise TypeError(
                    "channel_decisions must contain ReasonerChannelDecision objects only."
                )


@dataclass(slots=True)
class AggregationOutput(AgentOutput):
    """
    Output of the aggregator agent.

    prediction:
        Final channel-fused decision before task-level parsing.
    """

    query_id: str = ""
    prediction: Any = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be numeric if provided.")
            self.confidence = float(self.confidence)
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError("confidence must be within [0.0, 1.0].")

        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise TypeError("reasoning must be a string if provided.")