# src/agents/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import RepresentationType
from core.schemas import ChannelData, QueryInstance, TaskSpec
from retrieval.schemas import RetrievedSet


Metadata = dict[str, Any]


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


@dataclass(slots=True)
class ChannelDecomposerInput(AgentInput):
    query: QueryInstance | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.query is None:
            raise ValueError("ChannelDecomposerInput.query must not be None.")
        if not isinstance(self.query, QueryInstance):
            raise TypeError("ChannelDecomposerInput.query must be a QueryInstance.")


@dataclass(slots=True)
class ChannelDecomposerOutput(AgentOutput):
    query_id: str = ""
    all_channels: list[ChannelData] = field(default_factory=list)
    selected_channels: list[ChannelData] = field(default_factory=list)
    channel_scores: dict[int, float] = field(default_factory=dict)
    selection_applied: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        for channel in self.all_channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("all_channels must contain ChannelData objects only.")

        for channel in self.selected_channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("selected_channels must contain ChannelData objects only.")

        if not isinstance(self.channel_scores, dict):
            raise TypeError("channel_scores must be a dict[int, float].")


@dataclass(slots=True)
class RepresentationInput(AgentInput):
    query: QueryInstance | None = None
    channels: list[ChannelData] = field(default_factory=list)
    representation_type: RepresentationType | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("RepresentationInput.query must be a QueryInstance.")

        for channel in self.channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("channels must contain ChannelData objects only.")

        if self.representation_type is None or not isinstance(
            self.representation_type, RepresentationType
        ):
            raise TypeError("representation_type must be a RepresentationType.")


@dataclass(slots=True)
class RepresentationOutput(AgentOutput):
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


@dataclass(slots=True)
class RetrievalInput(AgentInput):
    query: QueryInstance | None = None
    channels: list[ChannelData] = field(default_factory=list)
    top_k: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("RetrievalInput.query must be a QueryInstance.")

        for channel in self.channels:
            if not isinstance(channel, ChannelData):
                raise TypeError("channels must contain ChannelData objects only.")

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")


@dataclass(slots=True)
class RetrievalOutput(AgentOutput):
    query_id: str = ""
    retrieved_sets: dict[int, RetrievedSet] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if not isinstance(self.retrieved_sets, dict):
            raise TypeError("retrieved_sets must be a dict.")

        for _, retrieved_set in self.retrieved_sets.items():
            if not isinstance(retrieved_set, RetrievedSet):
                raise TypeError("retrieved_sets values must be RetrievedSet objects.")


@dataclass(slots=True)
class ReasonerInput(AgentInput):
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

        for _, retrieved_set in self.retrieved_sets.items():
            if not isinstance(retrieved_set, RetrievedSet):
                raise TypeError("retrieved_sets values must be RetrievedSet objects.")


@dataclass(slots=True)
class ReasonerChannelDecision:
    channel_id: int
    prediction: Any
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, int) or self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be numeric if provided.")
            if self.confidence < 0.0 or self.confidence > 1.0:
                raise ValueError("confidence must be within [0.0, 1.0].")

        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise TypeError("reasoning must be a string if provided.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class ReasonerOutput(AgentOutput):
    query_id: str = ""
    channel_decisions: list[ReasonerChannelDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        for decision in self.channel_decisions:
            if not isinstance(decision, ReasonerChannelDecision):
                raise TypeError(
                    "channel_decisions must contain ReasonerChannelDecision objects only."
                )


@dataclass(slots=True)
class AggregationInput(AgentInput):
    query: QueryInstance | None = None
    channel_decisions: list[ReasonerChannelDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.query is None or not isinstance(self.query, QueryInstance):
            raise TypeError("AggregationInput.query must be a QueryInstance.")

        for decision in self.channel_decisions:
            if not isinstance(decision, ReasonerChannelDecision):
                raise TypeError(
                    "channel_decisions must contain ReasonerChannelDecision objects only."
                )


@dataclass(slots=True)
class AggregationOutput(AgentOutput):
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
            if self.confidence < 0.0 or self.confidence > 1.0:
                raise ValueError("confidence must be within [0.0, 1.0].")

        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise TypeError("reasoning must be a string if provided.")