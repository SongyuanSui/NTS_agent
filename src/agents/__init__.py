"""Agents module for pipeline orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseAgent": ("agents.agent_base", "BaseAgent"),
    "AgentInput": ("agents.schemas", "AgentInput"),
    "AgentOutput": ("agents.schemas", "AgentOutput"),
    "ChannelSelectorAgent": ("agents.channel_selector", "ChannelSelectorAgent"),
    "ChannelSelectorInput": ("agents.schemas", "ChannelSelectorInput"),
    "ChannelSelectorOutput": ("agents.schemas", "ChannelSelectorOutput"),
    "ChannelDecomposerAgent": ("agents.channel_decomposer", "ChannelDecomposerAgent"),
    "ChannelDecomposerInput": ("agents.schemas", "ChannelDecomposerInput"),
    "ChannelDecomposerOutput": ("agents.schemas", "ChannelDecomposerOutput"),
    "RepresentationAgentTS": ("agents.representation_agent_ts", "RepresentationAgentTS"),
    "RepresentationAgentSummary": ("agents.representation_agent_summary", "RepresentationAgentSummary"),
    "RepresentationAgentStatistic": (
        "agents.representation_agent_statistic",
        "RepresentationAgentStatistic",
    ),
    "RepresentationInput": ("agents.schemas", "RepresentationInput"),
    "RepresentationOutput": ("agents.schemas", "RepresentationOutput"),
    "RetrievalAgentTS": ("agents.retrieval_agent_ts", "RetrievalAgentTS"),
    "RetrievalAgentText": ("agents.retrieval_agent_text", "RetrievalAgentText"),
    "RetrievalAgentStat": ("agents.retrieval_agent_stat", "RetrievalAgentStat"),
    "RetrievalAgentHybrid": ("agents.retrieval_agent_hybrid", "RetrievalAgentHybrid"),
    "RetrievalInput": ("agents.schemas", "RetrievalInput"),
    "RetrievalOutput": ("agents.schemas", "RetrievalOutput"),
    "ReasonerAgent": ("agents.reasoner_agent", "ReasonerAgent"),
    "ReasonerInput": ("agents.schemas", "ReasonerInput"),
    "ReasonerOutput": ("agents.schemas", "ReasonerOutput"),
    "ReasonerChannelDecision": ("agents.schemas", "ReasonerChannelDecision"),
    "AggregationAgent": ("agents.aggregation_agent", "AggregationAgent"),
    "AggregationInput": ("agents.schemas", "AggregationInput"),
    "AggregationOutput": ("agents.schemas", "AggregationOutput"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'agents' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])


__all__ = sorted(_EXPORTS.keys())
