"""Pipeline orchestration package.

Exports are loaded lazily so importing ``pipelines`` does not force optional
agent or representation dependencies to load before they are needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AgentGraph": ("pipelines.agent_graph", "AgentGraph"),
    "AgentGraphNode": ("pipelines.agent_graph", "AgentGraphNode"),
    "BasePipeline": ("pipelines.pipeline_base", "BasePipeline"),
    "CallablePipelineHook": ("pipelines.hooks", "CallablePipelineHook"),
    "ExecutionContext": ("pipelines.execution_context", "ExecutionContext"),
    "HookManager": ("pipelines.hooks", "HookManager"),
    "InferencePipeline": ("pipelines.inference_pipeline", "InferencePipeline"),
    "MemoryBuildPipeline": ("pipelines.memory_build_pipeline", "MemoryBuildPipeline"),
    "MemoryBuildResult": ("pipelines.memory_build_pipeline", "MemoryBuildResult"),
    "MemoryPersistenceConfig": (
        "pipelines.stat_feature_retrieval_pipeline",
        "MemoryPersistenceConfig",
    ),
    "NoOpPipelineHook": ("pipelines.hooks", "NoOpPipelineHook"),
    "PipelineHook": ("pipelines.hooks", "PipelineHook"),
    "StatFeatureRetrievalPipeline": (
        "pipelines.stat_feature_retrieval_pipeline",
        "StatFeatureRetrievalPipeline",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'pipelines' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])


__all__ = sorted(_EXPORTS.keys())
