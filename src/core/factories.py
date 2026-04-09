# src/core/factories.py

from __future__ import annotations

from typing import Any

from core.exceptions import ConfigError, FactoryError
from core.registry import (
    AGENT_REGISTRY,
    PIPELINE_REGISTRY,
    RETRIEVER_REGISTRY,
    TASK_REGISTRY,
)
from core.schemas import TaskSpec


def _ensure_dict(cfg: dict[str, Any] | None, what: str) -> dict[str, Any]:
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{what} config must be a dict or None.")
    return cfg


def _extract_name_and_params(cfg: dict[str, Any], what: str) -> tuple[str, dict[str, Any]]:
    cfg = _ensure_dict(cfg, what)

    if "name" not in cfg:
        raise ConfigError(f"{what} config must contain key 'name'.")

    name = cfg["name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{what} config field 'name' must be a non-empty string.")

    params = cfg.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError(f"{what} config field 'params' must be a dict.")

    return name, params


def build_task(task_cfg: dict[str, Any]) -> Any:
    """
    Build a task instance from config.

    Expected config shape
    ---------------------
    {
        "name": "classification",
        "task_spec": {
            "task_type": "classification",
            "label_space": [...],
            "granularity": "sample",
            "description": "...",
            "metadata": {...}
        },
        "params": {...}
    }
    """
    task_cfg = _ensure_dict(task_cfg, "task")
    name = task_cfg.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("task config must contain a non-empty 'name'.")

    if "task_spec" not in task_cfg:
        raise ConfigError("task config must contain 'task_spec'.")

    task_spec_cfg = task_cfg["task_spec"]
    if not isinstance(task_spec_cfg, dict):
        raise ConfigError("task_spec must be a dict.")

    params = task_cfg.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("task config field 'params' must be a dict.")

    task_cls = TASK_REGISTRY.get(name)
    task_spec = TaskSpec(**task_spec_cfg)

    try:
        return task_cls(task_spec=task_spec, name=name, config=params)
    except Exception as e:
        raise FactoryError(f"Failed to build task '{name}': {e}") from e


def build_agent(agent_cfg: dict[str, Any]) -> Any:
    name, params = _extract_name_and_params(agent_cfg, "agent")
    agent_cls = AGENT_REGISTRY.get(name)

    try:
        return agent_cls(name=name, config=params)
    except Exception as e:
        raise FactoryError(f"Failed to build agent '{name}': {e}") from e


def build_retriever(retriever_cfg: dict[str, Any]) -> Any:
    name, params = _extract_name_and_params(retriever_cfg, "retriever")
    retriever_cls = RETRIEVER_REGISTRY.get(name)

    try:
        return retriever_cls(name=name, config=params)
    except Exception as e:
        raise FactoryError(f"Failed to build retriever '{name}': {e}") from e


def build_pipeline(
    pipeline_cfg: dict[str, Any],
    components: dict[str, Any] | None = None,
) -> Any:
    """
    Build a pipeline instance from config.

    Expected config shape
    ---------------------
    {
        "name": "inference_pipeline",
        "params": {...}
    }
    """
    name, params = _extract_name_and_params(pipeline_cfg, "pipeline")
    pipeline_cls = PIPELINE_REGISTRY.get(name)

    try:
        return pipeline_cls(name=name, config=params, components=components or {})
    except Exception as e:
        raise FactoryError(f"Failed to build pipeline '{name}': {e}") from e