from __future__ import annotations

from typing import Any

from core.registry import TASK_REGISTRY
from core.schemas import TaskSpec

# Import concrete tasks for registration side effects.
from tasks import anomaly_window as _anomaly_window  # noqa: F401
from tasks import classification as _classification  # noqa: F401
from tasks import prediction as _prediction  # noqa: F401


def get_task_class(name: str) -> type:
    return TASK_REGISTRY.get(name)


def list_tasks() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def build_task_from_name(
    name: str,
    task_spec: TaskSpec,
    config: dict[str, Any] | None = None,
) -> Any:
    task_cls = get_task_class(name)
    return task_cls(task_spec=task_spec, name=name, config=config or {})
