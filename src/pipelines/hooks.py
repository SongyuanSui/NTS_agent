from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


class PipelineHook(Protocol):
    """Protocol for lightweight pipeline lifecycle hooks."""

    name: str

    def before_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        ...

    def after_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        ...

    def on_error(self, stage: str, error: Exception, context: dict[str, Any]) -> None:
        ...


@dataclass
class NoOpPipelineHook:
    """Default hook implementation used when no instrumentation is needed."""

    name: str = "noop"

    def before_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        return None

    def after_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        return None

    def on_error(self, stage: str, error: Exception, context: dict[str, Any]) -> None:
        return None


@dataclass
class CallablePipelineHook:
    """
    Adapter for registering simple callables as pipeline hooks.

    Each callable receives keyword arguments: stage, payload/error, and context.
    """

    name: str
    before: Optional[Callable[..., None]] = None
    after: Optional[Callable[..., None]] = None
    error: Optional[Callable[..., None]] = None

    def before_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        if self.before is not None:
            self.before(stage=stage, payload=payload, context=context)

    def after_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        if self.after is not None:
            self.after(stage=stage, payload=payload, context=context)

    def on_error(self, stage: str, error: Exception, context: dict[str, Any]) -> None:
        if self.error is not None:
            self.error(stage=stage, error=error, context=context)


@dataclass
class HookManager:
    """Small dispatcher for pipeline lifecycle hooks."""

    hooks: list[PipelineHook] = field(default_factory=list)

    def add(self, hook: PipelineHook) -> None:
        if not hasattr(hook, "before_stage") or not hasattr(hook, "after_stage"):
            raise TypeError("hook must implement the PipelineHook protocol.")
        self.hooks.append(hook)

    def before_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.before_stage(stage=stage, payload=payload, context=context)

    def after_stage(self, stage: str, payload: Any, context: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.after_stage(stage=stage, payload=payload, context=context)

    def on_error(self, stage: str, error: Exception, context: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.on_error(stage=stage, error=error, context=context)
