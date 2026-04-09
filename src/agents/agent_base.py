# src/agents/agent_base.py

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Optional

from core.interfaces import BaseAgentInterface
from ts_logging.event_log import EventLogger


class BaseAgent(BaseAgentInterface, ABC):
    """
    Concrete reusable base class for all agents in the framework.

    Design goals
    ------------
    1. Keep the contract simple:
       - validate_input(...)
       - run(...)
    2. Provide common utilities that every agent can reuse:
       - name / config access
       - enabled flag
       - lightweight context normalization
       - optional logger / event logger access
       - standard metadata description
    3. Avoid imposing agent-specific logic here.

    Notes
    -----
    - Subclasses should return structured schema objects rather than raw dicts.
    - Subclasses may override `pre_run` / `post_run` hooks if needed.
    - This base class does NOT create loggers or decide any paths.
      Loggers are injected through runtime context.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, config=config)
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def validate_input(self, input_data: Any) -> None:
        """
        Default validation hook.

        Subclasses are expected to override this with stricter type/schema checks.
        The default behavior only prevents None input.
        """
        if input_data is None:
            raise ValueError(f"{self.name}: input_data must not be None.")

    def normalize_context(self, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Normalize runtime context into a mutable dict.

        This is useful because many callers may pass None.
        """
        if context is None:
            return {}
        if not isinstance(context, dict):
            raise TypeError(f"{self.name}: context must be a dict or None.")
        return context

    # ------------------------------------------------------------------
    # Optional logging helpers
    # ------------------------------------------------------------------
    def get_logger(self, context: Optional[dict[str, Any]] = None) -> Optional[logging.Logger]:
        """
        Return a human-readable logger from runtime context if available.
        """
        if context is None:
            return None
        logger = context.get("logger")
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError(f"{self.name}: context['logger'] must be a logging.Logger.")
        return logger

    def get_event_logger(self, context: Optional[dict[str, Any]] = None) -> Optional[EventLogger]:
        """
        Return a structured event logger from runtime context if available.
        """
        if context is None:
            return None
        event_logger = context.get("event_logger")
        if event_logger is not None and not isinstance(event_logger, EventLogger):
            raise TypeError(
                f"{self.name}: context['event_logger'] must be an EventLogger."
            )
        return event_logger

    def log_info(self, context: Optional[dict[str, Any]], message: str, *args: Any) -> None:
        """
        Convenience helper for optional human-readable logging.
        """
        logger = self.get_logger(context)
        if logger is not None:
            logger.info(message, *args)

    def log_warning(self, context: Optional[dict[str, Any]], message: str, *args: Any) -> None:
        """
        Convenience helper for optional warning logging.
        """
        logger = self.get_logger(context)
        if logger is not None:
            logger.warning(message, *args)

    def log_event(
        self,
        context: Optional[dict[str, Any]],
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Convenience helper for optional structured event logging.
        """
        event_logger = self.get_event_logger(context)
        if event_logger is not None:
            event_logger.log_event(event_type=event_type, payload=payload, **kwargs)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def pre_run(self, input_data: Any, context: Optional[dict[str, Any]] = None) -> None:
        """
        Hook executed before the main agent logic.

        Default behavior:
        - ensure the agent is enabled
        - validate input
        """
        if not self.enabled:
            raise RuntimeError(f"{self.name}: agent is disabled.")
        self.validate_input(input_data)

    def post_run(self, output_data: Any, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Hook executed after the main agent logic.

        Subclasses may override this for:
        - output validation
        - output normalization
        - logging / instrumentation

        By default, it returns the output unchanged.
        """
        return output_data

    def describe(self) -> str:
        """
        Human-readable description for debugging and experiment logs.
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"enabled={self.enabled}, "
            f"config_keys={sorted(self.config.keys())}"
            f")"
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Safe config getter.
        """
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        """
        Retrieve a required config value, raising an error if missing.
        """
        if key not in self.config:
            raise KeyError(f"{self.name}: required config key '{key}' is missing.")
        return self.config[key]

    def run(self, input_data: Any, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Standard execution entrypoint shared by all agents.

        Execution flow
        --------------
        1. normalize context
        2. log agent_start event
        3. pre_run(...)
        4. _run_impl(...)
        5. post_run(...)
        6. log agent_end event

        On exception
        ------------
        - emit warning log
        - emit agent_error event
        - re-raise the original exception

        Subclasses should implement `_run_impl(...)`, not usually override `run(...)`,
        unless they have a strong reason.
        """
        context = self.normalize_context(context)

        base_payload = {
            "agent_name": self.name,
            "agent_class": self.__class__.__name__,
            "stage": context.get("stage"),
            "sample_id": context.get("sample_id"),
            "query_id": context.get("query_id"),
            "task_type": context.get("task_type"),
        }

        self.log_info(context, "Starting agent '%s'", self.name)
        self.log_event(
            context,
            event_type="agent_start",
            payload=base_payload,
        )

        try:
            self.pre_run(input_data=input_data, context=context)
            output_data = self._run_impl(input_data=input_data, context=context)
            output_data = self.post_run(output_data=output_data, context=context)

        except Exception as exc:
            self.log_warning(
                context,
                "Agent '%s' failed with %s: %s",
                self.name,
                type(exc).__name__,
                str(exc),
            )
            self.log_event(
                context,
                event_type="agent_error",
                payload={
                    **base_payload,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

        self.log_info(context, "Finished agent '%s'", self.name)
        self.log_event(
            context,
            event_type="agent_end",
            payload=base_payload,
        )

        return output_data

    def _run_impl(self, input_data: Any, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Actual agent logic to be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run_impl(...)."
        )