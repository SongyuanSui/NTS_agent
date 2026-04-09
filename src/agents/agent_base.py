# src/agents/agent_base.py

from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from core.interfaces import BaseAgentInterface


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
       - standard metadata description
    3. Avoid imposing agent-specific logic here.

    Notes
    -----
    - Subclasses should return structured schema objects rather than raw dicts.
    - Subclasses may override `pre_run` / `post_run` hooks if needed.
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

        Execution flow:
        1. normalize context
        2. pre_run(...)
        3. _run_impl(...)
        4. post_run(...)

        Subclasses should implement `_run_impl(...)`, not usually override `run(...)`,
        unless they have a strong reason.
        """
        context = self.normalize_context(context)
        self.pre_run(input_data=input_data, context=context)
        output_data = self._run_impl(input_data=input_data, context=context)
        return self.post_run(output_data=output_data, context=context)

    def _run_impl(self, input_data: Any, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Actual agent logic to be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run_impl(...)."
        )