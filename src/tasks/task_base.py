# src/tasks/task_base.py

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Optional

from core.exceptions import ConfigError, ValidationError
from core.interfaces import BaseTaskInterface
from core.schemas import QueryInstance, TaskSpec, TimeSeriesSample
from ts_logging.event_log import EventLogger


class BaseTask(BaseTaskInterface, ABC):
    """
    Reusable base class for all task definitions.

    Responsibilities of a task
    --------------------------
    A concrete task implementation is responsible for:
    1. defining how a raw TimeSeriesSample is converted into a QueryInstance
    2. exposing the label space
    3. providing a task-specific prompt target / instruction
    4. parsing raw downstream outputs into task-aligned objects

    Non-responsibilities
    --------------------
    A task should NOT:
    - implement retrieval logic
    - implement agent reasoning logic
    - perform pipeline orchestration

    Design notes
    ------------
    - This base class keeps the contract thin and stable.
    - Subclasses should usually override:
        * build_query(...)
        * get_prompt_target(...)
        * parse_output(...)
    - `get_label_space()` already has a safe default from task_spec.
    - Loggers are not created here; they are injected through runtime context.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(task_spec=task_spec, name=name, config=config)

    # ------------------------------------------------------------------
    # Context / logging helpers
    # ------------------------------------------------------------------
    def normalize_context(self, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if context is None:
            return {}
        if not isinstance(context, dict):
            raise TypeError(f"{self.name}: context must be a dict or None.")
        return context

    def get_logger(self, context: Optional[dict[str, Any]] = None) -> Optional[logging.Logger]:
        if context is None:
            return None
        logger = context.get("logger")
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError(f"{self.name}: context['logger'] must be a logging.Logger.")
        return logger

    def get_event_logger(self, context: Optional[dict[str, Any]] = None) -> Optional[EventLogger]:
        if context is None:
            return None
        event_logger = context.get("event_logger")
        if event_logger is not None and not isinstance(event_logger, EventLogger):
            raise TypeError(
                f"{self.name}: context['event_logger'] must be an EventLogger."
            )
        return event_logger

    def log_info(self, context: Optional[dict[str, Any]], message: str, *args: Any) -> None:
        logger = self.get_logger(context)
        if logger is not None:
            logger.info(message, *args)

    def log_warning(self, context: Optional[dict[str, Any]], message: str, *args: Any) -> None:
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
        event_logger = self.get_event_logger(context)
        if event_logger is not None:
            event_logger.log_event(event_type=event_type, payload=payload, **kwargs)

    # ------------------------------------------------------------------
    # Validation / config helpers
    # ------------------------------------------------------------------
    def validate_sample(self, sample: TimeSeriesSample) -> None:
        """
        Validate that the incoming object is a legal task input.

        Subclasses may override this if they need stronger checks
        (for example, anomaly-window tasks may require window metadata).
        """
        if not isinstance(sample, TimeSeriesSample):
            raise ValidationError(
                f"{self.name}: sample must be a TimeSeriesSample, "
                f"but got {type(sample).__name__}."
            )

    def describe(self) -> str:
        """
        Human-readable description for debugging and experiment logs.
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"task_type={self.task_spec.task_type.value}, "
            f"granularity={self.task_spec.granularity}, "
            f"num_labels={len(self.task_spec.label_space)}, "
            f"config_keys={sorted(self.config.keys())}"
            f")"
        )

    def get_label_space(self) -> list[str]:
        """
        Default label space accessor.

        Most tasks can directly reuse the label_space from TaskSpec.
        Subclasses may override this if they need dynamic label spaces.
        """
        return list(self.task_spec.label_space)

    def has_label_space(self) -> bool:
        """
        Whether this task currently has a non-empty label space.
        """
        return len(self.task_spec.label_space) > 0

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
            raise ConfigError(f"{self.name}: required config key '{key}' is missing.")
        return self.config[key]

    # ------------------------------------------------------------------
    # Core task methods
    # ------------------------------------------------------------------
    def build_query(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> QueryInstance:
        """
        Default query builder.

        This creates a minimal QueryInstance directly from the raw sample.
        It is intentionally generic and is mainly useful as a fallback or
        for the simplest task settings.

        Concrete tasks will often override this method to:
        - build sliding-window queries
        - attach horizon metadata for prediction
        - inject anomaly-specific metadata
        - customize query_id generation
        """
        context = self.normalize_context(context)
        self.validate_sample(sample)

        query = QueryInstance(
            query_id=self._build_default_query_id(sample),
            sample=sample,
            task_spec=self.task_spec,
            channels=[],
            metadata=self._build_default_query_metadata(sample),
        )

        self.log_info(
            context,
            "Task '%s': built query_id=%s for sample_id=%s",
            self.name,
            query.query_id,
            sample.sample_id,
        )
        self.log_event(
            context,
            event_type="task_build_query",
            payload={
                "task_name": self.name,
                "task_type": self.task_spec.task_type.value,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
            },
        )

        return query

    def get_prompt_target(self) -> str:
        """
        Default task instruction string.

        Concrete tasks should usually override this with more explicit wording.
        """
        task_type = self.task_spec.task_type.value
        if task_type == "classification":
            return "predict the class label of the input time series"
        if task_type == "prediction":
            return "predict the future label of the input time series"
        if task_type == "anomaly_sequence":
            return "determine whether the sequence is anomalous"
        if task_type == "anomaly_window":
            return "determine whether the window is anomalous"
        return "make a prediction for the input time series"

    def parse_output(
        self,
        raw_output: Any,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Default raw output parser.

        The base implementation is intentionally conservative:
        it returns raw_output unchanged after validating the sample.

        Concrete tasks should typically override this method to parse:
        - LLM JSON outputs
        - label strings
        - anomaly decisions
        - task-specific prediction records
        """
        context = self.normalize_context(context)
        self.validate_sample(sample)

        self.log_event(
            context,
            event_type="task_parse_output",
            payload={
                "task_name": self.name,
                "task_type": self.task_spec.task_type.value,
                "sample_id": sample.sample_id,
            },
        )

        return raw_output

    def normalize_label(self, label: Any) -> Any:
        """
        Normalize a label candidate into a task-compatible form.

        Default behavior:
        - if the task has a label space, string labels are stripped
        - exact membership is checked for string labels
        - otherwise return label as-is

        Subclasses may override for more flexible parsing.
        """
        if not self.has_label_space():
            return label

        if isinstance(label, str):
            normalized = label.strip()
            if normalized not in self.task_spec.label_space:
                raise ValidationError(
                    f"{self.name}: label '{normalized}' is not in label_space "
                    f"{self.task_spec.label_space}."
                )
            return normalized

        return label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_default_query_id(self, sample: TimeSeriesSample) -> str:
        """
        Internal helper for stable default query id generation.
        """
        return f"{self.task_spec.task_type.value}:{sample.sample_id}"

    def _build_default_query_metadata(self, sample: TimeSeriesSample) -> dict[str, Any]:
        """
        Internal helper for default query metadata.
        """
        return {
            "source_sample_id": sample.sample_id,
            "task_type": self.task_spec.task_type.value,
            "granularity": self.task_spec.granularity,
            "data_mode": sample.data_mode.value,
            "sequence_length": sample.length,
            "num_channels": sample.num_channels,
        }