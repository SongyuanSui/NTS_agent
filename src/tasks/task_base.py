# src/tasks/task_base.py

from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from core.interfaces import BaseTaskInterface
from core.schemas import QueryInstance, TaskSpec, TimeSeriesSample


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
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(task_spec=task_spec, name=name, config=config)

    def validate_sample(self, sample: TimeSeriesSample) -> None:
        """
        Validate that the incoming object is a legal task input.

        Subclasses may override this if they need stronger checks
        (for example, anomaly-window tasks may require window metadata).
        """
        if not isinstance(sample, TimeSeriesSample):
            raise TypeError(
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
            raise KeyError(f"{self.name}: required config key '{key}' is missing.")
        return self.config[key]

    def build_query(self, sample: TimeSeriesSample) -> QueryInstance:
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
        self.validate_sample(sample)

        return QueryInstance(
            query_id=self._build_default_query_id(sample),
            sample=sample,
            task_spec=self.task_spec,
            channels=[],
            metadata=self._build_default_query_metadata(sample),
        )

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

    def parse_output(self, raw_output: Any, sample: TimeSeriesSample) -> Any:
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
        self.validate_sample(sample)
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
                raise ValueError(
                    f"{self.name}: label '{normalized}' is not in label_space "
                    f"{self.task_spec.label_space}."
                )
            return normalized

        return label

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