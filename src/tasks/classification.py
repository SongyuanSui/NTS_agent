# src/tasks/classification.py

from __future__ import annotations

from typing import Any, Optional

from core.schemas import QueryInstance, TimeSeriesSample
from tasks.task_base import BaseTask


class ClassificationTask(BaseTask):
    """
    Concrete task implementation for time series classification.

    Responsibilities
    ----------------
    - build a classification query from a raw sample
    - expose classification-specific prompt instruction
    - parse downstream outputs into a normalized class prediction

    Expected label behavior
    -----------------------
    - label_space is typically a list[str]
    - normalized labels should belong to label_space when label_space is non-empty
    """

    def __init__(
        self,
        task_spec,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(task_spec=task_spec, name=name, config=config)

        if not self.task_spec.is_classification:
            raise ValueError(
                f"{self.name}: ClassificationTask requires task_type='classification', "
                f"but got '{self.task_spec.task_type.value}'."
            )

    def build_query(self, sample: TimeSeriesSample) -> QueryInstance:
        """
        Build a classification query from a raw sample.

        For classification, the default behavior is already close to what we want:
        one sample -> one query. We just enrich metadata a bit.
        """
        self.validate_sample(sample)

        metadata = self._build_default_query_metadata(sample)
        metadata.update(
            {
                "task_family": "classification",
                "has_ground_truth": sample.y is not None,
            }
        )

        return QueryInstance(
            query_id=self._build_default_query_id(sample),
            sample=sample,
            task_spec=self.task_spec,
            channels=[],
            metadata=metadata,
        )

    def get_prompt_target(self) -> str:
        """
        Classification-specific instruction string for prompt builders.
        """
        if self.has_label_space():
            label_text = ", ".join(self.get_label_space())
            return (
                "predict the class label of the input time series from the given "
                f"label space: [{label_text}]"
            )

        return "predict the class label of the input time series"

    def parse_output(self, raw_output: Any, sample: TimeSeriesSample) -> Any:
        """
        Parse raw downstream output into a normalized classification label.

        Supported lightweight behaviors in this first version:
        - raw_output is already a label string / label object
        - raw_output is a dict with key 'prediction'
        - raw_output is a dict with key 'decision'
        - raw_output is an object with attribute 'prediction'

        More sophisticated parsing (e.g. full LLM JSON parsing) can later be
        implemented either here or in tasks/output_parsers.py.
        """
        self.validate_sample(sample)

        candidate = raw_output

        if isinstance(raw_output, dict):
            if "prediction" in raw_output:
                candidate = raw_output["prediction"]
            elif "decision" in raw_output:
                candidate = raw_output["decision"]

        elif hasattr(raw_output, "prediction"):
            candidate = getattr(raw_output, "prediction")

        return self.normalize_label(candidate)