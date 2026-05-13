from __future__ import annotations

from typing import Any, Optional

from core.registry import TASK_REGISTRY
from core.schemas import QueryInstance, TimeSeriesSample
from tasks.output_parsers import parse_binary_anomaly
from tasks.task_base import BaseTask


@TASK_REGISTRY.decorator("anomaly_window")
class AnomalyWindowTask(BaseTask):
    """Task implementation for window-level anomaly detection."""

    def __init__(
        self,
        task_spec,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(task_spec=task_spec, name=name, config=config)
        if not self.task_spec.is_anomaly_window:
            raise ValueError(
                f"{self.name}: AnomalyWindowTask requires task_type='anomaly_window', "
                f"but got '{self.task_spec.task_type.value}'."
            )

    def build_query(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> QueryInstance:
        context = self.normalize_context(context)
        self.validate_sample(sample)

        metadata = self._build_default_query_metadata(sample)
        metadata.update(
            {
                "task_family": "anomaly_window",
                "window_start": sample.metadata.get("window_start"),
                "window_end": sample.metadata.get("window_end"),
                "source_sequence_id": sample.metadata.get("source_sequence_id"),
                "has_ground_truth": sample.y is not None,
            }
        )

        query = QueryInstance(
            query_id=self._build_default_query_id(sample),
            sample=sample,
            task_spec=self.task_spec,
            channels=[],
            metadata=metadata,
        )
        self.log_event(
            context,
            event_type="task_build_query",
            payload={
                "task_name": self.name,
                "task_type": self.task_spec.task_type.value,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "task_family": "anomaly_window",
            },
        )
        return query

    def get_prompt_target(self) -> str:
        return "determine whether the time-series window is normal or anomalous"

    def parse_output(
        self,
        raw_output: Any,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        context = self.normalize_context(context)
        self.validate_sample(sample)
        prediction = parse_binary_anomaly(raw_output)
        self.log_event(
            context,
            event_type="task_parse_output",
            payload={
                "task_name": self.name,
                "task_type": self.task_spec.task_type.value,
                "sample_id": sample.sample_id,
                "normalized_prediction": prediction,
            },
        )
        return prediction
