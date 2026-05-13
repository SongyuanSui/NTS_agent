from __future__ import annotations

from typing import Any, Optional

from core.registry import TASK_REGISTRY
from core.schemas import QueryInstance, TimeSeriesSample
from tasks.output_parsers import parse_prediction_value
from tasks.task_base import BaseTask


@TASK_REGISTRY.decorator("prediction")
class PredictionTask(BaseTask):
    """Task implementation for future-target prediction/forecasting."""

    def __init__(
        self,
        task_spec,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(task_spec=task_spec, name=name, config=config)
        if not self.task_spec.is_prediction:
            raise ValueError(
                f"{self.name}: PredictionTask requires task_type='prediction', "
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
                "task_family": "prediction",
                "horizon": self.get_config("horizon", sample.metadata.get("horizon")),
                "target_name": self.get_config("target_name", sample.metadata.get("target_name")),
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
                "task_family": "prediction",
            },
        )
        return query

    def get_prompt_target(self) -> str:
        horizon = self.get_config("horizon", self.task_spec.metadata.get("horizon"))
        if horizon is not None:
            return f"predict the future target {horizon} step(s) ahead"
        return "predict the future target of the input time series"

    def parse_output(
        self,
        raw_output: Any,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        context = self.normalize_context(context)
        self.validate_sample(sample)
        prediction = parse_prediction_value(raw_output)
        self.log_event(
            context,
            event_type="task_parse_output",
            payload={
                "task_name": self.name,
                "task_type": self.task_spec.task_type.value,
                "sample_id": sample.sample_id,
            },
        )
        return prediction
