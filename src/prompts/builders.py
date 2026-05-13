"""Task-specific prompt builders."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from core.enums import TaskType
from core.schemas import ChannelData, QueryInstance, TaskSpec
from prompts.formatters import (
    format_channels_for_llm,
    format_retrieved_set_for_llm,
    format_series_for_llm,
)
from prompts.prompt_base import PromptBuilder, PromptContext, PromptOutput
from prompts.templates import anomaly_window, classification, prediction, summary
from retrieval.schemas import RetrievedSet


class TimeSeriesPromptBuilder(PromptBuilder):
    """Shared formatting behavior for time-series prompt builders."""

    template_module: Any = None

    def build_system_prompt(self, context: PromptContext) -> str:
        return str(self.get_config("system_prompt", self.template_module.SYSTEM_PROMPT))

    def _query_block(self, query: QueryInstance) -> str:
        decimals = int(self.get_config("decimals", 3))
        max_items = self.get_config("max_items_per_channel", 128)
        include_score = bool(self.get_config("include_channel_score", True))

        lines = [
            f"query_id: {query.query_id}",
            f"sample_id: {query.sample.sample_id}",
            f"sequence_length: {query.sample.length}",
            f"num_channels: {query.sample.num_channels}",
        ]

        if query.metadata:
            lines.append(f"query_metadata: {_format_mapping(query.metadata)}")
        if query.sample.metadata:
            lines.append(f"sample_metadata: {_format_mapping(query.sample.metadata)}")

        channels = query.channels if query.channels else _channels_from_sample(query)
        if channels:
            lines.append("series:")
            lines.append(
                format_channels_for_llm(
                    channels,
                    decimals=decimals,
                    max_items_per_channel=max_items,
                    include_score=include_score,
                )
            )
        else:
            lines.append("series:")
            lines.append(
                format_series_for_llm(
                    query.sample.x.reshape(-1),
                    decimals=decimals,
                    max_items=max_items,
                    include_length_suffix=True,
                )
            )

        return "\n".join(lines)

    def _retrieved_block(self, context: PromptContext) -> str:
        if not context.retrieved_sets:
            return "(No retrieved examples provided.)"

        decimals = int(self.get_config("decimals", 3))
        max_items = self.get_config("max_items_per_retrieved_payload", 64)
        include_payload = bool(self.get_config("include_retrieved_payload", True))

        blocks: list[str] = []
        for key, retrieved_set in _sorted_items(context.retrieved_sets):
            title = f"Channel {key}" if isinstance(key, int) else str(key)
            if isinstance(retrieved_set, RetrievedSet):
                body = format_retrieved_set_for_llm(
                    retrieved_set,
                    decimals=decimals,
                    max_items_per_payload=max_items,
                    include_payload=include_payload,
                )
            else:
                body = str(retrieved_set)
            blocks.append(f"{title}:\n{body}")

        return "\n\n".join(blocks)

    def _task_description(self, task_spec: TaskSpec) -> str:
        if task_spec.description:
            return task_spec.description
        return f"Task type: {task_spec.task_type.value}; granularity: {task_spec.granularity}."

    def _label_space(self, task_spec: TaskSpec, fallback: str = "(No fixed label space.)") -> str:
        if not task_spec.label_space:
            return fallback
        return "[" + ", ".join(task_spec.label_space) + "]"


class ClassificationPromptBuilder(TimeSeriesPromptBuilder):
    """Prompt builder for classification tasks."""

    template_module = classification

    def build_user_message(self, context: PromptContext) -> str:
        return classification.USER_TEMPLATE.format(
            task_description=self._task_description(context.task_spec),
            label_space=self._label_space(context.task_spec),
            query_block=self._query_block(context.query),
            retrieved_block=self._retrieved_block(context),
        )


class PredictionPromptBuilder(TimeSeriesPromptBuilder):
    """Prompt builder for forecasting or future-target prediction tasks."""

    template_module = prediction

    def build_user_message(self, context: PromptContext) -> str:
        metadata = {}
        metadata.update(context.task_spec.metadata)
        metadata.update(context.query.metadata)

        return prediction.USER_TEMPLATE.format(
            task_description=self._task_description(context.task_spec),
            prediction_metadata=_format_mapping(metadata) if metadata else "(None provided.)",
            query_block=self._query_block(context.query),
            retrieved_block=self._retrieved_block(context),
        )


class AnomalyWindowPromptBuilder(TimeSeriesPromptBuilder):
    """Prompt builder for anomaly-window detection tasks."""

    template_module = anomaly_window

    def build_user_message(self, context: PromptContext) -> str:
        metadata = {}
        metadata.update(context.task_spec.metadata)
        metadata.update(context.query.metadata)
        label_fallback = "[normal, anomaly]"

        return anomaly_window.USER_TEMPLATE.format(
            task_description=self._task_description(context.task_spec),
            anomaly_metadata=_format_mapping(metadata) if metadata else "(None provided.)",
            label_space=self._label_space(context.task_spec, fallback=label_fallback),
            query_block=self._query_block(context.query),
            retrieved_block=self._retrieved_block(context),
        )


class SummaryPromptBuilder(TimeSeriesPromptBuilder):
    """Prompt builder for LLM-generated textual summaries."""

    template_module = summary

    def build_user_message(self, context: PromptContext) -> str:
        return summary.USER_TEMPLATE.format(
            summary_style=str(self.get_config("style", "statistical")),
            query_block=self._query_block(context.query),
        )

    def build(self, context: PromptContext) -> PromptOutput:
        output = super().build(context)
        output.metadata["purpose"] = "summary"
        return output


def get_prompt_builder(
    task_type: TaskType | str,
    config: dict[str, Any] | None = None,
) -> PromptBuilder:
    """Create a prompt builder for a task type."""
    if isinstance(task_type, str):
        if task_type == "summary":
            return SummaryPromptBuilder(config=config)
        task_type = TaskType(task_type)

    builder_cls: type[PromptBuilder]
    if task_type == TaskType.CLASSIFICATION:
        builder_cls = ClassificationPromptBuilder
    elif task_type == TaskType.PREDICTION:
        builder_cls = PredictionPromptBuilder
    elif task_type in {TaskType.ANOMALY_WINDOW, TaskType.ANOMALY_SEQUENCE}:
        builder_cls = AnomalyWindowPromptBuilder
    else:
        raise ValueError(f"Unsupported task_type for prompt builder: {task_type!r}")

    return builder_cls(config=config)


def build_prompt(
    task_spec: TaskSpec,
    query: QueryInstance,
    retrieved_sets: dict[int, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> PromptOutput:
    """Convenience function that builds a complete prompt for a query."""
    context = PromptContext(
        task_spec=task_spec,
        query=query,
        retrieved_sets=retrieved_sets or {},
    )
    builder = get_prompt_builder(task_spec.task_type, config=config)
    return builder.build(context)


def _channels_from_sample(query: QueryInstance) -> list[ChannelData]:
    sample = query.sample
    if sample.x.ndim == 1:
        return [
            ChannelData(
                sample_id=sample.sample_id,
                channel_id=0,
                values=sample.x,
                metadata={"source": "sample.x"},
            )
        ]

    return [
        ChannelData(
            sample_id=sample.sample_id,
            channel_id=channel_id,
            values=sample.x[:, channel_id],
            metadata={"source": "sample.x"},
        )
        for channel_id in range(sample.num_channels)
    ]


def _format_mapping(mapping: dict[str, Any]) -> str:
    parts = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, np.ndarray):
            value_text = value.tolist()
        else:
            value_text = value
        parts.append(f"{key}={value_text}")
    return "{ " + ", ".join(parts) + " }"


def _sorted_items(mapping: dict[Any, Any]) -> Iterable[tuple[Any, Any]]:
    return sorted(mapping.items(), key=lambda item: (str(type(item[0])), str(item[0])))
