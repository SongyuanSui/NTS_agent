from __future__ import annotations

import numpy as np

from core.enums import RepresentationType, TaskType
from core.schemas import QueryInstance, TaskSpec, TimeSeriesSample
from prompts.builders import (
    AnomalyWindowPromptBuilder,
    ClassificationPromptBuilder,
    PredictionPromptBuilder,
    SummaryPromptBuilder,
    build_prompt,
    get_prompt_builder,
)
from retrieval.schemas import RetrievedExample, RetrievedSet, RetrievalScore


def _query(task_type: TaskType = TaskType.CLASSIFICATION) -> QueryInstance:
    task_spec = TaskSpec(
        task_type=task_type,
        label_space=["A", "B"] if task_type == TaskType.CLASSIFICATION else [],
        granularity="sample",
        description="Test task.",
    )
    sample = TimeSeriesSample(sample_id="sample-1", x=np.array([1.0, 2.0, 3.0]))
    return QueryInstance(
        query_id="query-1",
        sample=sample,
        task_spec=task_spec,
        metadata={"split": "test"},
    )


def test_get_prompt_builder_dispatches_by_task_type() -> None:
    assert isinstance(get_prompt_builder(TaskType.CLASSIFICATION), ClassificationPromptBuilder)
    assert isinstance(get_prompt_builder("prediction"), PredictionPromptBuilder)
    assert isinstance(get_prompt_builder(TaskType.ANOMALY_WINDOW), AnomalyWindowPromptBuilder)
    assert isinstance(get_prompt_builder("summary"), SummaryPromptBuilder)


def test_build_prompt_formats_classification_messages() -> None:
    query = _query()

    output = build_prompt(query.task_spec, query)
    messages = output.to_messages()

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Allowed labels:" in output.user_message
    assert "[A, B]" in output.user_message
    assert "Channel 0: [1, 2, 3]" in output.user_message
    assert output.metadata["task_type"] == TaskType.CLASSIFICATION


def test_build_prompt_includes_retrieved_examples() -> None:
    query = _query()
    retrieved = RetrievedSet(
        query_id=query.query_id,
        retrieval_mode="ts",
        examples=[
            RetrievedExample(
                sample_id="neighbor-1",
                label="A",
                channel_id=0,
                representation_type=RepresentationType.TS,
                score=RetrievalScore(value=0.25, higher_is_better=False),
                payload=np.array([1.0, 2.1, 3.2]),
            )
        ],
    )

    output = build_prompt(query.task_spec, query, retrieved_sets={0: retrieved})

    assert "neighbor-1" in output.user_message
    assert "Label: A" in output.user_message
    assert "Payload: [1, 2.1, 3.2]" in output.user_message
