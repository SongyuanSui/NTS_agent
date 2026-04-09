# src/pipelines/inference_pipeline.py

from __future__ import annotations

from typing import Any, Optional

from agents.schemas import (
    AggregationInput,
    AggregationOutput,
    ChannelDecomposerInput,
    ChannelDecomposerOutput,
    ReasonerInput,
    ReasonerOutput,
    RetrievalInput,
    RetrievalOutput,
)
from core.schemas import PipelineResult, PredictionRecord, QueryInstance, TimeSeriesSample
from pipelines.pipeline_base import BasePipeline


class InferencePipeline(BasePipeline):
    """
    End-to-end inference pipeline for the current agentic framework.

    Minimal execution flow
    ----------------------
    1. task.build_query(sample)
    2. channel_decomposer.run(...)
    3. retrieval_agent.run(...)
    4. reasoner_agent.run(...)
    5. aggregator_agent.run(...)
    6. package final PredictionRecord

    Notes
    -----
    - This first version intentionally keeps the flow minimal.
    - Representation agents are not explicitly called yet here, because your
      retrieval agents may internally depend on:
        * raw TS only
        * or pre-built memory views
        * or later explicit representation outputs
    - Once your representation agents are ready, this pipeline can be extended
      without changing the public run(...) API.
    """

    REQUIRED_COMPONENTS = (
        "task",
        "channel_decomposer",
        "retrieval_agent",
        "reasoner_agent",
        "aggregator_agent",
    )

    def validate_components(self) -> None:
        super().validate_components()

        for key in self.REQUIRED_COMPONENTS:
            if not self.has_component(key):
                raise KeyError(f"{self.name}: required component '{key}' is missing.")

    def _run_impl(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        context = self.normalize_context(context)

        task = self.require_component("task")
        channel_decomposer = self.require_component("channel_decomposer")
        retrieval_agent = self.require_component("retrieval_agent")
        reasoner_agent = self.require_component("reasoner_agent")
        aggregator_agent = self.require_component("aggregator_agent")

        # ------------------------------------------------------------------
        # Step 1: build task-specific query
        # ------------------------------------------------------------------
        query = task.build_query(sample)
        if not isinstance(query, QueryInstance):
            raise TypeError(
                f"{self.name}: task.build_query(...) must return QueryInstance, "
                f"but got {type(query).__name__}."
            )

        # ------------------------------------------------------------------
        # Step 2: channel decomposition (and optional channel selection)
        # ------------------------------------------------------------------
        decomposer_input = ChannelDecomposerInput(query=query)
        decomposer_output = channel_decomposer.run(
            decomposer_input,
            context=self._build_agent_context(
                stage="channel_decomposer",
                query=query,
                sample=sample,
                extra_context=context,
            ),
        )

        if not isinstance(decomposer_output, ChannelDecomposerOutput):
            raise TypeError(
                f"{self.name}: channel_decomposer must return ChannelDecomposerOutput, "
                f"but got {type(decomposer_output).__name__}."
            )

        selected_channels = decomposer_output.selected_channels
        if len(selected_channels) == 0:
            raise ValueError(
                f"{self.name}: channel_decomposer returned no selected channels."
            )

        # ------------------------------------------------------------------
        # Step 3: retrieval
        # ------------------------------------------------------------------
        top_k = self.get_config("top_k", 3)

        retrieval_input = RetrievalInput(
            query=query,
            channels=selected_channels,
            top_k=top_k,
        )
        retrieval_output = retrieval_agent.run(
            retrieval_input,
            context=self._build_agent_context(
                stage="retrieval",
                query=query,
                sample=sample,
                extra_context=context,
                extras={
                    "decomposer_output": decomposer_output,
                },
            ),
        )

        if not isinstance(retrieval_output, RetrievalOutput):
            raise TypeError(
                f"{self.name}: retrieval_agent must return RetrievalOutput, "
                f"but got {type(retrieval_output).__name__}."
            )

        # ------------------------------------------------------------------
        # Step 4: reasoning
        # ------------------------------------------------------------------
        reasoner_input = ReasonerInput(
            query=query,
            task_spec=query.task_spec,
            retrieved_sets=retrieval_output.retrieved_sets,
        )
        reasoner_output = reasoner_agent.run(
            reasoner_input,
            context=self._build_agent_context(
                stage="reasoning",
                query=query,
                sample=sample,
                extra_context=context,
                extras={
                    "decomposer_output": decomposer_output,
                    "retrieval_output": retrieval_output,
                },
            ),
        )

        if not isinstance(reasoner_output, ReasonerOutput):
            raise TypeError(
                f"{self.name}: reasoner_agent must return ReasonerOutput, "
                f"but got {type(reasoner_output).__name__}."
            )

        # ------------------------------------------------------------------
        # Step 5: aggregation
        # ------------------------------------------------------------------
        aggregation_input = AggregationInput(
            query=query,
            channel_decisions=reasoner_output.channel_decisions,
        )
        aggregation_output = aggregator_agent.run(
            aggregation_input,
            context=self._build_agent_context(
                stage="aggregation",
                query=query,
                sample=sample,
                extra_context=context,
                extras={
                    "decomposer_output": decomposer_output,
                    "retrieval_output": retrieval_output,
                    "reasoner_output": reasoner_output,
                },
            ),
        )

        if not isinstance(aggregation_output, AggregationOutput):
            raise TypeError(
                f"{self.name}: aggregator_agent must return AggregationOutput, "
                f"but got {type(aggregation_output).__name__}."
            )

        # ------------------------------------------------------------------
        # Step 6: task-level output parsing and final packaging
        # ------------------------------------------------------------------
        parsed_prediction = task.parse_output(aggregation_output, sample)

        prediction_record = PredictionRecord(
            sample_id=sample.sample_id,
            task_type=query.task_spec.task_type,
            prediction=parsed_prediction,
            confidence=aggregation_output.confidence,
            reasoning=aggregation_output.reasoning,
            metadata={
                "query_id": query.query_id,
                "task_name": task.name,
                "pipeline_name": self.name,
            },
        )

        intermediates = {
            "query": query,
            "decomposer_output": decomposer_output,
            "retrieval_output": retrieval_output,
            "reasoner_output": reasoner_output,
            "aggregation_output": aggregation_output,
        }

        return PipelineResult(
            prediction=prediction_record,
            intermediates=intermediates,
            metadata={
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
            },
        )

    def _build_agent_context(
        self,
        stage: str,
        query: QueryInstance,
        sample: TimeSeriesSample,
        extra_context: Optional[dict[str, Any]] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Internal helper to build a consistent agent runtime context.
        """
        context: dict[str, Any] = {
            "stage": stage,
            "pipeline_name": self.name,
            "sample_id": sample.sample_id,
            "query_id": query.query_id,
            "task_type": query.task_spec.task_type.value,
        }

        if extra_context:
            context.update(extra_context)

        if extras:
            context.update(extras)

        return context