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

    Execution flow
    --------------
    1. task.build_query(sample)
    2. channel_decomposer.run(...)
    3. retrieval_agent.run(...)
    4. reasoner_agent.run(...)
    5. aggregator_agent.run(...)
    6. package final PredictionRecord

    Notes
    -----
    - Representation agents are not explicitly called yet here, because the
      current retrieval agents may internally depend on:
        * raw TS only
        * or pre-built memory views
        * or later explicit representation outputs
    - selected_channel_ids are expected to come from:
        * context["selected_channel_ids"], or
        * pipeline config "selected_channel_ids"
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

        self.log_info(
            context,
            "InferencePipeline '%s': building query for sample_id=%s",
            self.name,
            sample.sample_id,
        )

        # ------------------------------------------------------------------
        # Step 1: build task-specific query
        # ------------------------------------------------------------------
        query = task.build_query(sample)
        if not isinstance(query, QueryInstance):
            raise TypeError(
                f"{self.name}: task.build_query(...) must return QueryInstance, "
                f"but got {type(query).__name__}."
            )

        self.log_event(
            context,
            event_type="query_built",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "task_type": query.task_spec.task_type.value,
            },
        )

        # ------------------------------------------------------------------
        # Step 2: channel decomposition
        # ------------------------------------------------------------------
        selected_channel_ids = self._resolve_selected_channel_ids(context=context)
        if selected_channel_ids is not None:
            self.log_info(
                context,
                "InferencePipeline '%s': using selected_channel_ids=%s",
                self.name,
                selected_channel_ids,
            )

        self.log_event(
            context,
            event_type="channel_decomposer_start",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "selected_channel_ids": selected_channel_ids,
            },
        )

        decomposer_input = ChannelDecomposerInput(
            query=query,
            selected_channel_ids=selected_channel_ids,
        )
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

        self.log_info(
            context,
            "InferencePipeline '%s': decomposed sample_id=%s into %d selected channels",
            self.name,
            sample.sample_id,
            len(selected_channels),
        )
        self.log_event(
            context,
            event_type="channel_decomposer_end",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "num_all_channels": len(decomposer_output.all_channels),
                "num_selected_channels": len(decomposer_output.selected_channels),
                "selected_channel_ids": decomposer_output.selected_channel_ids,
            },
        )

        # ------------------------------------------------------------------
        # Step 3: retrieval
        # ------------------------------------------------------------------
        top_k = self.get_config("top_k", 3)

        self.log_info(
            context,
            "InferencePipeline '%s': retrieval start for query_id=%s with top_k=%s",
            self.name,
            query.query_id,
            top_k,
        )
        self.log_event(
            context,
            event_type="retrieval_start",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "top_k": top_k,
                "selected_channel_ids": decomposer_output.selected_channel_ids,
            },
        )

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

        retrieval_sizes = {
            int(channel_id): len(retrieved_set)
            for channel_id, retrieved_set in retrieval_output.retrieved_sets.items()
        }

        self.log_info(
            context,
            "InferencePipeline '%s': retrieval finished for query_id=%s; channel_topk_sizes=%s",
            self.name,
            query.query_id,
            retrieval_sizes,
        )
        self.log_event(
            context,
            event_type="retrieval_end",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "retrieved_set_sizes": retrieval_sizes,
            },
        )

        # ------------------------------------------------------------------
        # Step 4: reasoning
        # ------------------------------------------------------------------
        self.log_info(
            context,
            "InferencePipeline '%s': reasoning start for query_id=%s",
            self.name,
            query.query_id,
        )
        self.log_event(
            context,
            event_type="reasoning_start",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "num_channel_inputs": len(retrieval_output.retrieved_sets),
            },
        )

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

        self.log_info(
            context,
            "InferencePipeline '%s': reasoning finished for query_id=%s; num_channel_decisions=%d",
            self.name,
            query.query_id,
            len(reasoner_output.channel_decisions),
        )
        self.log_event(
            context,
            event_type="reasoning_end",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "num_channel_decisions": len(reasoner_output.channel_decisions),
            },
        )

        # ------------------------------------------------------------------
        # Step 5: aggregation
        # ------------------------------------------------------------------
        self.log_info(
            context,
            "InferencePipeline '%s': aggregation start for query_id=%s",
            self.name,
            query.query_id,
        )
        self.log_event(
            context,
            event_type="aggregation_start",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "num_channel_decisions": len(reasoner_output.channel_decisions),
            },
        )

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

        self.log_info(
            context,
            "InferencePipeline '%s': aggregation finished for query_id=%s; raw_prediction=%r confidence=%r",
            self.name,
            query.query_id,
            aggregation_output.prediction,
            aggregation_output.confidence,
        )
        self.log_event(
            context,
            event_type="aggregation_end",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "raw_prediction": aggregation_output.prediction,
                "confidence": aggregation_output.confidence,
            },
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

        self.log_info(
            context,
            "InferencePipeline '%s': final prediction for sample_id=%s -> %r",
            self.name,
            sample.sample_id,
            parsed_prediction,
        )
        self.log_event(
            context,
            event_type="prediction_packaged",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
                "prediction": parsed_prediction,
                "confidence": aggregation_output.confidence,
            },
        )

        return PipelineResult(
            prediction=prediction_record,
            intermediates=intermediates,
            metadata={
                "sample_id": sample.sample_id,
                "query_id": query.query_id,
            },
        )

    def _resolve_selected_channel_ids(
        self,
        context: Optional[dict[str, Any]] = None,
    ) -> list[int] | None:
        """
        Resolve selected channel ids from runtime context first, then config.

        Priority
        --------
        1. context["selected_channel_ids"]
        2. self.config["selected_channel_ids"]

        Returns
        -------
        list[int] | None
        """
        selected_channel_ids = None

        if context is not None and "selected_channel_ids" in context:
            selected_channel_ids = context["selected_channel_ids"]
        elif "selected_channel_ids" in self.config:
            selected_channel_ids = self.config["selected_channel_ids"]

        if selected_channel_ids is None:
            return None

        if not isinstance(selected_channel_ids, list):
            selected_channel_ids = list(selected_channel_ids)

        selected_channel_ids = [int(x) for x in selected_channel_ids]
        for channel_id in selected_channel_ids:
            if channel_id < 0:
                raise ValueError(
                    f"{self.name}: selected_channel_ids must contain non-negative integers."
                )

        return selected_channel_ids

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