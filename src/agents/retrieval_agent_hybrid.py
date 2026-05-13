from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import RetrievalInput, RetrievalOutput
from core.registry import AGENT_REGISTRY
from retrieval.hybrid_retriever import HybridRetriever, WeightedHybridRetriever
from retrieval.raw_retrievers import RawSeriesRetriever
from retrieval.stat_retrievers import StatKNNRetriever
from retrieval.text_retrievers import TextSummaryRetriever


@AGENT_REGISTRY.decorator("retrieval_agent_hybrid")
class RetrievalAgentHybrid(BaseAgent):
	"""Hybrid retrieval agent combining multiple retrieval modes."""

	def validate_input(self, input_data: Any) -> None:
		if not isinstance(input_data, RetrievalInput):
			raise TypeError(
				f"{self.name}: input_data must be RetrievalInput, "
				f"but got {type(input_data).__name__}."
			)

	def _run_impl(
		self,
		input_data: RetrievalInput,
		context: Optional[dict[str, Any]] = None,
	) -> RetrievalOutput:
		query = input_data.query
		channels = input_data.channels
		top_k = input_data.top_k

		memory_bank = context.get("memory_bank") if context else None
		if memory_bank is None:
			raise ValueError(f"{self.name}: context['memory_bank'] is required")

		use_weights = bool(self.get_config("use_weights", False))
		fusion_method = str(self.get_config("fusion_method", "average"))

		if use_weights:
			hybrid_retriever = WeightedHybridRetriever(
				name=self.name,
				config={"fusion_method": fusion_method},
			)
		else:
			hybrid_retriever = HybridRetriever(
				name=self.name,
				config={"fusion_method": fusion_method},
			)

		ts_config = self.get_config("ts_config", {})
		stat_config = self.get_config("stat_config", {})
		text_config = self.get_config("text_config", {})

		ts_retriever = RawSeriesRetriever(config=ts_config)
		stat_retriever = StatKNNRetriever(config=stat_config)
		text_retriever = TextSummaryRetriever(config=text_config)

		hybrid_retriever.add_retriever("ts", ts_retriever)
		hybrid_retriever.add_retriever("stat", stat_retriever)
		hybrid_retriever.add_retriever("text", text_retriever)

		if use_weights and isinstance(hybrid_retriever, WeightedHybridRetriever):
			ts_weight = float(self.get_config("ts_weight", 0.33))
			stat_weight = float(self.get_config("stat_weight", 0.33))
			text_weight = float(self.get_config("text_weight", 0.34))
			hybrid_retriever.set_retriever_weight("ts", ts_weight)
			hybrid_retriever.set_retriever_weight("stat", stat_weight)
			hybrid_retriever.set_retriever_weight("text", text_weight)

		retrieved_sets: dict[int, Any] = {}

		for channel in channels:
			channel_query = type(query)(
				query_id=query.query_id,
				sample=query.sample,
				task_spec=query.task_spec,
				channels=query.channels,
				metadata={**query.metadata, "channel_id": channel.channel_id},
			)

			retrieved_set = hybrid_retriever.retrieve(
				query=channel_query,
				memory_bank=memory_bank,
				top_k=top_k,
				context=context,
			)

			retrieved_sets[channel.channel_id] = retrieved_set

		self.log_info(
			context,
			"RetrievalAgentHybrid '%s': hybrid retrieval for %d channels",
			self.name,
			len(retrieved_sets),
		)

		return RetrievalOutput(
			query_id=query.query_id,
			retrieved_sets=retrieved_sets,
			metadata={
				"sample_id": query.sample.sample_id,
				"num_channels": len(channels),
				"top_k": top_k,
				"retrieval_mode": "hybrid",
				"fusion_method": fusion_method,
			},
		)
