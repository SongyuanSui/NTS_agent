from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import RetrievalInput, RetrievalOutput
from core.registry import AGENT_REGISTRY
from retrieval.text_retrievers import TextSummaryRetriever


@AGENT_REGISTRY.decorator("retrieval_agent_text")
class RetrievalAgentText(BaseAgent):
	"""Retrieval agent using text summaries (SUMMARY view)."""

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

		retriever = TextSummaryRetriever(
			name=self.name,
			config=self.get_config("retriever_config", self.config),
		)

		retrieved_sets: dict[int, Any] = {}

		for channel in channels:
			channel_query = type(query)(
				query_id=query.query_id,
				sample=query.sample,
				task_spec=query.task_spec,
				channels=query.channels,
				metadata={**query.metadata, "channel_id": channel.channel_id},
			)

			retrieved_set = retriever.retrieve(
				query=channel_query,
				memory_bank=memory_bank,
				top_k=top_k,
				context=context,
			)

			retrieved_sets[channel.channel_id] = retrieved_set

		self.log_info(
			context,
			"RetrievalAgentText '%s': retrieved results for %d channels",
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
				"retrieval_mode": "text",
			},
		)
