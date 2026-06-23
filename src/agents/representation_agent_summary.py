from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import RepresentationInput, RepresentationOutput
from core.enums import RepresentationType
from core.factories import build_llm_client
from core.registry import AGENT_REGISTRY
from representations.text_summary import TextSummaryRepresentation
from representations.schemas import RepresentationInput as RepInput
from utils.retrieval_compat import unwrap_payload


@AGENT_REGISTRY.decorator("representation_agent_summary")
class RepresentationAgentSummary(BaseAgent):
	"""Representation agent for LLM-based text summaries (SUMMARY view).

	Config keys (under agent.params):
	  channel_names : dict mapping channel_id (int) -> display name (str)
	  domain        : domain label forwarded to the LLM prompt (e.g. 'weather')
	  time_step     : temporal granularity label (e.g. 'hour', 'day')
	  entity        : optional location/entity name for the prompt
	  llm           : LLM config dict passed to core.factories.build_llm_client
	"""

	def validate_input(self, input_data: Any) -> None:
		if not isinstance(input_data, RepresentationInput):
			raise TypeError(
				f"{self.name}: input_data must be RepresentationInput, "
				f"but got {type(input_data).__name__}."
			)

	def _run_impl(
		self,
		input_data: RepresentationInput,
		context: Optional[dict[str, Any]] = None,
	) -> RepresentationOutput:
		query = input_data.query
		channels = input_data.channels

		if input_data.representation_type != RepresentationType.SUMMARY:
			raise ValueError(f"Expected SUMMARY representation, got {input_data.representation_type}")

		rep_component = TextSummaryRepresentation(name=self.name)
		channel_payloads: dict[int, Any] = {}
		rep_context = self._build_rep_context(context)

		for channel in channels:
			samples_for_channel = [query.sample]
			channel_names: dict[int, str] = self.get_config("channel_names", {})

			rep_input = RepInput(
				samples=samples_for_channel,
				channel_id=channel.channel_id,
				metadata={
					"source": "channel_decomposer",
					"channel_name": channel_names.get(channel.channel_id),
				},
			)

			rep_output = rep_component.run(rep_input, context=rep_context)

			if rep_output.records:
				channel_payloads[channel.channel_id] = unwrap_payload(rep_output.records[0].payload)

		self.log_info(
			context,
			"RepresentationAgentSummary '%s': created summaries for %d channels",
			self.name,
			len(channel_payloads),
		)

		return RepresentationOutput(
			query_id=query.query_id,
			representation_type=RepresentationType.SUMMARY,
			channel_payloads=channel_payloads,
			metadata={
				"sample_id": query.sample.sample_id,
				"num_channels": len(channels),
				"representation_type": RepresentationType.SUMMARY.value,
			},
		)

	def _build_rep_context(
		self,
		parent_context: Optional[dict[str, Any]],
	) -> dict[str, Any]:
		"""Build the context dict passed to TextSummaryRepresentation.run()."""
		llm_cfg: dict = self.get_config("llm", {})

		# Prefer an llm_client already in the parent context (e.g. injected by
		# the pipeline), but fall back to building one from the agent's own config.
		llm_client = (parent_context or {}).get("llm_client")
		if llm_client is None and llm_cfg:
			llm_client = build_llm_client(llm_cfg)

		channel_names: dict = self.get_config("channel_names", {})
		# YAML keys are strings; normalise to int so lookup by channel_id works.
		channel_names = {int(k): str(v) for k, v in channel_names.items()}

		return {
			"llm_client": llm_client,
			"channel_names": channel_names,
			"domain": self.get_config("domain", "time series"),
			"time_step": self.get_config("time_step", "step"),
			"entity": self.get_config("entity", ""),
		}
