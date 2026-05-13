from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import RepresentationInput, RepresentationOutput
from core.enums import RepresentationType
from core.registry import AGENT_REGISTRY
from representations.statistics import StatisticsRepresentation
from representations.schemas import RepresentationInput as RepInput


@AGENT_REGISTRY.decorator("representation_agent_statistic")
class RepresentationAgentStatistic(BaseAgent):
	"""Representation agent for statistical features (STATISTIC view)."""

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

		if input_data.representation_type != RepresentationType.STATISTIC:
			raise ValueError(f"Expected STATISTIC representation, got {input_data.representation_type}")

		rep_component = StatisticsRepresentation(name=self.name)
		channel_payloads: dict[int, Any] = {}
		feature_groups = self.get_config("feature_groups")
		verbose = bool(self.get_config("verbose", False))

		for channel in channels:
			samples_for_channel = [query.sample]

			rep_input = RepInput(
				samples=samples_for_channel,
				channel_id=channel.channel_id,
				metadata={
					"feature_groups": feature_groups,
					"verbose": verbose,
					"source": "channel_decomposer",
				},
			)

			rep_output = rep_component.run(rep_input)

			if rep_output.records:
				channel_payloads[channel.channel_id] = rep_output.records[0].payload

		self.log_info(
			context,
			"RepresentationAgentStatistic '%s': created statistical features for %d channels",
			self.name,
			len(channel_payloads),
		)

		return RepresentationOutput(
			query_id=query.query_id,
			representation_type=RepresentationType.STATISTIC,
			channel_payloads=channel_payloads,
			metadata={
				"sample_id": query.sample.sample_id,
				"num_channels": len(channels),
				"representation_type": RepresentationType.STATISTIC.value,
				"feature_groups": str(feature_groups),
			},
		)
