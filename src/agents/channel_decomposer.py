from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import ChannelDecomposerInput, ChannelDecomposerOutput
from core.registry import AGENT_REGISTRY
from core.schemas import ChannelData


@AGENT_REGISTRY.decorator("channel_decomposer")
class ChannelDecomposerAgent(BaseAgent):
	"""
	Per-sample channel decomposition agent.

	Extracts individual channels from a multivariate time series sample
	for independent processing in downstream agents.
	"""

	def validate_input(self, input_data: Any) -> None:
		if not isinstance(input_data, ChannelDecomposerInput):
			raise TypeError(
				f"{self.name}: input_data must be ChannelDecomposerInput, "
				f"but got {type(input_data).__name__}."
			)

	def _run_impl(
		self,
		input_data: ChannelDecomposerInput,
		context: Optional[dict[str, Any]] = None,
	) -> ChannelDecomposerOutput:
		query = input_data.query
		sample = query.sample
		selected_channel_ids = input_data.selected_channel_ids

		all_channels: list[ChannelData] = []

		if sample.is_univariate:
			channel_data = ChannelData(
				sample_id=sample.sample_id,
				channel_id=0,
				values=sample.x,
				metadata={"data_mode": "univariate"},
			)
			all_channels.append(channel_data)
		else:
			for ch_id in range(sample.num_channels):
				channel_data = ChannelData(
					sample_id=sample.sample_id,
					channel_id=ch_id,
					values=sample.x[:, ch_id],
					metadata={"data_mode": "multivariate"},
				)
				all_channels.append(channel_data)

		selected_channels = all_channels
		selected_ids = [ch.channel_id for ch in all_channels]

		if selected_channel_ids is not None:
			selected_channels = [ch for ch in all_channels if ch.channel_id in selected_channel_ids]
			selected_ids = [ch.channel_id for ch in selected_channels]

		self.log_info(
			context,
			"ChannelDecomposer '%s': decomposed %d channels, selected %d",
			self.name,
			len(all_channels),
			len(selected_channels),
		)

		return ChannelDecomposerOutput(
			query_id=query.query_id,
			all_channels=all_channels,
			selected_channels=selected_channels,
			selected_channel_ids=selected_ids,
			metadata={
				"sample_id": sample.sample_id,
				"data_mode": sample.data_mode.value,
				"num_channels": len(all_channels),
				"num_selected": len(selected_channels),
			},
		)
