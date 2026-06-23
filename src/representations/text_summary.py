from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.enums import RepresentationType
from core.schemas import RepresentationRecord, TimeSeriesSample
from representations.rep_base import BaseRepresentation
from representations.schemas import RepresentationInput, RepresentationOutput


class TextSummaryRepresentation(BaseRepresentation):
	"""Representation component that produces LLM-based text summaries of time series."""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
	) -> None:
		super().__init__(name=name, config=config)

	@property
	def rep_type(self) -> RepresentationType:
		return RepresentationType.SUMMARY

	def transform(
		self,
		input_data: RepresentationInput,
		context: Optional[dict[str, Any]] = None,
	) -> RepresentationOutput:
		records: list[RepresentationRecord] = []

		for sample in input_data.samples:
			summary = generate_text_summary(
				sample,
				channel_id=input_data.channel_id,
				context=context,
				channel_name=input_data.metadata.get("channel_name"),
			)

			records.append(
				RepresentationRecord(
					rep_type=self.rep_type,
					payload=summary,
					metadata={
						"sample_id": sample.sample_id,
						"channel_id": input_data.channel_id,
					},
				)
			)

		return RepresentationOutput(
			rep_type=self.rep_type,
			records=records,
			metadata={
				"channel_id": input_data.channel_id,
				"num_samples": len(input_data.samples),
				"num_records": len(records),
			},
		)


def generate_text_summary(
	sample: TimeSeriesSample,
	channel_id: int = 0,
	context: Optional[dict[str, Any]] = None,
	channel_name: Optional[str] = None,
) -> str:
	"""Generate an LLM-based text summary of a time series channel.

	context must contain 'llm_client' (an LLMClient instance).
	Optional context keys: 'channel_names' (dict[int, str]), 'domain', 'time_step', 'entity'.
	"""
	x = sample.x
	if x.ndim == 2:
		if channel_id < 0 or channel_id >= x.shape[1]:
			raise ValueError(
				f"channel_id out of range for sample '{sample.sample_id}': "
				f"got {channel_id}, valid range is [0, {x.shape[1] - 1}]"
			)
		x = x[:, channel_id]

	return _llm_channel_summary(x, channel_id, channel_name, context)


def _llm_channel_summary(
	x: np.ndarray,
	channel_id: int,
	channel_name: Optional[str],
	context: Optional[dict[str, Any]],
) -> str:
	"""Call an LLM to produce a per-channel natural-language summary.

	Follows the by-channel approach from localLM_TimeCAP.contextualize_channel:
	one LLM call per channel, prompt asks for trend / range / interpretation in
	three sentences without numerical values.

	Falls back to _basic_summary if no llm_client is available in context.
	"""
	from llm.client_base import LLMRequest
	from prompts.templates.summary import CHANNEL_SYSTEM_PROMPT, CHANNEL_USER_TEMPLATE

	ctx = context or {}
	llm_client = ctx.get("llm_client")
	if llm_client is None:
		raise ValueError("context['llm_client'] is required for LLM-based text summarization")

	channel_names: dict[int, str] = ctx.get("channel_names", {})
	name = channel_name or channel_names.get(channel_id, f"Channel {channel_id}")
	domain: str = ctx.get("domain", "time series")
	time_step: str = ctx.get("time_step", "step")
	entity: str = ctx.get("entity", "")

	indicator_series = "|".join(f"{v:.2f}" for v in x)
	window_size = len(x)

	system_prompt = CHANNEL_SYSTEM_PROMPT.format(DOMAIN=domain)
	user_prompt = CHANNEL_USER_TEMPLATE.format(
		ENTITY=entity,
		INDICATOR_NAME=name,
		WINDOW_SIZE=window_size,
		TIME_STEP=time_step,
		INDICATOR_SERIES=indicator_series,
	)

	request = LLMRequest(
		messages=[{"role": "user", "content": user_prompt}],
		system=system_prompt,
	)

	# print("=== LLM REQUEST ===")
	# print(f"[system]\n{system_prompt}")
	# print(f"[user]\n{user_prompt}")
	# print("===================")

	response = llm_client.complete(request)
	result = response.content.strip()

	# print(f"[response]\n{result}")
	# print("===================")

	# return result


def compute_summary_for_sample(
	sample: TimeSeriesSample,
	channel_id: int = 0,
	context: Optional[dict[str, Any]] = None,
) -> str:
	"""Generate an LLM-based text summary for a single sample."""
	return generate_text_summary(sample, channel_id=channel_id, context=context)


def compute_summary_for_batch(
	samples: list[TimeSeriesSample],
	channel_id: int = 0,
	context: Optional[dict[str, Any]] = None,
) -> list[str]:
	"""Generate LLM-based text summaries for a batch of samples."""
	return [compute_summary_for_sample(sample, channel_id=channel_id, context=context) for sample in samples]
