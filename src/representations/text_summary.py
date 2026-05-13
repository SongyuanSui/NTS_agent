from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.enums import RepresentationType
from core.schemas import RepresentationRecord, TimeSeriesSample
from representations.rep_base import BaseRepresentation
from representations.schemas import RepresentationInput, RepresentationOutput


class TextSummaryRepresentation(BaseRepresentation):
	"""Representation component that produces text summaries of time series."""

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
				style=input_data.metadata.get("style", "basic"),
			)

			records.append(
				RepresentationRecord(
					rep_type=self.rep_type,
					payload=summary,
					metadata={
						"sample_id": sample.sample_id,
						"channel_id": input_data.channel_id,
						"style": input_data.metadata.get("style", "basic"),
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
	style: str = "basic",
) -> str:
	"""Generate a text summary of a time series."""
	x = sample.x
	if x.ndim == 2:
		if channel_id < 0 or channel_id >= x.shape[1]:
			raise ValueError(
				f"channel_id out of range for sample '{sample.sample_id}': "
				f"got {channel_id}, valid range is [0, {x.shape[1] - 1}]"
			)
		x = x[:, channel_id]

	if style == "basic":
		return _basic_summary(x)
	elif style == "statistical":
		return _statistical_summary(x)
	else:
		return _basic_summary(x)


def _basic_summary(x: np.ndarray) -> str:
	"""Generate a basic text summary with key statistics."""
	mean_val = float(np.mean(x))
	std_val = float(np.std(x))
	min_val = float(np.min(x))
	max_val = float(np.max(x))
	length = len(x)

	return (
		f"Time series with {length} observations. "
		f"Mean: {mean_val:.4f}, Std: {std_val:.4f}, "
		f"Min: {min_val:.4f}, Max: {max_val:.4f}."
	)


def _statistical_summary(x: np.ndarray) -> str:
	"""Generate a detailed statistical summary."""
	mean_val = float(np.mean(x))
	median_val = float(np.median(x))
	std_val = float(np.std(x))
	min_val = float(np.min(x))
	max_val = float(np.max(x))
	q25 = float(np.percentile(x, 25))
	q75 = float(np.percentile(x, 75))
	length = len(x)

	trend = "increasing" if x[-1] > x[0] else "decreasing"

	return (
		f"Time series of length {length}. "
		f"Mean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}. "
		f"Range: [{min_val:.4f}, {max_val:.4f}]. "
		f"IQR: [{q25:.4f}, {q75:.4f}]. "
		f"Overall trend: {trend}."
	)


def compute_summary_for_sample(
	sample: TimeSeriesSample,
	channel_id: int = 0,
	style: str = "basic",
) -> str:
	"""Generate a text summary for a single sample."""
	return generate_text_summary(sample, channel_id=channel_id, style=style)


def compute_summary_for_batch(
	samples: list[TimeSeriesSample],
	channel_id: int = 0,
	style: str = "basic",
) -> list[str]:
	"""Generate text summaries for a batch of samples."""
	return [compute_summary_for_sample(sample, channel_id=channel_id, style=style) for sample in samples]
