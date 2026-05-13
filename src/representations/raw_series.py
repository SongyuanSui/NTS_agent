from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.enums import RepresentationType
from core.schemas import RepresentationRecord, TimeSeriesSample
from representations.rep_base import BaseRepresentation
from representations.schemas import RepresentationInput, RepresentationOutput


class RawSeriesRepresentation(BaseRepresentation):
	"""Representation component that produces raw time series data."""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
	) -> None:
		super().__init__(name=name, config=config)

	@property
	def rep_type(self) -> RepresentationType:
		return RepresentationType.TS

	def transform(
		self,
		input_data: RepresentationInput,
		context: Optional[dict[str, Any]] = None,
	) -> RepresentationOutput:
		records: list[RepresentationRecord] = []

		for sample in input_data.samples:
			x = sample.x
			if x.ndim == 2:
				x = x[:, input_data.channel_id]

			records.append(
				RepresentationRecord(
					rep_type=self.rep_type,
					payload=x.astype(float),
					metadata={
						"sample_id": sample.sample_id,
						"channel_id": input_data.channel_id,
						"length": len(x),
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


def compute_raw_series_for_sample(
	sample: TimeSeriesSample,
	channel_id: int = 0,
) -> np.ndarray:
	"""Extract raw time series array for a single sample."""
	x = sample.x
	if x.ndim == 2:
		if channel_id < 0 or channel_id >= x.shape[1]:
			raise ValueError(
				f"channel_id out of range for sample '{sample.sample_id}': "
				f"got {channel_id}, valid range is [0, {x.shape[1] - 1}]"
			)
		return x[:, channel_id].astype(float)
	return x.astype(float)


def compute_raw_series_for_batch(
	samples: list[TimeSeriesSample],
	channel_id: int = 0,
) -> list[np.ndarray]:
	"""Extract raw time series arrays for a batch of samples."""
	return [compute_raw_series_for_sample(sample, channel_id=channel_id) for sample in samples]
