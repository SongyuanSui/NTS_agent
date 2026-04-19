from __future__ import annotations

from typing import Any, Iterable, Optional

from core.enums import RepresentationType
from core.schemas import TimeSeriesSample
from core.schemas import RepresentationRecord
from representations.rep_base import BaseRepresentation
from representations.schemas import RepresentationInput, RepresentationOutput
from representations.stat_feature.feature_calculation import (
	generate_single_statistical_features,
	generate_statistical_prompts,
)

# Invoke 2 main API for statistics feature computation.
def compute_statistics_for_sample(
	sample: TimeSeriesSample,
	feature_groups: Optional[Iterable[str]] = None,
	channel_id: int = 0,
) -> dict[str, float]:
	features_df = generate_single_statistical_features(
		sample,
		file_name_list=feature_groups,
		channel_id=channel_id,
	)
	return {str(k): float(v) for k, v in features_df.iloc[0].to_dict().items()}


def compute_statistics_for_batch(
	samples: list[TimeSeriesSample],
	feature_groups: Optional[Iterable[str]] = None,
	channel_id: int = 0,
	verbose: bool = False,
) -> dict | None:
	return generate_statistical_prompts(
		data=samples,
		file_name_list=feature_groups,
		channel_id=channel_id,
		verbose=verbose,
	)


class StatisticsRepresentation(BaseRepresentation):
	"""Representation component that produces statistical features."""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
	) -> None:
		super().__init__(name=name, config=config)

	@property
	def rep_type(self) -> RepresentationType:
		return RepresentationType.STATISTIC

	def transform(
		self,
		input_data: RepresentationInput,
		context: Optional[dict[str, Any]] = None,
	) -> RepresentationOutput:
		result = compute_statistics_for_batch(
			samples=input_data.samples,
			feature_groups=input_data.metadata.get("feature_groups"),
			channel_id=input_data.channel_id,
			verbose=bool(input_data.metadata.get("verbose", False)),
		)

		records: list[RepresentationRecord] = []
		if result is not None:
			records = _statistics_result_to_records(result)

		return RepresentationOutput(
			rep_type=self.rep_type,
			records=records,
			metadata={
				"channel_id": input_data.channel_id,
				"num_samples": len(input_data.samples),
				"num_records": len(records),
			},
		)


def _statistics_result_to_records(result: dict) -> list[RepresentationRecord]:
	"""Convert statistics batch output into RepresentationRecord list."""
	feature_names = result.get("stat_feature_names")
	feature_matrix = result.get("statistical_features")
	sample_names = result.get("sample_names")

	if feature_names is None or feature_matrix is None or sample_names is None:
		return [
			RepresentationRecord(
				rep_type=RepresentationType.STATISTIC,
				payload=result,
				metadata={"format": "raw_statistics_output"},
			)
		]

	records: list[RepresentationRecord] = []
	for idx, sample_id in enumerate(sample_names):
		row = feature_matrix[idx]
		payload = {str(name): float(value) for name, value in zip(feature_names, row)}
		records.append(
			RepresentationRecord(
				rep_type=RepresentationType.STATISTIC,
				payload=payload,
				metadata={"sample_id": str(sample_id)},
			)
		)
	return records
