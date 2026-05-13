"""Representations module for feature extraction and caching."""

from representations.bundle import RepresentationBundle, RepresentationBundler
from representations.normalizers import (
	clip_array,
	normalize_array,
	normalize_records,
	scale_array,
)
from representations.raw_series import (
	RawSeriesRepresentation,
	compute_raw_series_for_batch,
	compute_raw_series_for_sample,
)
from representations.rep_base import BaseRepresentation
from representations.schemas import RepresentationInput, RepresentationOutput
from representations.statistics import (
	StatisticsRepresentation,
	compute_statistics_for_batch,
	compute_statistics_for_sample,
)
from representations.text_summary import (
	TextSummaryRepresentation,
	compute_summary_for_batch,
	compute_summary_for_sample,
)

__all__ = [
	"BaseRepresentation",
	"RepresentationInput",
	"RepresentationOutput",
	"StatisticsRepresentation",
	"compute_statistics_for_sample",
	"compute_statistics_for_batch",
	"RawSeriesRepresentation",
	"compute_raw_series_for_sample",
	"compute_raw_series_for_batch",
	"TextSummaryRepresentation",
	"compute_summary_for_sample",
	"compute_summary_for_batch",
	"RepresentationBundle",
	"RepresentationBundler",
	"normalize_array",
	"normalize_records",
	"scale_array",
	"clip_array",
]
