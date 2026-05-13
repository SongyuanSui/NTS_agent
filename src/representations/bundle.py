from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import RepresentationType
from core.schemas import RepresentationRecord
from representations.schemas import RepresentationOutput


Metadata = dict[str, Any]


@dataclass(slots=True)
class RepresentationBundle:
	"""
	Container for bundling multiple representation types for a single sample or channel.

	This enables efficient storage and retrieval of different representation views
	(e.g., raw time series, text summary, statistical features) for the same sample.
	"""

	sample_id: str
	channel_id: int = 0
	ts_repr: Optional[RepresentationOutput] = None
	summary_repr: Optional[RepresentationOutput] = None
	statistic_repr: Optional[RepresentationOutput] = None
	metadata: Metadata = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not isinstance(self.sample_id, str) or not self.sample_id:
			raise ValueError("sample_id must be a non-empty string.")

		self.channel_id = int(self.channel_id)
		if self.channel_id < 0:
			raise ValueError("channel_id must be a non-negative integer.")

		if not isinstance(self.metadata, dict):
			self.metadata = dict(self.metadata)

	def add_representation(
		self,
		repr_output: RepresentationOutput,
		rep_type: Optional[RepresentationType | str] = None,
	) -> None:
		"""Add a representation output to the bundle."""
		if not isinstance(repr_output, RepresentationOutput):
			raise TypeError("repr_output must be a RepresentationOutput.")

		rep_type = rep_type or repr_output.rep_type
		if isinstance(rep_type, str):
			rep_type = RepresentationType(rep_type)

		if rep_type == RepresentationType.TS:
			self.ts_repr = repr_output
		elif rep_type == RepresentationType.SUMMARY:
			self.summary_repr = repr_output
		elif rep_type == RepresentationType.STATISTIC:
			self.statistic_repr = repr_output
		else:
			raise ValueError(f"Unknown representation type: {rep_type}")

	def get_representation(
		self,
		rep_type: RepresentationType | str,
	) -> Optional[RepresentationOutput]:
		"""Get a specific representation from the bundle."""
		if isinstance(rep_type, str):
			rep_type = RepresentationType(rep_type)

		if rep_type == RepresentationType.TS:
			return self.ts_repr
		elif rep_type == RepresentationType.SUMMARY:
			return self.summary_repr
		elif rep_type == RepresentationType.STATISTIC:
			return self.statistic_repr
		else:
			return None

	def has_representation(self, rep_type: RepresentationType | str) -> bool:
		"""Check if a representation type is present."""
		return self.get_representation(rep_type) is not None

	def get_records(
		self,
		rep_type: RepresentationType | str,
	) -> list[RepresentationRecord]:
		"""Get records for a specific representation type."""
		repr_output = self.get_representation(rep_type)
		return repr_output.records if repr_output is not None else []

	def get_available_types(self) -> list[RepresentationType]:
		"""Get list of available representation types in this bundle."""
		types = []
		if self.ts_repr is not None:
			types.append(RepresentationType.TS)
		if self.summary_repr is not None:
			types.append(RepresentationType.SUMMARY)
		if self.statistic_repr is not None:
			types.append(RepresentationType.STATISTIC)
		return types

	def to_dict(self) -> dict[str, Any]:
		"""Convert bundle to dictionary."""
		return {
			"sample_id": self.sample_id,
			"channel_id": self.channel_id,
			"ts_repr": self.ts_repr,
			"summary_repr": self.summary_repr,
			"statistic_repr": self.statistic_repr,
			"metadata": self.metadata,
		}

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> RepresentationBundle:
		"""Create bundle from dictionary."""
		return cls(
			sample_id=data["sample_id"],
			channel_id=data.get("channel_id", 0),
			ts_repr=data.get("ts_repr"),
			summary_repr=data.get("summary_repr"),
			statistic_repr=data.get("statistic_repr"),
			metadata=data.get("metadata", {}),
		)


class RepresentationBundler:
	"""Utility class for creating and managing representation bundles."""

	@staticmethod
	def create_bundle(
		sample_id: str,
		channel_id: int = 0,
	) -> RepresentationBundle:
		"""Create an empty representation bundle."""
		return RepresentationBundle(
			sample_id=sample_id,
			channel_id=channel_id,
		)

	@staticmethod
	def merge_bundles(
		bundles: list[RepresentationBundle],
		merged_id: Optional[str] = None,
	) -> RepresentationBundle:
		"""
		Merge multiple bundles by taking the first non-None representation of each type.

		Parameters
		----------
		bundles : list[RepresentationBundle]
			Bundles to merge.
		merged_id : str, optional
			ID for the merged bundle. If None, uses the first bundle's sample_id.

		Returns
		-------
		RepresentationBundle
			Merged bundle.
		"""
		if not bundles:
			raise ValueError("bundles list is empty.")

		merged = RepresentationBundle(
			sample_id=merged_id or bundles[0].sample_id,
			channel_id=bundles[0].channel_id,
		)

		for bundle in bundles:
			if bundle.ts_repr is not None and merged.ts_repr is None:
				merged.ts_repr = bundle.ts_repr
			if bundle.summary_repr is not None and merged.summary_repr is None:
				merged.summary_repr = bundle.summary_repr
			if bundle.statistic_repr is not None and merged.statistic_repr is None:
				merged.statistic_repr = bundle.statistic_repr

		return merged

	@staticmethod
	def extract_records_by_type(
		bundle: RepresentationBundle,
		rep_type: RepresentationType | str,
	) -> list[RepresentationRecord]:
		"""Extract records of a specific type from a bundle."""
		return bundle.get_records(rep_type)

	@staticmethod
	def extract_all_records(
		bundle: RepresentationBundle,
	) -> dict[RepresentationType, list[RepresentationRecord]]:
		"""Extract all records organized by representation type."""
		result = {}
		for rep_type in bundle.get_available_types():
			result[rep_type] = bundle.get_records(rep_type)
		return result
