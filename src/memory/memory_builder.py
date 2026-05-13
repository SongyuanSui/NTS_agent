from __future__ import annotations

from typing import Any, Iterable, Optional

from core.enums import RepresentationType, TaskType
from core.schemas import RepresentationRecord, TimeSeriesSample
from memory.schemas import MemoryEntry
from representations.schemas import RepresentationOutput


class MemoryBuilder:
	"""
	Builder for constructing MemoryEntry objects from samples and representations.

	This class handles the logic for creating memory entries from time series samples
	and their representations (raw, text summary, statistics).
	"""

	def __init__(self, task_type: TaskType | str) -> None:
		if isinstance(task_type, str):
			task_type = TaskType(task_type)
		self.task_type = task_type

	def build_entry(
		self,
		sample: TimeSeriesSample,
		channel_id: int = 0,
		ts_repr: Optional[RepresentationOutput] = None,
		summary_repr: Optional[RepresentationOutput] = None,
		statistic_repr: Optional[RepresentationOutput] = None,
		metadata: Optional[dict[str, Any]] = None,
	) -> MemoryEntry:
		"""
		Build a single memory entry from a sample and its representations.

		Parameters
		----------
		sample : TimeSeriesSample
			The time series sample.
		channel_id : int
			Channel index (for multivariate samples).
		ts_repr : RepresentationOutput, optional
			Raw time series representation.
		summary_repr : RepresentationOutput, optional
			Text summary representation.
		statistic_repr : RepresentationOutput, optional
			Statistical features representation.
		metadata : dict, optional
			Additional metadata to attach to the entry.

		Returns
		-------
		MemoryEntry
			The constructed memory entry.
		"""
		ts_view = self._extract_view(ts_repr)
		summary_view = self._extract_view(summary_repr)
		statistic_view = self._extract_view(statistic_repr)

		entry_metadata = metadata or {}
		if sample.metadata:
			entry_metadata = {**sample.metadata, **entry_metadata}

		return MemoryEntry(
			entry_id=f"{sample.sample_id}__ch{int(channel_id)}",
			sample_id=sample.sample_id,
			channel_id=int(channel_id),
			task_type=self.task_type,
			label=sample.y,
			ts_view=ts_view,
			summary_view=summary_view,
			statistic_view=statistic_view,
			metadata=entry_metadata,
		)

	def build_batch(
		self,
		samples: Iterable[TimeSeriesSample],
		channel_id: int = 0,
		ts_reprs: Optional[list[RepresentationOutput]] = None,
		summary_reprs: Optional[list[RepresentationOutput]] = None,
		statistic_reprs: Optional[list[RepresentationOutput]] = None,
		metadata: Optional[dict[str, Any]] = None,
	) -> list[MemoryEntry]:
		"""
		Build memory entries for a batch of samples.

		Parameters
		----------
		samples : Iterable[TimeSeriesSample]
			The time series samples.
		channel_id : int
			Channel index (for multivariate samples).
		ts_reprs : list[RepresentationOutput], optional
			Raw time series representations (one per sample).
		summary_reprs : list[RepresentationOutput], optional
			Text summary representations (one per sample).
		statistic_reprs : list[RepresentationOutput], optional
			Statistical features representations (one per sample).
		metadata : dict, optional
			Shared metadata to attach to all entries.

		Returns
		-------
		list[MemoryEntry]
			List of constructed memory entries.
		"""
		sample_list = list(samples)
		entries: list[MemoryEntry] = []

		for idx, sample in enumerate(sample_list):
			ts_repr = ts_reprs[idx] if ts_reprs and idx < len(ts_reprs) else None
			summary_repr = summary_reprs[idx] if summary_reprs and idx < len(summary_reprs) else None
			statistic_repr = statistic_reprs[idx] if statistic_reprs and idx < len(statistic_reprs) else None

			entry = self.build_entry(
				sample=sample,
				channel_id=channel_id,
				ts_repr=ts_repr,
				summary_repr=summary_repr,
				statistic_repr=statistic_repr,
				metadata=metadata,
			)
			entries.append(entry)

		return entries

	@staticmethod
	def _extract_view(repr_output: Optional[RepresentationOutput]) -> Any:
		"""Extract the view payload from a RepresentationOutput."""
		if repr_output is None or not repr_output.records:
			return None

		if len(repr_output.records) == 1:
			return repr_output.records[0].payload

		payloads = [record.payload for record in repr_output.records]
		return payloads


class MemoryEntryBuilder:
	"""Fluent builder for constructing a single MemoryEntry."""

	def __init__(self, sample_id: str, channel_id: int = 0) -> None:
		self.sample_id = sample_id
		self.channel_id = channel_id
		self.entry_id: Optional[str] = None
		self.task_type: Optional[TaskType] = None
		self.label: Optional[Any] = None
		self.ts_view: Optional[Any] = None
		self.summary_view: Optional[str] = None
		self.statistic_view: Optional[Any] = None
		self.metadata: dict[str, Any] = {}

	def with_entry_id(self, entry_id: str) -> MemoryEntryBuilder:
		self.entry_id = entry_id
		return self

	def with_task_type(self, task_type: TaskType | str) -> MemoryEntryBuilder:
		if isinstance(task_type, str):
			task_type = TaskType(task_type)
		self.task_type = task_type
		return self

	def with_label(self, label: Any) -> MemoryEntryBuilder:
		self.label = label
		return self

	def with_ts_view(self, ts_view: Any) -> MemoryEntryBuilder:
		self.ts_view = ts_view
		return self

	def with_summary_view(self, summary_view: str) -> MemoryEntryBuilder:
		self.summary_view = summary_view
		return self

	def with_statistic_view(self, statistic_view: Any) -> MemoryEntryBuilder:
		self.statistic_view = statistic_view
		return self

	def with_metadata(self, metadata: dict[str, Any]) -> MemoryEntryBuilder:
		self.metadata = {**self.metadata, **metadata}
		return self

	def build(self) -> MemoryEntry:
		if self.task_type is None:
			raise ValueError("task_type is required")

		entry_id = self.entry_id or f"{self.sample_id}__ch{self.channel_id}"

		return MemoryEntry(
			entry_id=entry_id,
			sample_id=self.sample_id,
			channel_id=self.channel_id,
			task_type=self.task_type,
			label=self.label,
			ts_view=self.ts_view,
			summary_view=self.summary_view,
			statistic_view=self.statistic_view,
			metadata=self.metadata,
		)
