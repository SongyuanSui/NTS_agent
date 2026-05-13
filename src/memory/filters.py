from __future__ import annotations

from typing import Any, Callable, Optional

from core.enums import RepresentationType, TaskType
from memory.schemas import MemoryEntry


class MemoryFilter:
	"""Base class for filtering memory entries."""

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		"""Apply the filter to a list of memory entries."""
		raise NotImplementedError


class TaskTypeFilter(MemoryFilter):
	"""Filter entries by task type."""

	def __init__(self, task_type: TaskType | str) -> None:
		if isinstance(task_type, str):
			task_type = TaskType(task_type)
		self.task_type = task_type

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		return [e for e in entries if e.task_type == self.task_type]


class LabelFilter(MemoryFilter):
	"""Filter entries by label."""

	def __init__(self, label: Any) -> None:
		self.label = label

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		return [e for e in entries if e.label == self.label]


class ChannelFilter(MemoryFilter):
	"""Filter entries by channel ID."""

	def __init__(self, channel_ids: int | list[int]) -> None:
		if isinstance(channel_ids, int):
			channel_ids = [channel_ids]
		self.channel_ids = set(int(cid) for cid in channel_ids)

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		return [e for e in entries if e.channel_id in self.channel_ids]


class RepresentationFilter(MemoryFilter):
	"""Filter entries that have a specific representation type."""

	def __init__(self, rep_type: RepresentationType | str) -> None:
		if isinstance(rep_type, str):
			rep_type = RepresentationType(rep_type)
		self.rep_type = rep_type

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		return [e for e in entries if e.has_view(self.rep_type)]


class SampleIdFilter(MemoryFilter):
	"""Filter entries by sample ID (include or exclude)."""

	def __init__(
		self,
		include_ids: Optional[list[str]] = None,
		exclude_ids: Optional[list[str]] = None,
	) -> None:
		self.include_ids = set(include_ids) if include_ids else None
		self.exclude_ids = set(exclude_ids) if exclude_ids else None

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		result = entries

		if self.include_ids is not None:
			result = [e for e in result if e.sample_id in self.include_ids]

		if self.exclude_ids is not None:
			result = [e for e in result if e.sample_id not in self.exclude_ids]

		return result


class CustomFilter(MemoryFilter):
	"""Filter entries using a custom predicate function."""

	def __init__(self, predicate: Callable[[MemoryEntry], bool]) -> None:
		self.predicate = predicate

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		return [e for e in entries if self.predicate(e)]


class CompositeFilter(MemoryFilter):
	"""Combine multiple filters with AND logic (all must pass)."""

	def __init__(self, filters: list[MemoryFilter]) -> None:
		self.filters = filters

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		result = entries
		for filt in self.filters:
			result = filt.apply(result)
		return result

	def add_filter(self, filt: MemoryFilter) -> CompositeFilter:
		self.filters.append(filt)
		return self


class UnionFilter(MemoryFilter):
	"""Combine multiple filters with OR logic (at least one must pass)."""

	def __init__(self, filters: list[MemoryFilter]) -> None:
		self.filters = filters

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		result_sets = [set(filt.apply(entries)) for filt in self.filters]
		if not result_sets:
			return []
		combined = set.union(*result_sets)
		return [e for e in entries if e in combined]

	def add_filter(self, filt: MemoryFilter) -> UnionFilter:
		self.filters.append(filt)
		return self


class LabelSpaceFilter(MemoryFilter):
	"""Filter entries by allowed label space."""

	def __init__(self, allowed_labels: list[Any]) -> None:
		self.allowed_labels = set(allowed_labels)

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		return [e for e in entries if e.label in self.allowed_labels]


class ViewCountFilter(MemoryFilter):
	"""Filter entries by minimum number of representation views."""

	def __init__(self, min_views: int = 1) -> None:
		self.min_views = int(min_views)

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		result = []
		for entry in entries:
			count = sum([
				entry.ts_view is not None,
				entry.summary_view is not None,
				entry.statistic_view is not None,
			])
			if count >= self.min_views:
				result.append(entry)
		return result


class MetadataFilter(MemoryFilter):
	"""Filter entries by metadata key-value pairs."""

	def __init__(self, metadata_filters: dict[str, Any]) -> None:
		self.metadata_filters = metadata_filters

	def apply(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
		result = []
		for entry in entries:
			match = True
			for key, value in self.metadata_filters.items():
				if entry.metadata.get(key) != value:
					match = False
					break
			if match:
				result.append(entry)
		return result


def filter_entries(
	entries: list[MemoryEntry],
	task_type: Optional[TaskType | str] = None,
	labels: Optional[list[Any]] = None,
	channel_ids: Optional[list[int]] = None,
	rep_type: Optional[RepresentationType | str] = None,
	include_sample_ids: Optional[list[str]] = None,
	exclude_sample_ids: Optional[list[str]] = None,
) -> list[MemoryEntry]:
	"""
	Convenience function to filter entries with multiple criteria.

	All specified criteria are combined with AND logic (all must pass).
	"""
	filters: list[MemoryFilter] = []

	if task_type is not None:
		filters.append(TaskTypeFilter(task_type))

	if labels is not None:
		filters.append(LabelSpaceFilter(labels))

	if channel_ids is not None:
		filters.append(ChannelFilter(channel_ids))

	if rep_type is not None:
		filters.append(RepresentationFilter(rep_type))

	if include_sample_ids is not None or exclude_sample_ids is not None:
		filters.append(SampleIdFilter(include_ids=include_sample_ids, exclude_ids=exclude_sample_ids))

	if not filters:
		return entries

	composite = CompositeFilter(filters)
	return composite.apply(entries)
