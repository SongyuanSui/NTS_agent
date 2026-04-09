# src/memory/memory_bank.py

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable, Optional

from core.enums import RepresentationType, TaskType
from memory.schemas import MemoryEntry


class MemoryBank:
    """
    In-memory container for retrieval candidates.

    Design goals
    ------------
    - Keep the first version simple and predictable
    - Support basic filtering needed by retrievers
    - Serve as the canonical source for per-sample, per-channel memory entries

    Notes
    -----
    This class intentionally does not implement advanced ANN indexing.
    That can be added later in memory/indexing.py without changing the public API.

    Logging policy
    --------------
    - MemoryBank does not create or configure loggers by itself.
    - A logger can be optionally injected from outside.
    - Logging is used only for lightweight operational reporting, especially
      around summary/statistics reporting.
    """

    def __init__(
        self,
        entries: Optional[Iterable[MemoryEntry]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._entries: list[MemoryEntry] = []
        self._entry_id_to_index: dict[str, int] = {}
        self._logger = logger

        if entries is not None:
            for entry in entries:
                self.add(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        return f"MemoryBank(num_entries={len(self)})"

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        """
        Attach or replace an external logger.
        """
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger or None.")
        self._logger = logger

    def add(self, entry: MemoryEntry) -> None:
        if not isinstance(entry, MemoryEntry):
            raise TypeError(
                f"MemoryBank.add expects MemoryEntry, got {type(entry).__name__}."
            )

        if entry.entry_id in self._entry_id_to_index:
            raise ValueError(f"Duplicated entry_id detected: {entry.entry_id}")

        self._entry_id_to_index[entry.entry_id] = len(self._entries)
        self._entries.append(entry)

    def extend(self, entries: Iterable[MemoryEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def get_by_entry_id(self, entry_id: str) -> MemoryEntry:
        if entry_id not in self._entry_id_to_index:
            raise KeyError(f"entry_id '{entry_id}' not found in MemoryBank.")

        return self._entries[self._entry_id_to_index[entry_id]]

    def get_by_sample_id(self, sample_id: str) -> list[MemoryEntry]:
        return [entry for entry in self._entries if entry.sample_id == sample_id]

    def filter(
        self,
        task_type: Optional[TaskType] = None,
        label: Optional[Any] = None,
        channel_id: Optional[int] = None,
        channel_ids: Optional[list[int]] = None,
        representation_type: Optional[RepresentationType] = None,
        exclude_sample_id: Optional[str] = None,
        exclude_sample_ids: Optional[list[str] | set[str]] = None,
    ) -> list[MemoryEntry]:
        """
        Return entries satisfying all provided conditions.

        Parameters
        ----------
        task_type:
            Keep only entries of this task type.
        label:
            Keep only entries with this label.
        channel_id:
            Keep only entries for a single channel.
        channel_ids:
            Keep only entries whose channel_id is in this collection.
        representation_type:
            Keep only entries that contain the given representation view.
        exclude_sample_id:
            Exclude a single sample id.
        exclude_sample_ids:
            Exclude multiple sample ids.
        """
        if channel_id is not None:
            channel_id = int(channel_id)

        channel_id_set = None
        if channel_ids is not None:
            channel_id_set = {int(x) for x in channel_ids}

        exclude_sample_id_set = None
        if exclude_sample_ids is not None:
            exclude_sample_id_set = {str(x) for x in exclude_sample_ids}

        results: list[MemoryEntry] = []

        for entry in self._entries:
            if task_type is not None and entry.task_type != task_type:
                continue

            if label is not None and entry.label != label:
                continue

            if channel_id is not None and entry.channel_id != channel_id:
                continue

            if channel_id_set is not None and entry.channel_id not in channel_id_set:
                continue

            if exclude_sample_id is not None and entry.sample_id == exclude_sample_id:
                continue

            if exclude_sample_id_set is not None and entry.sample_id in exclude_sample_id_set:
                continue

            if representation_type is not None and not entry.has_view(representation_type):
                continue

            results.append(entry)

        return results

    def filter_by_selected_channels(self, selected_channel_ids: list[int]) -> list[MemoryEntry]:
        """
        Convenience wrapper for filtering entries by selected channel ids.
        """
        return self.filter(channel_ids=selected_channel_ids)

    def group_by_sample(self) -> dict[str, list[MemoryEntry]]:
        grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
        for entry in self._entries:
            grouped[entry.sample_id].append(entry)
        return dict(grouped)

    def group_by_channel(self) -> dict[int, list[MemoryEntry]]:
        grouped: dict[int, list[MemoryEntry]] = defaultdict(list)
        for entry in self._entries:
            grouped[entry.channel_id].append(entry)
        return dict(grouped)

    def summary(self, log: bool = False) -> dict[str, Any]:
        """
        Return a summary of the memory bank.

        Parameters
        ----------
        log:
            If True and a logger is attached, emit the summary to the logger.
        """
        task_counter: dict[str, int] = defaultdict(int)
        channel_counter: dict[int, int] = defaultdict(int)

        num_ts = 0
        num_summary = 0
        num_statistic = 0
        unique_samples: set[str] = set()
        unique_channels: set[int] = set()

        for entry in self._entries:
            task_counter[entry.task_type.value] += 1
            channel_counter[entry.channel_id] += 1
            unique_samples.add(entry.sample_id)
            unique_channels.add(entry.channel_id)

            if entry.ts_view is not None:
                num_ts += 1
            if entry.summary_view is not None:
                num_summary += 1
            if entry.statistic_view is not None:
                num_statistic += 1

        result = {
            "num_entries": len(self._entries),
            "num_unique_samples": len(unique_samples),
            "num_unique_channels": len(unique_channels),
            "task_counts": dict(task_counter),
            "channel_counts": dict(channel_counter),
            "num_ts_views": num_ts,
            "num_summary_views": num_summary,
            "num_statistic_views": num_statistic,
        }

        if log:
            self.log_summary(result)

        return result

    def log_summary(self, summary: Optional[dict[str, Any]] = None) -> None:
        """
        Log a human-readable summary of the memory bank if a logger is attached.

        Parameters
        ----------
        summary:
            Optional precomputed summary dict. If omitted, summary() is called
            internally without recursive logging.
        """
        if self._logger is None:
            return

        if summary is None:
            summary = self.summary(log=False)

        self._logger.info("MemoryBank summary:")
        self._logger.info("  num_entries = %s", summary.get("num_entries"))
        self._logger.info("  num_unique_samples = %s", summary.get("num_unique_samples"))
        self._logger.info("  num_unique_channels = %s", summary.get("num_unique_channels"))
        self._logger.info("  num_ts_views = %s", summary.get("num_ts_views"))
        self._logger.info("  num_summary_views = %s", summary.get("num_summary_views"))
        self._logger.info("  num_statistic_views = %s", summary.get("num_statistic_views"))
        self._logger.info("  task_counts = %s", summary.get("task_counts"))
        self._logger.info("  channel_counts = %s", summary.get("channel_counts"))