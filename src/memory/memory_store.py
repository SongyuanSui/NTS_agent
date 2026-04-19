from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from core.constants import DEFAULT_MEMORY_BANK_FILENAME
from core.enums import TaskType
from memory.memory_bank import MemoryBank
from memory.schemas import MemoryEntry
from utils.io import read_jsonl, write_jsonl
from utils.json_utils import to_jsonable


def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "sample_id": entry.sample_id,
        "channel_id": int(entry.channel_id),
        "task_type": entry.task_type.value,
        "label": entry.label,
        "statistic_view": entry.statistic_view,
        "ts_view": entry.ts_view,
        "summary_view": entry.summary_view,
        "metadata": dict(entry.metadata),
    }


def _entry_from_dict(data: dict[str, Any]) -> MemoryEntry:
    task_type = data.get("task_type")
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    return MemoryEntry(
        entry_id=str(data["entry_id"]),
        sample_id=str(data["sample_id"]),
        channel_id=int(data["channel_id"]),
        task_type=task_type,
        label=data.get("label"),
        statistic_view=data.get("statistic_view"),
        ts_view=data.get("ts_view"),
        summary_view=data.get("summary_view"),
        metadata=dict(data.get("metadata", {})),
    )


def save_memory_bank_jsonl(
    memory_bank: MemoryBank,
    path: str | Path,
) -> Path:
    if not isinstance(memory_bank, MemoryBank):
        raise TypeError("memory_bank must be a MemoryBank")

    records = [to_jsonable(_entry_to_dict(entry)) for entry in memory_bank.get_all()]
    return write_jsonl(path, records, ensure_ascii=False)


def load_memory_bank_jsonl(path: str | Path) -> MemoryBank:
    records = read_jsonl(path)
    entries = [_entry_from_dict(dict(record)) for record in records]
    return MemoryBank(entries=entries)


def save_memory_bank_pickle(
    memory_bank: MemoryBank,
    path: str | Path,
) -> Path:
    if not isinstance(memory_bank, MemoryBank):
        raise TypeError("memory_bank must be a MemoryBank")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [to_jsonable(_entry_to_dict(entry)) for entry in memory_bank.get_all()]
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load_memory_bank_pickle(path: str | Path) -> MemoryBank:
    with open(path, "rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list):
        raise TypeError("pickle content must be a list of memory-entry dicts")

    entries = [_entry_from_dict(dict(record)) for record in records]
    return MemoryBank(entries=entries)


def resolve_default_memory_bank_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / DEFAULT_MEMORY_BANK_FILENAME
