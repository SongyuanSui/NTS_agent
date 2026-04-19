from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from core.constants import DEFAULT_STAT_INDEX_FILENAME
from memory.memory_bank import MemoryBank


def _vector_from_stat_payload(
    payload: Any,
    feature_order: list[str] | None,
) -> tuple[np.ndarray, list[str] | None]:
    if payload is None:
        raise ValueError("statistic_view is missing")

    if isinstance(payload, dict):
        if feature_order is None:
            feature_order = sorted(str(k) for k in payload.keys())

        values = []
        for key in feature_order:
            if key not in payload:
                raise ValueError(f"Missing feature '{key}' in statistic_view")
            values.append(float(payload[key]))

        return np.asarray(values, dtype=float), feature_order

    arr = np.asarray(payload, dtype=float).reshape(-1)
    return arr, feature_order


def build_stat_index(memory_bank: MemoryBank) -> dict[str, Any]:
    if not isinstance(memory_bank, MemoryBank):
        raise TypeError("memory_bank must be a MemoryBank")

    entries = [entry for entry in memory_bank.get_all() if entry.statistic_view is not None]
    if len(entries) == 0:
        raise ValueError("No entries with statistic_view found in memory_bank")

    feature_order: list[str] | None = None
    vectors: list[np.ndarray] = []
    sample_ids: list[str] = []
    labels: list[Any] = []
    channel_ids: list[int] = []
    entry_ids: list[str] = []

    for entry in entries:
        vec, feature_order = _vector_from_stat_payload(entry.statistic_view, feature_order)
        vectors.append(vec)
        sample_ids.append(entry.sample_id)
        labels.append(entry.label)
        channel_ids.append(int(entry.channel_id))
        entry_ids.append(entry.entry_id)

    vector_matrix = np.vstack(vectors)

    return {
        "index_type": "stat",
        "feature_order": feature_order,
        "vectors": vector_matrix,
        "sample_ids": sample_ids,
        "labels": labels,
        "channel_ids": channel_ids,
        "entry_ids": entry_ids,
        "num_entries": len(entries),
        "dim": int(vector_matrix.shape[1]),
    }


def save_stat_index(index: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    return path


def load_stat_index(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        index = pickle.load(f)

    if not isinstance(index, dict):
        raise TypeError("index file must contain a dict")
    return index


def resolve_default_stat_index_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / DEFAULT_STAT_INDEX_FILENAME
