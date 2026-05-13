"""Memory module for storing and retrieving time series representations."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "MemoryEntry": ("memory.schemas", "MemoryEntry"),
    "MemoryBank": ("memory.memory_bank", "MemoryBank"),
    "MemoryBuilder": ("memory.memory_builder", "MemoryBuilder"),
    "MemoryEntryBuilder": ("memory.memory_builder", "MemoryEntryBuilder"),
    "MemoryFilter": ("memory.filters", "MemoryFilter"),
    "TaskTypeFilter": ("memory.filters", "TaskTypeFilter"),
    "LabelFilter": ("memory.filters", "LabelFilter"),
    "ChannelFilter": ("memory.filters", "ChannelFilter"),
    "RepresentationFilter": ("memory.filters", "RepresentationFilter"),
    "SampleIdFilter": ("memory.filters", "SampleIdFilter"),
    "CustomFilter": ("memory.filters", "CustomFilter"),
    "CompositeFilter": ("memory.filters", "CompositeFilter"),
    "UnionFilter": ("memory.filters", "UnionFilter"),
    "LabelSpaceFilter": ("memory.filters", "LabelSpaceFilter"),
    "ViewCountFilter": ("memory.filters", "ViewCountFilter"),
    "MetadataFilter": ("memory.filters", "MetadataFilter"),
    "filter_entries": ("memory.filters", "filter_entries"),
    "ensure_run_dir": ("memory.artifacts", "ensure_run_dir"),
    "get_memory_root": ("memory.artifacts", "get_memory_root"),
    "get_memory_run_dir": ("memory.artifacts", "get_memory_run_dir"),
    "get_logs_dir": ("memory.artifacts", "get_logs_dir"),
    "get_selected_channels_path": ("memory.artifacts", "get_selected_channels_path"),
    "get_memory_bank_path": ("memory.artifacts", "get_memory_bank_path"),
    "get_index_ts_path": ("memory.artifacts", "get_index_ts_path"),
    "get_index_text_path": ("memory.artifacts", "get_index_text_path"),
    "get_index_stat_path": ("memory.artifacts", "get_index_stat_path"),
    "get_build_meta_path": ("memory.artifacts", "get_build_meta_path"),
    "infer_dataset_name": ("memory.artifacts", "infer_dataset_name"),
    "normalize_experiment_name": ("memory.artifacts", "normalize_experiment_name"),
    "make_memory_run_name": ("memory.artifacts", "make_memory_run_name"),
    "load_selected_channels": ("memory.artifacts", "load_selected_channels"),
    "save_selected_channels": ("memory.artifacts", "save_selected_channels"),
    "save_build_meta": ("memory.artifacts", "save_build_meta"),
    "resolve_run_dir_from_dataset_path": ("memory.artifacts", "resolve_run_dir_from_dataset_path"),
    "save_memory_bank_jsonl": ("memory.memory_store", "save_memory_bank_jsonl"),
    "load_memory_bank_jsonl": ("memory.memory_store", "load_memory_bank_jsonl"),
    "save_memory_bank_pickle": ("memory.memory_store", "save_memory_bank_pickle"),
    "load_memory_bank_pickle": ("memory.memory_store", "load_memory_bank_pickle"),
    "resolve_default_memory_bank_path": ("memory.memory_store", "resolve_default_memory_bank_path"),
    "build_stat_index": ("memory.indexing", "build_stat_index"),
    "save_stat_index": ("memory.indexing", "save_stat_index"),
    "load_stat_index": ("memory.indexing", "load_stat_index"),
    "resolve_default_stat_index_path": ("memory.indexing", "resolve_default_stat_index_path"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'memory' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])


__all__ = sorted(_EXPORTS.keys())
