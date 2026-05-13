"""Representations module for feature extraction and caching."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseRepresentation": ("representations.rep_base", "BaseRepresentation"),
    "RepresentationInput": ("representations.schemas", "RepresentationInput"),
    "RepresentationOutput": ("representations.schemas", "RepresentationOutput"),
    "StatisticsRepresentation": ("representations.statistics", "StatisticsRepresentation"),
    "compute_statistics_for_sample": ("representations.statistics", "compute_statistics_for_sample"),
    "compute_statistics_for_batch": ("representations.statistics", "compute_statistics_for_batch"),
    "RawSeriesRepresentation": ("representations.raw_series", "RawSeriesRepresentation"),
    "compute_raw_series_for_sample": ("representations.raw_series", "compute_raw_series_for_sample"),
    "compute_raw_series_for_batch": ("representations.raw_series", "compute_raw_series_for_batch"),
    "TextSummaryRepresentation": ("representations.text_summary", "TextSummaryRepresentation"),
    "compute_summary_for_sample": ("representations.text_summary", "compute_summary_for_sample"),
    "compute_summary_for_batch": ("representations.text_summary", "compute_summary_for_batch"),
    "RepresentationBundle": ("representations.bundle", "RepresentationBundle"),
    "RepresentationBundler": ("representations.bundle", "RepresentationBundler"),
    "normalize_array": ("representations.normalizers", "normalize_array"),
    "normalize_records": ("representations.normalizers", "normalize_records"),
    "scale_array": ("representations.normalizers", "scale_array"),
    "clip_array": ("representations.normalizers", "clip_array"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'representations' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])


__all__ = sorted(_EXPORTS.keys())
