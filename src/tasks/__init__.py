"""Task definitions and registry helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AnomalyWindowTask": ("tasks.anomaly_window", "AnomalyWindowTask"),
    "ClassificationTask": ("tasks.classification", "ClassificationTask"),
    "PredictionTask": ("tasks.prediction", "PredictionTask"),
    "BaseTask": ("tasks.task_base", "BaseTask"),
    "build_task_from_name": ("tasks.task_registry", "build_task_from_name"),
    "get_task_class": ("tasks.task_registry", "get_task_class"),
    "normalize_label_space": ("tasks.label_space", "normalize_label_space"),
    "parse_prediction_value": ("tasks.output_parsers", "parse_prediction_value"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'tasks' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])


__all__ = sorted(_EXPORTS.keys())
