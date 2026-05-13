from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "generate_single_statistical_features": (
        "representations.stat_feature.feature_calculation",
        "generate_single_statistical_features",
    ),
    "generate_statistical_prompts": (
        "representations.stat_feature.feature_calculation",
        "generate_statistical_prompts",
    ),
    "load_statistical_features": (
        "representations.stat_feature.feature_calculation",
        "load_statistical_features",
    ),
    "select_feature_groups": (
        "representations.stat_feature.selector",
        "select_feature_groups",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'representations.stat_feature' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])


__all__ = sorted(_EXPORTS.keys())
