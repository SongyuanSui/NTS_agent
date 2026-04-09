# src/utils/validation.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def validate_non_empty_string(value: Any, name: str) -> str:
    """
    Validate and return a non-empty string.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def validate_positive_int(value: Any, name: str) -> int:
    """
    Validate and return a positive integer.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_non_negative_int(value: Any, name: str) -> int:
    """
    Validate and return a non-negative integer.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def validate_optional_path(value: Any, name: str) -> Path | None:
    """
    Validate a path-like value or None.
    """
    if value is None:
        return None
    return Path(value)


def validate_1d_numeric_array(values: Any, name: str) -> np.ndarray:
    """
    Convert input into a 1D float NumPy array and validate shape.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D numeric array, got ndim={arr.ndim}.")
    return arr


def validate_iterable_not_empty(values: Any, name: str) -> Iterable[Any]:
    """
    Validate that an object is iterable and non-empty.
    """
    if values is None:
        raise ValueError(f"{name} must not be None.")

    try:
        values = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable.") from exc

    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")

    return values