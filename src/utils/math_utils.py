# src/utils/math_utils.py

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def safe_mean(values: Sequence[float]) -> float:
    """
    Return the mean of a sequence. Return 0.0 for empty input.
    """
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def safe_std(values: Sequence[float], ddof: int = 0) -> float:
    """
    Return the standard deviation of a sequence. Return 0.0 for empty input.
    """
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=ddof))


def euclidean_sq(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Squared Euclidean distance between two equal-length sequences.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length.")
    return float(sum((x - y) ** 2 for x, y in zip(a, b)))


def l2norm(x: Sequence[float]) -> float:
    """
    L2 norm of a vector.
    """
    return float(math.sqrt(sum(v * v for v in x)))


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two vectors.
    Return 0.0 when one side has near-zero norm.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length.")

    na = l2norm(a)
    nb = l2norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)) / (na * nb))


def zscore_list(values: list[float]) -> list[float]:
    """
    Z-score normalize a list.
    If variance is zero, return all zeros.
    """
    if len(values) == 0:
        return []

    mu = safe_mean(values)
    var = sum((x - mu) ** 2 for x in values) / max(1, len(values) - 1)
    sd = math.sqrt(var)

    if sd <= 1e-12:
        return [0.0 for _ in values]

    return [float((x - mu) / sd) for x in values]