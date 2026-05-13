from __future__ import annotations

from typing import Iterable

import numpy as np


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    if true_arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(true_arr - pred_arr)))


def mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    if true_arr.size == 0:
        return 0.0
    return float(np.mean((true_arr - pred_arr) ** 2))


def root_mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    if true_arr.size == 0:
        return 0.0
    denom = np.maximum(np.abs(true_arr), float(eps))
    return float(np.mean(np.abs((true_arr - pred_arr) / denom)))


def _as_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(list(y_true), dtype=float)
    pred_arr = np.asarray(list(y_pred), dtype=float)
    if true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return true_arr, pred_arr
