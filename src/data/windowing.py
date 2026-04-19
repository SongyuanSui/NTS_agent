from __future__ import annotations

import numpy as np


def sliding_window_1d(series: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Create sliding windows from 1D series for future prediction/anomaly tasks."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if len(series) < window_size:
        raise ValueError("series length must be >= window_size")

    windows = []
    for start in range(0, len(series) - window_size + 1, stride):
        windows.append(series[start : start + window_size])
    return np.array(windows, dtype=float)
