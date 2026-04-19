from __future__ import annotations

import numpy as np


def take_first_n_per_split(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an optional cap for split-level smoke testing."""
    if max_samples is None:
        return X, y
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    return X[:max_samples], y[:max_samples]
