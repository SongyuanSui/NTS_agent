from __future__ import annotations

import numpy as np


def ensure_multivariate_shape(X: np.ndarray) -> np.ndarray:
    """Ensure input is (N, T, C) for multivariate pipelines."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return X[:, :, None]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected 2D or 3D array, got ndim={X.ndim}")
