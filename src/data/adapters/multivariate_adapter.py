from __future__ import annotations

from typing import Any

import numpy as np

from core.schemas import TimeSeriesSample


def ensure_multivariate_shape(X: np.ndarray) -> np.ndarray:
    """Ensure input is (N, T, C) for multivariate pipelines."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return X[:, :, None]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected 2D or 3D array, got ndim={X.ndim}")


def array3d_split_to_samples(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    split: str,
) -> list[TimeSeriesSample]:
    """Convert multivariate array split (N, T, C) into TimeSeriesSample list."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, C), got ndim={X.ndim}")

    y = np.asarray(y)
    if int(X.shape[0]) != int(y.shape[0]):
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")

    n, t, c = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    out: list[TimeSeriesSample] = []
    for idx in range(n):
        label: Any = y[idx]
        if hasattr(label, "item"):
            label = label.item()
        out.append(
            TimeSeriesSample(
                sample_id=f"{split}_{idx}",
                x=X[idx],
                y=label,
                metadata={
                    "dataset_name": dataset_name,
                    "split": split,
                    "n_channels": c,
                    "series_length": t,
                },
            )
        )

    return out
