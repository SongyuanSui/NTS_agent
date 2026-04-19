from __future__ import annotations

from typing import Optional

import numpy as np

from core.schemas import TimeSeriesSample


def array_split_to_samples(
    X: np.ndarray,
    y: Optional[np.ndarray],
    dataset_name: str,
    split: str,
) -> list[TimeSeriesSample]:
    """Convert univariate array split (N, T) into TimeSeriesSample list."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array with shape (n_samples, series_len), got ndim={X.ndim}")

    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("y length must match X number of samples")

    samples: list[TimeSeriesSample] = []
    for idx in range(X.shape[0]):
        label = None if y is None else y[idx]
        samples.append(
            TimeSeriesSample(
                sample_id=f"{split}_{idx}",
                x=X[idx],
                y=label,
                metadata={
                    "dataset_name": dataset_name,
                    "split": split,
                    "original_index": idx,
                },
            )
        )
    return samples
