from __future__ import annotations

import numpy as np

from core.schemas import TimeSeriesSample


def collate_samples_to_array(samples: list[TimeSeriesSample]) -> tuple[np.ndarray, list]:
    """Collate sample list into (X, y_list) for downstream batch processing."""
    if not samples:
        raise ValueError("samples must be non-empty")

    X = np.array([sample.x for sample in samples], dtype=float)
    y = [sample.y for sample in samples]
    return X, y
