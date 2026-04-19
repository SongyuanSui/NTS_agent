from __future__ import annotations

import numpy as np


def remap_labels_zero_based(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[float, int]]:
    """Map labels to contiguous integers 0..C-1 using classes from train+test."""
    classes = np.unique(np.concatenate([y_train, y_test]))
    label_map = {float(label): idx for idx, label in enumerate(classes)}

    y_train_mapped = np.array([label_map[float(label)] for label in y_train], dtype=np.int64)
    y_test_mapped = np.array([label_map[float(label)] for label in y_test], dtype=np.int64)
    return y_train_mapped, y_test_mapped, label_map
