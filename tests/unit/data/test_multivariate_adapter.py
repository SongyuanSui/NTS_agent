from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.adapters.multivariate_adapter import array3d_split_to_samples


def test_array3d_split_to_samples_success() -> None:
    X = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    y = np.array([0, 1])

    samples = array3d_split_to_samples(X, y, dataset_name="ToyUEA", split="train")

    assert len(samples) == 2
    assert samples[0].sample_id == "train_0"
    assert samples[0].x.shape == (2, 2)
    assert samples[0].y == 0
    assert samples[0].metadata["dataset_name"] == "ToyUEA"
    assert samples[0].metadata["n_channels"] == 2


def test_array3d_split_to_samples_rejects_ndim_not_3() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="Expected X shape"):
        array3d_split_to_samples(X, y, dataset_name="ToyUEA", split="train")


def test_array3d_split_to_samples_rejects_length_mismatch() -> None:
    X = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    y = np.array([0])

    with pytest.raises(ValueError, match="X/y length mismatch"):
        array3d_split_to_samples(X, y, dataset_name="ToyUEA", split="train")
