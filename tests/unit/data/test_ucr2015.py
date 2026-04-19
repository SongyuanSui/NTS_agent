from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.ucr2015 import (
    list_ucr2015_datasets,
    load_ucr2015_as_samples,
    load_ucr2015_local,
    remap_labels_zero_based,
    to_time_series_samples,
)


def _write_ucr_split(file_path: Path, rows: list[list[float]]) -> None:
    lines = [",".join(str(v) for v in row) for row in rows]
    file_path.write_text("\n".join(lines), encoding="utf-8")


def test_load_ucr2015_local_and_list_datasets(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ToyDS"
    ds_dir.mkdir(parents=True)

    _write_ucr_split(ds_dir / "ToyDS_TRAIN", [[1.0, 0.1, 0.2], [2.0, 0.3, 0.4]])
    _write_ucr_split(ds_dir / "ToyDS_TEST", [[2.0, 0.5, 0.6]])

    assert list_ucr2015_datasets(tmp_path) == ["ToyDS"]

    X_train, y_train, X_test, y_test = load_ucr2015_local(tmp_path, "ToyDS")
    assert X_train.shape == (2, 2)
    assert y_train.tolist() == [1.0, 2.0]
    assert X_test.shape == (1, 2)
    assert y_test.tolist() == [2.0]


def test_remap_labels_zero_based() -> None:
    y_train = np.array([5.0, 9.0, 5.0])
    y_test = np.array([9.0, 11.0])

    y_train_m, y_test_m, label_map = remap_labels_zero_based(y_train, y_test)

    assert set(label_map.keys()) == {5.0, 9.0, 11.0}
    assert sorted(label_map.values()) == [0, 1, 2]
    assert y_train_m.dtype == np.int64
    assert y_test_m.dtype == np.int64


def test_to_time_series_samples() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    samples = to_time_series_samples(X, y, dataset_name="ToyDS", split="train")

    assert len(samples) == 2
    assert samples[0].sample_id == "train_0"
    assert samples[1].y == 1
    assert samples[0].metadata["dataset_name"] == "ToyDS"
    assert samples[0].metadata["split"] == "train"


def test_load_ucr2015_as_samples(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ToyDS"
    ds_dir.mkdir(parents=True)

    _write_ucr_split(
        ds_dir / "ToyDS_TRAIN",
        [[1.0, 0.1, 0.2, 0.3], [2.0, 0.4, 0.5, 0.6], [1.0, 0.7, 0.8, 0.9]],
    )
    _write_ucr_split(
        ds_dir / "ToyDS_TEST",
        [[2.0, 1.1, 1.2, 1.3], [3.0, 1.4, 1.5, 1.6]],
    )

    bundle = load_ucr2015_as_samples(
        dataset_name="ToyDS",
        base_dir=tmp_path,
        remap_labels=True,
        max_samples_per_split=2,
    )

    assert bundle["dataset_name"] == "ToyDS"
    assert bundle["metadata"]["n_train"] == 2
    assert bundle["metadata"]["n_test"] == 2
    assert bundle["metadata"]["series_length"] == 3
    assert len(bundle["train_samples"]) == 2
    assert len(bundle["test_samples"]) == 2
    assert bundle["label_map"] is not None


def test_to_time_series_samples_rejects_length_mismatch() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1])
    with pytest.raises(ValueError):
        to_time_series_samples(X, y, dataset_name="ToyDS", split="train")
