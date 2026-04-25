from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from core.enums import TaskType
from data.adapters.univariate_adapter import array_split_to_samples
from data.dataset_base import DatasetLoaderBase
from data.schemas import ClassificationDatasetBundle, DatasetSplit
from data.split import take_first_n_per_split
from data.transforms import remap_labels_zero_based


DEFAULT_UCR2015_DIR = Path(__file__).resolve().parents[3] / "datasets" / "UCR_TS_Archive_2015"


def list_ucr2015_datasets(base_dir: str | Path = DEFAULT_UCR2015_DIR) -> list[str]:
    """List valid UCR2015 dataset names under base_dir."""
    root = Path(base_dir)
    if not root.exists():
        return []

    dataset_names: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        name = child.name
        if (child / f"{name}_TRAIN").exists() and (child / f"{name}_TEST").exists():
            dataset_names.append(name)

    return dataset_names


def load_ucr2015_local(
    base_dir: str | Path,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one local UCR2015 dataset from TRAIN/TEST files."""
    root = Path(base_dir)
    ds_dir = root / dataset_name
    train_path = ds_dir / f"{dataset_name}_TRAIN"
    test_path = ds_dir / f"{dataset_name}_TEST"

    if not train_path.exists():
        raise FileNotFoundError(f"TRAIN file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"TEST file not found: {test_path}")

    train = np.atleast_2d(np.loadtxt(train_path, delimiter=",", dtype=np.float64))
    test = np.atleast_2d(np.loadtxt(test_path, delimiter=",", dtype=np.float64))

    y_train = train[:, 0]
    X_train = train[:, 1:]
    y_test = test[:, 0]
    X_test = test[:, 1:]
    return X_train, y_train, X_test, y_test


class UCR2015ClassificationLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def load(
        self,
        dataset_name: str,
        base_dir: str | Path = DEFAULT_UCR2015_DIR,
        remap_labels: bool = True,
        max_samples_per_split: Optional[int] = None,
    ) -> ClassificationDatasetBundle:
        X_train, y_train, X_test, y_test = load_ucr2015_local(base_dir=base_dir, dataset_name=dataset_name)

        X_train, y_train = take_first_n_per_split(X_train, y_train, max_samples_per_split)
        X_test, y_test = take_first_n_per_split(X_test, y_test, max_samples_per_split)

        label_map = None
        if remap_labels:
            y_train, y_test, label_map = remap_labels_zero_based(y_train, y_test)

        train_samples = array_split_to_samples(X_train, y_train, dataset_name=dataset_name, split="train")
        test_samples = array_split_to_samples(X_test, y_test, dataset_name=dataset_name, split="test")

        return ClassificationDatasetBundle(
            dataset_name=dataset_name,
            train=DatasetSplit(samples=train_samples, split_name="train"),
            test=DatasetSplit(samples=test_samples, split_name="test"),
            label_map=label_map,
            metadata={
                "n_train": len(train_samples),
                "n_test": len(test_samples),
                "series_length": int(X_train.shape[1]) if len(X_train) else 0,
                "base_dir": str(Path(base_dir)),
            },
        )


def load_ucr2015_as_samples(
    dataset_name: str,
    base_dir: str | Path = DEFAULT_UCR2015_DIR,
    remap_labels: bool = True,
    max_samples_per_split: Optional[int] = None,
) -> dict:
    """Compatibility helper returning old dict schema used in existing code."""
    bundle = UCR2015ClassificationLoader().load(
        dataset_name=dataset_name,
        base_dir=base_dir,
        remap_labels=remap_labels,
        max_samples_per_split=max_samples_per_split,
    )

    return {
        "dataset_name": bundle.dataset_name,
        "label_map": bundle.label_map,
        "train_samples": bundle.train.samples,
        "test_samples": bundle.test.samples,
        "metadata": bundle.metadata,
    }
