from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.enums import TaskType
from data.dataset_registry import (
    get_dataset_loader,
    list_dataset_loaders,
    register_dataset_loader,
)
from data.loaders.anomaly_loader import (
    NotImplementedAnomalySequenceLoader,
    NotImplementedAnomalyWindowLoader,
)
from data.loaders.classification_multivariate_loader import (
    UEAMultivariateClassificationLoader,
)
from data.loaders.classification_univariate_loader import UCR2015ClassificationLoader


def test_registry_contains_dataset_loaders_only() -> None:
    names = list_dataset_loaders()
    assert "ucr2015" in names
    assert "uea" in names
    assert "anomaly_sequence" not in names
    assert "anomaly_window" not in names


def test_uea_loader_routes_to_multivariate_loader() -> None:
    loader = get_dataset_loader("uea")
    assert isinstance(loader, UEAMultivariateClassificationLoader)
    assert loader.task_type == TaskType.CLASSIFICATION


def test_anomaly_loader_classes_keep_distinct_task_types() -> None:
    assert NotImplementedAnomalySequenceLoader().task_type == TaskType.ANOMALY_SEQUENCE
    assert NotImplementedAnomalyWindowLoader().task_type == TaskType.ANOMALY_WINDOW


def test_register_dataset_loader_rejects_non_loader() -> None:
    with pytest.raises(TypeError):
        register_dataset_loader("bad_loader", object())  # type: ignore[arg-type]


def test_register_and_fetch_custom_loader() -> None:
    key = "tmp_classification_loader"
    loader = UCR2015ClassificationLoader()
    register_dataset_loader(key, loader)

    out = get_dataset_loader(key)
    assert out is loader
    assert out.task_type == TaskType.CLASSIFICATION
