from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.loaders.classification_multivariate_loader import (
    UEAMultivariateClassificationLoader,
    load_uea_local,
)


def _write_ts(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _build_header(
    *,
    univariate: str = "false",
    dimensions: str = "2",
    include_dimensions: bool = True,
    equal_length: str = "true",
    series_length: str = "3",
    class_label_line: str = "@classLabel true dog cat",
) -> str:
    lines = [
        "@problemName ToyUEA",
        "@timeStamps false",
        "@missing false",
        f"@univariate {univariate}",
    ]
    if include_dimensions:
        lines.append(f"@dimensions {dimensions}")
    lines.extend(
        [
            f"@equalLength {equal_length}",
            f"@seriesLength {series_length}",
            class_label_line,
            "@data",
        ]
    )
    return "\n".join(lines) + "\n"


def _make_toy_uea_dataset(
    root: Path,
    dataset_name: str,
    train_rows: list[str],
    test_rows: list[str],
    *,
    train_header: str | None = None,
    test_header: str | None = None,
) -> None:
    ds_dir = root / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_h = train_header or _build_header()
    test_h = test_header or train_h

    _write_ts(ds_dir / f"{dataset_name}_TRAIN.ts", train_h + "\n".join(train_rows))
    _write_ts(ds_dir / f"{dataset_name}_TEST.ts", test_h + "\n".join(test_rows))


def test_uea_remap_respects_classlabel_order(tmp_path: Path) -> None:
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="ToyUEA",
        train_rows=[
            "1,2,3:4,5,6:cat",
            "1,1,1:2,2,2:dog",
        ],
        test_rows=[
            "3,3,3:1,1,1:cat",
        ],
    )

    bundle = UEAMultivariateClassificationLoader().load(
        dataset_name="ToyUEA",
        base_dir=tmp_path,
        remap_labels=True,
    )

    assert bundle.label_map == {"dog": 0, "cat": 1}
    assert bundle.train.samples[0].y == 1  # cat
    assert bundle.train.samples[1].y == 0  # dog
    assert bundle.train.samples[0].x.shape == (3, 2)


def test_uea_load_rejects_labels_not_in_declared_classes(tmp_path: Path) -> None:
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="ToyUEA",
        train_rows=[
            "1,2,3:4,5,6:dog",
        ],
        test_rows=[
            "3,3,3:1,1,1:bird",
        ],
    )

    with pytest.raises(ValueError, match="Observed labels not"):
        load_uea_local(base_dir=tmp_path, dataset_name="ToyUEA")


def test_uea_load_rejects_missing_required_header_field(tmp_path: Path) -> None:
    ds_dir = tmp_path / "BrokenUEA"
    ds_dir.mkdir(parents=True, exist_ok=True)

    broken_header = _build_header(include_dimensions=False)
    _write_ts(ds_dir / "BrokenUEA_TRAIN.ts", broken_header)
    _write_ts(ds_dir / "BrokenUEA_TEST.ts", broken_header)

    with pytest.raises(ValueError, match="Missing required header fields"):
        load_uea_local(base_dir=tmp_path, dataset_name="BrokenUEA")


def test_uea_load_rejects_classlabel_true_without_values(tmp_path: Path) -> None:
    bad_header = _build_header(class_label_line="@classLabel true")
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="BadHeader3",
        train_rows=["1,2,3:4,5,6:dog"],
        test_rows=["1,2,3:4,5,6:dog"],
        train_header=bad_header,
        test_header=bad_header,
    )

    with pytest.raises(ValueError, match="requires explicit class values"):
        load_uea_local(base_dir=tmp_path, dataset_name="BadHeader3")


def test_uea_load_rejects_non_numeric_values(tmp_path: Path) -> None:
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="BadRow2",
        train_rows=["1,2,a:4,5,6:dog"],
        test_rows=["1,2,3:4,5,6:cat"],
    )

    with pytest.raises(ValueError, match="non-numeric"):
        load_uea_local(base_dir=tmp_path, dataset_name="BadRow2")


def test_uea_load_rejects_nan_or_inf_values(tmp_path: Path) -> None:
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="BadRow4",
        train_rows=["1,2,nan:4,5,6:dog"],
        test_rows=["1,2,3:4,5,inf:cat"],
    )

    with pytest.raises(ValueError, match="NaN/Inf"):
        load_uea_local(base_dir=tmp_path, dataset_name="BadRow4")


def test_uea_load_rejects_series_length_mismatch(tmp_path: Path) -> None:
    bad_header = _build_header(series_length="2")
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="BadSeriesLength",
        train_rows=["1,2,3:4,5,6:dog"],
        test_rows=["1,2,3:4,5,6:cat"],
        train_header=bad_header,
        test_header=bad_header,
    )

    with pytest.raises(ValueError, match="seriesLength mismatch"):
        load_uea_local(base_dir=tmp_path, dataset_name="BadSeriesLength")


def test_uea_load_rejects_train_test_shape_mismatch(tmp_path: Path) -> None:
    train_header = _build_header(dimensions="2")
    test_header = _build_header(dimensions="3")
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="BadShape",
        train_rows=["1,2,3:4,5,6:dog"],
        test_rows=["1,2,3:4,5,6:7,8,9:dog"],
        train_header=train_header,
        test_header=test_header,
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        load_uea_local(base_dir=tmp_path, dataset_name="BadShape")


def test_uea_load_rejects_train_test_declared_class_mismatch(tmp_path: Path) -> None:
    train_header = _build_header(class_label_line="@classLabel true dog cat")
    test_header = _build_header(class_label_line="@classLabel true dog bird")
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="BadClassDecl",
        train_rows=["1,2,3:4,5,6:dog"],
        test_rows=["1,2,3:4,5,6:dog"],
        train_header=train_header,
        test_header=test_header,
    )

    with pytest.raises(ValueError, match="class label declaration mismatch"):
        load_uea_local(base_dir=tmp_path, dataset_name="BadClassDecl")


def test_uea_numeric_labels_use_fast_remap_path(tmp_path: Path) -> None:
    numeric_header = _build_header(class_label_line="@classLabel true 10 20")
    _make_toy_uea_dataset(
        root=tmp_path,
        dataset_name="NumericUEA",
        train_rows=["1,2,3:4,5,6:20", "1,1,1:2,2,2:10"],
        test_rows=["3,3,3:1,1,1:20"],
        train_header=numeric_header,
        test_header=numeric_header,
    )

    bundle = UEAMultivariateClassificationLoader().load(
        dataset_name="NumericUEA",
        base_dir=tmp_path,
        remap_labels=True,
    )

    assert bundle.label_map == {10.0: 0, 20.0: 1}
    assert bundle.train.samples[0].y == 1
    assert bundle.train.samples[1].y == 0
    train_labels = np.array([sample.y for sample in bundle.train.samples])
    assert np.issubdtype(train_labels.dtype, np.integer)
