from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from core.enums import TaskType
from data.adapters.multivariate_adapter import array3d_split_to_samples
from data.dataset_base import DatasetLoaderBase
from data.schemas import ClassificationDatasetBundle, DatasetSplit
from data.split import take_first_n_per_split
from data.transforms import remap_labels_zero_based


DEFAULT_UEA_DIR = Path(__file__).resolve().parents[3] / "datasets" / "Multivariate_ts"


def list_uea_datasets(base_dir: str | Path = DEFAULT_UEA_DIR) -> list[str]:
    """List valid UEA multivariate dataset names under base_dir."""
    root = Path(base_dir)
    if not root.exists():
        return []

    dataset_names: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        name = child.name
        train_path = child / f"{name}_TRAIN.ts"
        test_path = child / f"{name}_TEST.ts"
        if train_path.exists() and test_path.exists():
            dataset_names.append(name)

    return dataset_names


def _normalize_header_key(key: str) -> str:
    return key.strip().lower()


def _parse_bool_token(token: str, field_name: str) -> bool:
    value = token.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Invalid boolean for {field_name}: {token}")


def _parse_header(lines: list[str]) -> tuple[dict[str, Any], int]:
    """Parse UEA .ts header and return (header_dict, data_start_index)."""
    header: dict[str, Any] = {
        "problem_name": None,
        "timestamps": None,
        "missing": None,
        "univariate": None,
        "dimensions": None,
        "equal_length": None,
        "series_length": None,
        "class_label": None,
        "class_values": [],
    }

    data_start = -1

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("@"):
            continue

        parts = line.split()
        tag = _normalize_header_key(parts[0])

        if tag == "@data":
            data_start = idx + 1
            break

        if tag == "@problemname":
            if len(parts) >= 2:
                header["problem_name"] = " ".join(parts[1:])
            continue

        if tag == "@timestamps":
            if len(parts) < 2:
                raise ValueError("@timeStamps requires true/false")
            header["timestamps"] = _parse_bool_token(parts[1], "@timeStamps")
            continue

        if tag == "@missing":
            if len(parts) < 2:
                raise ValueError("@missing requires true/false")
            header["missing"] = _parse_bool_token(parts[1], "@missing")
            continue

        if tag == "@univariate":
            if len(parts) < 2:
                raise ValueError("@univariate requires true/false")
            header["univariate"] = _parse_bool_token(parts[1], "@univariate")
            continue

        if tag == "@dimensions":
            if len(parts) < 2:
                raise ValueError("@dimensions requires integer value")
            header["dimensions"] = int(parts[1])
            continue

        if tag == "@equallength":
            if len(parts) < 2:
                raise ValueError("@equalLength requires true/false")
            header["equal_length"] = _parse_bool_token(parts[1], "@equalLength")
            continue

        if tag == "@serieslength":
            if len(parts) < 2:
                raise ValueError("@seriesLength requires integer value")
            header["series_length"] = int(parts[1])
            continue

        if tag == "@classlabel":
            if len(parts) < 2:
                raise ValueError("@classLabel requires true/false")
            has_class = _parse_bool_token(parts[1], "@classLabel")
            header["class_label"] = has_class
            if has_class and len(parts) > 2:
                header["class_values"] = parts[2:]
            continue

    if data_start < 0:
        raise ValueError("Missing @data section in .ts file")

    required_fields = ("dimensions", "equal_length", "class_label", "univariate")
    missing = [field for field in required_fields if header.get(field) is None]
    if missing:
        raise ValueError(f"Missing required header fields: {', '.join(missing)}")

    dims = int(header["dimensions"])
    if dims <= 0:
        raise ValueError("@dimensions must be a positive integer")

    is_univariate = bool(header["univariate"])
    if is_univariate and dims != 1:
        raise ValueError("Header mismatch: @univariate true but @dimensions != 1")
    if not is_univariate and dims < 2:
        raise ValueError("Header mismatch: @univariate false but @dimensions < 2")

    class_values = [str(v) for v in header.get("class_values", [])]
    if bool(header["class_label"]):
        if not class_values:
            raise ValueError("@classLabel true requires explicit class values")
        if len(set(class_values)) != len(class_values):
            raise ValueError("@classLabel values contain duplicates")

    return header, data_start


def _parse_data_line(
    line: str,
    expected_dims: int | None,
    has_class_label: bool,
) -> tuple[np.ndarray, Any]:
    """
    Parse one UEA .ts data row into:
    - sample array with shape (T, C)
    - label value
    """
    fields = line.strip().split(":")
    if has_class_label:
        if len(fields) < 2:
            raise ValueError("Malformed row: expected dimensions and class label")
        label = fields[-1].strip()
        if not label:
            raise ValueError("Malformed row: empty class label")
        dim_fields = fields[:-1]
    else:
        label = None
        dim_fields = fields

    if expected_dims is not None and len(dim_fields) != expected_dims:
        raise ValueError(
            f"Dimension mismatch: expected {expected_dims}, got {len(dim_fields)}"
        )

    channels: list[np.ndarray] = []
    series_length: Optional[int] = None
    for dim in dim_fields:
        tokens = [tok.strip() for tok in dim.split(",") if tok.strip()]
        if not tokens:
            raise ValueError("Malformed row: empty channel sequence")

        try:
            values = np.asarray([float(tok) for tok in tokens], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Malformed row: non-numeric value in channel '{dim}'") from exc

        if not np.isfinite(values).all():
            raise ValueError("Malformed row: NaN/Inf detected in channel values")

        if series_length is None:
            series_length = int(values.shape[0])
        elif int(values.shape[0]) != series_length:
            raise ValueError("Unequal channel lengths within one sample")

        channels.append(values)

    sample_tc = np.stack(channels, axis=0).T
    return sample_tc, label


def _parse_ts_file(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f".ts file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    header, data_start = _parse_header(lines)

    has_class_label = bool(header.get("class_label", False))
    if not has_class_label:
        raise ValueError(f"Classification loader requires @classLabel true in file: {path}")

    expected_dims = header.get("dimensions")
    equal_length = header.get("equal_length")
    declared_series_len = header.get("series_length")

    if equal_length is False:
        raise ValueError("Variable-length datasets are not supported in v1")

    samples: list[np.ndarray] = []
    labels: list[Any] = []
    observed_series_len: Optional[int] = None
    observed_dims: Optional[int] = None

    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        sample_tc, label = _parse_data_line(
            line=line,
            expected_dims=expected_dims,
            has_class_label=has_class_label,
        )

        t_len, c_dim = int(sample_tc.shape[0]), int(sample_tc.shape[1])

        if observed_series_len is None:
            observed_series_len = t_len
        elif t_len != observed_series_len:
            raise ValueError(f"Unequal series length across samples in file: {path}")

        if observed_dims is None:
            observed_dims = c_dim
        elif c_dim != observed_dims:
            raise ValueError(f"Unequal dimensions across samples in file: {path}")

        if declared_series_len is not None and t_len != int(declared_series_len):
            raise ValueError(
                f"seriesLength mismatch in {path}: declared {declared_series_len}, observed {t_len}"
            )

        samples.append(sample_tc)
        labels.append(label)

    if not samples:
        raise ValueError(f"No sample rows found after @data in file: {path}")

    X = np.stack(samples, axis=0)
    y = np.asarray(labels, dtype=object)

    if not np.isfinite(X).all():
        raise ValueError(f"NaN/Inf detected in parsed sample matrix: {path}")

    declared_class_values = [str(v) for v in header.get("class_values", [])]
    if declared_class_values:
        observed = {str(v) for v in y.tolist()}
        unknown = sorted(observed.difference(declared_class_values))
        if unknown:
            raise ValueError(
                f"Observed labels not declared by @classLabel in {path}: {unknown}"
            )

    meta = dict(header)
    meta["n_samples"] = int(X.shape[0])
    meta["series_length_observed"] = int(X.shape[1])
    meta["n_channels_observed"] = int(X.shape[2])

    return X, y, meta


def load_uea_local(
    base_dir: str | Path,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one UEA multivariate dataset from TRAIN/TEST .ts files."""
    root = Path(base_dir)
    ds_dir = root / dataset_name
    train_path = ds_dir / f"{dataset_name}_TRAIN.ts"
    test_path = ds_dir / f"{dataset_name}_TEST.ts"

    if not train_path.exists():
        raise FileNotFoundError(f"TRAIN file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"TEST file not found: {test_path}")

    X_train, y_train, train_header = _parse_ts_file(train_path)
    X_test, y_test, test_header = _parse_ts_file(test_path)

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "TRAIN/TEST shape mismatch on (T, C): "
            f"{X_train.shape[1:]} vs {X_test.shape[1:]}"
        )

    train_classes = [str(v) for v in train_header.get("class_values", [])]
    test_classes = [str(v) for v in test_header.get("class_values", [])]
    if train_classes and test_classes and train_classes != test_classes:
        raise ValueError(
            "TRAIN/TEST class label declaration mismatch: "
            f"{train_classes} vs {test_classes}"
        )

    if train_classes:
        observed = {str(v) for v in np.concatenate([y_train, y_test]).tolist()}
        unknown = sorted(observed.difference(train_classes))
        if unknown:
            raise ValueError(
                "Observed labels not present in declared class values: "
                f"{unknown}"
            )

    metadata: dict[str, Any] = {
        "train_header": train_header,
        "test_header": test_header,
        "series_length": int(X_train.shape[1]),
        "n_channels": int(X_train.shape[2]),
        "class_values": train_classes or test_classes,
    }

    return X_train, y_train, X_test, y_test, metadata


def _remap_labels_zero_based_generic(
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_values: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[Any, int]]:
    """Remap arbitrary labels (including strings) to contiguous 0..C-1."""
    observed_labels = [str(v) for v in np.concatenate([y_train, y_test]).tolist()]

    ordered_classes: list[str] = []
    if class_values:
        ordered_classes = [str(v) for v in class_values]
        unknown = sorted(set(observed_labels).difference(ordered_classes))
        if unknown:
            raise ValueError(
                "Observed labels not present in class_values order: "
                f"{unknown}"
            )
    else:
        seen: set[str] = set()
        for label in observed_labels:
            if label not in seen:
                seen.add(label)
                ordered_classes.append(label)

    label_map: dict[Any, int] = {
        label: idx for idx, label in enumerate(ordered_classes)
    }

    y_train_m = np.array([label_map[str(label)] for label in y_train], dtype=np.int64)
    y_test_m = np.array([label_map[str(label)] for label in y_test], dtype=np.int64)
    return y_train_m, y_test_m, label_map


class UEAMultivariateClassificationLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def load(
        self,
        dataset_name: str,
        base_dir: str | Path = DEFAULT_UEA_DIR,
        remap_labels: bool = True,
        max_samples_per_split: Optional[int] = None,
    ) -> ClassificationDatasetBundle:
        X_train, y_train, X_test, y_test, uea_meta = load_uea_local(
            base_dir=base_dir,
            dataset_name=dataset_name,
        )

        X_train, y_train = take_first_n_per_split(X_train, y_train, max_samples_per_split)
        X_test, y_test = take_first_n_per_split(X_test, y_test, max_samples_per_split)

        label_map = None
        if remap_labels:
            # Keep existing fast path for numeric labels, otherwise use generic mapping.
            try:
                y_train_num = y_train.astype(np.float64)
                y_test_num = y_test.astype(np.float64)
                y_train, y_test, label_map = remap_labels_zero_based(y_train_num, y_test_num)
            except (ValueError, TypeError):
                y_train, y_test, label_map = _remap_labels_zero_based_generic(
                    y_train,
                    y_test,
                    class_values=uea_meta.get("class_values"),
                )

        train_samples = array3d_split_to_samples(X_train, y_train, dataset_name=dataset_name, split="train")
        test_samples = array3d_split_to_samples(X_test, y_test, dataset_name=dataset_name, split="test")

        return ClassificationDatasetBundle(
            dataset_name=dataset_name,
            train=DatasetSplit(samples=train_samples, split_name="train"),
            test=DatasetSplit(samples=test_samples, split_name="test"),
            label_map=label_map,
            metadata={
                "n_train": len(train_samples),
                "n_test": len(test_samples),
                "series_length": int(X_train.shape[1]) if len(train_samples) else 0,
                "n_channels": int(X_train.shape[2]) if len(train_samples) else 0,
                "base_dir": str(Path(base_dir)),
                "uea_header": uea_meta,
            },
        )
