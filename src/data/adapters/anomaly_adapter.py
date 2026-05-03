from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.schemas import TimeSeriesSample


def _pick_feature_columns(
    columns: list[str],
    label_col: str,
    drop_columns: Optional[list[str]] = None,
) -> list[str]:
    drop = set(drop_columns or [])
    drop.add(label_col)
    feature_cols = [column for column in columns if column not in drop]
    if not feature_cols:
        raise ValueError("No feature columns left after dropping label/time columns.")
    return feature_cols


@dataclass(slots=True)
class AnomalySequenceArtifact:
    """Intermediate adapter artifact for one raw anomaly sequence."""

    sample_id: str
    x: np.ndarray
    point_labels: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim != 2 or self.x.shape[0] <= 0:
            raise ValueError("x must have shape (T, C) with T > 0.")

        if self.point_labels is not None:
            self.point_labels = np.asarray(self.point_labels, dtype=int)
            if self.point_labels.ndim != 1:
                raise ValueError("point_labels must be a 1D array.")
            if self.point_labels.shape[0] != self.x.shape[0]:
                raise ValueError("point_labels length must match x length.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def length(self) -> int:
        return int(self.x.shape[0])

    def aggregate_label(self, rule: str = "any", ratio_threshold: float = 0.1) -> int:
        if self.point_labels is None:
            raise ValueError("point_labels are required to aggregate a sequence label.")

        positive = self.point_labels > 0
        if rule == "any":
            return int(np.any(positive))
        if rule == "all":
            return int(np.all(positive))
        if rule == "ratio":
            return int(float(np.mean(positive)) >= float(ratio_threshold))
        raise ValueError(f"Unsupported rule '{rule}'. Use any/all/ratio.")

    def to_sequence_sample(self, rule: str = "any", ratio_threshold: float = 0.1) -> TimeSeriesSample:
        label = None if self.point_labels is None else self.aggregate_label(rule=rule, ratio_threshold=ratio_threshold)
        return TimeSeriesSample(
            sample_id=self.sample_id,
            x=self.x,
            y=label,
            metadata={
                **self.metadata,
                "label_type": "sequence",
                "aggregation_rule": rule if self.point_labels is not None else None,
                "ratio_threshold": ratio_threshold if rule == "ratio" else None,
            },
        )

    def to_window_samples(
        self,
        window_size: int,
        stride: int = 1,
        rule: str = "any",
        ratio_threshold: float = 0.1,
    ) -> list[TimeSeriesSample]:
        if self.point_labels is None:
            raise ValueError("point_labels are required to build anomaly windows.")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive.")
        if self.length < window_size:
            return []

        windows: list[TimeSeriesSample] = []
        for start in range(0, self.length - window_size + 1, stride):
            end = start + window_size
            window_labels = self.point_labels[start:end]
            label = AnomalySequenceArtifact(
                sample_id=self.sample_id,
                x=self.x[start:end],
                point_labels=window_labels,
                metadata=self.metadata,
            ).aggregate_label(rule=rule, ratio_threshold=ratio_threshold)
            windows.append(
                TimeSeriesSample(
                    sample_id=f"{self.sample_id}__w{start}_{end}",
                    x=self.x[start:end],
                    y=label,
                    metadata={
                        **self.metadata,
                        "label_type": "window",
                        "window_start": start,
                        "window_end": end,
                        "window_size": window_size,
                        "stride": stride,
                        "window_rule": rule,
                        "ratio_threshold": ratio_threshold if rule == "ratio" else None,
                    },
                )
            )

        return windows


def load_anomaly_sequence_artifact_from_csv(
    csv_path: str | Path,
    dataset_name: str,
    label_col: str = "anomaly",
    time_col: Optional[str] = "datetime",
    drop_columns: Optional[list[str]] = None,
    root_dir: Optional[str | Path] = None,
) -> AnomalySequenceArtifact:
    """Load one CSV file into an AnomalySequenceArtifact.
    
    Args:
        csv_path: Path to the CSV file
        dataset_name: Name of the dataset
        label_col: Column name for the anomaly labels
        time_col: Column name for time/datetime (will be dropped)
        drop_columns: Additional columns to drop
        root_dir: Root directory for computing relative paths in sample_id.
                  If provided, sample_id will be relative path without extension, with / replaced by __.
                  E.g., root_dir=/skab and csv_path=/skab/valve1/1.csv → sample_id="valve1__1"
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=";")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}' in {csv_path}")

    auto_drop = list(drop_columns or [])
    if time_col and time_col in df.columns:
        auto_drop.append(time_col)
    if "changepoint" in df.columns:
        auto_drop.append("changepoint")

    feature_cols = _pick_feature_columns(
        columns=list(df.columns),
        label_col=label_col,
        drop_columns=auto_drop,
    )

    x = df[feature_cols].to_numpy(dtype=float)
    point_labels = df[label_col].to_numpy(dtype=int)

    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Invalid feature shape from {csv_path}: {x.shape}")
    if point_labels.shape[0] != x.shape[0]:
        raise ValueError(f"Length mismatch x/y in {csv_path}: {x.shape[0]} vs {point_labels.shape[0]}")

    # Generate unique sample_id using relative path if root_dir provided
    if root_dir is not None:
        relative_path = csv_path.relative_to(Path(root_dir))
        sample_id = str(relative_path.with_suffix(""))  # Remove .csv extension
        sample_id = sample_id.replace("\\", "/")  # Normalize to forward slashes
        sample_id = sample_id.replace("/", "__")  # Replace / with __ to avoid path separator issues
    else:
        sample_id = csv_path.stem

    return AnomalySequenceArtifact(
        sample_id=sample_id,
        x=x,
        point_labels=point_labels,
        metadata={
            "dataset_name": dataset_name,
            "source_file": str(csv_path),
            "split": "sequence",
            "label_type": "point",
            "feature_columns": feature_cols,
        },
    )


def load_anomaly_sequence_artifacts_from_dir(
    base_dir: str | Path,
    dataset_name: str,
    label_col: str = "anomaly",
    time_col: Optional[str] = "datetime",
    csv_glob: str = "**/*.csv",
    drop_columns: Optional[list[str]] = None,
    max_files: Optional[int] = None,
    subdirs: Optional[list[str]] = None,
) -> list[AnomalySequenceArtifact]:
    """Load all anomaly sequence artifacts from a hierarchical dataset directory.
    
    Supports both flat and hierarchical structures:
    - Flat: base_dir/dataset_name/*.csv
    - Hierarchical: base_dir/dataset_name/subdir/*.csv (e.g., valve1/, valve2/, other/)
    
    Args:
        base_dir: Parent directory (e.g., /datasets)
        dataset_name: Dataset name (e.g., 'skab')
        label_col: Column name for anomaly labels
        time_col: Column name for time/datetime
        csv_glob: Glob pattern for finding CSV files. Default "**/*.csv" for recursive search.
        drop_columns: Additional columns to drop
        max_files: Maximum number of files to load (None = unlimited)
        subdirs: List of subdirectories to load. If None, loads all. 
                E.g., subdirs=["valve1", "valve2"] will only load from those subdirs.
    
    Returns:
        List of AnomalySequenceArtifact objects with unique sample_ids.
    """
    root = Path(base_dir) / dataset_name
    if not root.exists():
        raise FileNotFoundError(f"Dataset dir not found: {root}")

    # Find all CSV files recursively
    files = sorted(root.rglob(csv_glob))
    
    # Filter by subdirectories if specified
    if subdirs is not None:
        subdirs_set = set(subdirs)
        files = [
            f for f in files
            if f.parent.relative_to(root).parts and f.parent.relative_to(root).parts[0] in subdirs_set
        ]
    
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root}")

    return [
        load_anomaly_sequence_artifact_from_csv(
            csv_path=csv_path,
            dataset_name=dataset_name,
            label_col=label_col,
            time_col=time_col,
            drop_columns=drop_columns,
            root_dir=root,  # Pass root for relative path computation
        )
        for csv_path in files
    ]
