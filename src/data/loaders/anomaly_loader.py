from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.enums import TaskType
from core.schemas import TimeSeriesSample
from data.adapters.anomaly_adapter import load_anomaly_sequence_artifacts_from_dir
from data.dataset_base import DatasetLoaderBase
from data.schemas import (
    AnomalySequenceDatasetBundle,
    AnomalyWindowDatasetBundle,
    DatasetSplit,
)


def _split_samples(samples: list[TimeSeriesSample], train_ratio: float) -> tuple[list[TimeSeriesSample], list[TimeSeriesSample]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1).")

    if len(samples) == 1:
        return samples, samples

    cut = max(1, int(len(samples) * train_ratio))
    cut = min(cut, len(samples) - 1)
    return samples[:cut], samples[cut:]


@dataclass(slots=True)
class SKABAnomalySequenceLoader(DatasetLoaderBase):
    """Load SKAB dataset as anomaly_sequence task (sequence-level binary labels).
    
    Each CSV file becomes one TimeSeriesSample with:
    - x: features (shape T x C)
    - y: single binary label (0 or 1) for the entire sequence
    
    Aggregation rule: 'any' by default (if ANY point is anomalous, sequence is anomalous).
    """

    train_ratio: float = 0.5
    agg_rule: str = "any"
    ratio_threshold: float = 0.1

    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_SEQUENCE

    def load(
        self,
        dataset_name: str,
        base_dir: str | Path,
        label_col: str = "anomaly",
        time_col: Optional[str] = "datetime",
        csv_glob: str = "**/*.csv",
        drop_columns: Optional[list[str]] = None,
        train_ratio: Optional[float] = None,
        max_files: Optional[int] = None,
        subdirs: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AnomalySequenceDatasetBundle:
        """Load SKAB anomaly sequence dataset.
        
        Args:
            dataset_name: Name of the dataset
            base_dir: Parent directory containing dataset
            label_col: Column name for anomaly labels
            time_col: Column name for time/datetime
            csv_glob: Glob pattern for finding CSV files (default: "**/*.csv" for recursive)
            drop_columns: Additional columns to drop
            train_ratio: Ratio of training samples
            max_files: Maximum number of files to load
            subdirs: List of subdirectories to load. E.g., ["valve1", "valve2"] to load only anomalies
                    from valve experiments, or None to load all.
            **kwargs: Additional parameters (agg_rule, ratio_threshold)
        """
        artifacts = load_anomaly_sequence_artifacts_from_dir(
            base_dir=base_dir,
            dataset_name=dataset_name,
            label_col=label_col,
            time_col=time_col,
            csv_glob=csv_glob,
            drop_columns=drop_columns,
            max_files=max_files,
            subdirs=subdirs,
        )

        agg_rule = str(kwargs.get("agg_rule", self.agg_rule))
        agg_threshold = float(kwargs.get("ratio_threshold", self.ratio_threshold))
        samples = [artifact.to_sequence_sample(rule=agg_rule, ratio_threshold=agg_threshold) for artifact in artifacts]

        ratio = float(self.train_ratio if train_ratio is None else train_ratio)
        train_samples, test_samples = _split_samples(samples, ratio)

        for sample in train_samples:
            sample.metadata["split"] = "train"
            sample.metadata["agg_rule"] = agg_rule
        for sample in test_samples:
            sample.metadata["split"] = "test"
            sample.metadata["agg_rule"] = agg_rule

        return AnomalySequenceDatasetBundle(
            dataset_name=dataset_name,
            train=DatasetSplit(samples=train_samples, split_name="train"),
            test=DatasetSplit(samples=test_samples, split_name="test"),
            metadata={
                "num_files": len(samples),
                "base_dir": str(base_dir),
                "label_col": label_col,
                "time_col": time_col,
                "agg_rule": agg_rule,
                "ratio_threshold": agg_threshold if agg_rule == "ratio" else None,
            },
        )


@dataclass(slots=True)
class SKABAnomalyWindowLoader(SKABAnomalySequenceLoader):
    """Load SKAB dataset as anomaly_window task (window-level labels)."""

    window_size: int = 60
    stride: int = 10
    rule: str = "any"
    ratio_threshold: float = 0.1

    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_WINDOW

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> AnomalyWindowDatasetBundle:
        w = int(kwargs.get("window_size", self.window_size))
        s = int(kwargs.get("stride", self.stride))
        r = str(kwargs.get("rule", self.rule))
        th = float(kwargs.get("ratio_threshold", self.ratio_threshold))

        artifacts = load_anomaly_sequence_artifacts_from_dir(
            base_dir=base_dir,
            dataset_name=dataset_name,
            label_col=kwargs.get("label_col", "anomaly"),
            time_col=kwargs.get("time_col", "datetime"),
            csv_glob=kwargs.get("csv_glob", "**/*.csv"),
            drop_columns=kwargs.get("drop_columns"),
            max_files=kwargs.get("max_files"),
            subdirs=kwargs.get("subdirs"),
        )

        ratio = float(kwargs.get("train_ratio", self.train_ratio))
        train_artifacts, test_artifacts = _split_samples(artifacts, ratio)

        train_windows: list[TimeSeriesSample] = []
        test_windows: list[TimeSeriesSample] = []

        for artifact in train_artifacts:
            train_windows.extend(
                artifact.to_window_samples(window_size=w, stride=s, rule=r, ratio_threshold=th)
            )

        for artifact in test_artifacts:
            test_windows.extend(
                artifact.to_window_samples(window_size=w, stride=s, rule=r, ratio_threshold=th)
            )

        for sample in train_windows:
            sample.metadata["split"] = "train"
        for sample in test_windows:
            sample.metadata["split"] = "test"

        return AnomalyWindowDatasetBundle(
            dataset_name=dataset_name,
            train=DatasetSplit(samples=train_windows, split_name="train"),
            test=DatasetSplit(samples=test_windows, split_name="test"),
            window_size=w,
            stride=s,
            rule=r,
            metadata={
                "num_files": len(artifacts),
                "base_dir": str(base_dir),
                "label_col": kwargs.get("label_col", "anomaly"),
                "time_col": kwargs.get("time_col", "datetime"),
                "ratio_threshold": th if r == "ratio" else None,
            },
        )


class NotImplementedAnomalySequenceLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_SEQUENCE

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> dict:
        raise NotImplementedError("Anomaly-sequence dataset loader is not implemented yet.")


class NotImplementedAnomalyWindowLoader(DatasetLoaderBase):
    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_WINDOW

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> dict:
        raise NotImplementedError("Anomaly-window dataset loader is not implemented yet.")


# Backward-compatible alias for earlier code paths.
NotImplementedAnomalyLoader = NotImplementedAnomalyWindowLoader
