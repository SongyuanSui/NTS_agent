from __future__ import annotations

import random
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


def _per_file_window_split(
    windows: list[TimeSeriesSample],
    file_len: int,
    ratio: float,
    gap: int,
) -> tuple[list[TimeSeriesSample], list[TimeSeriesSample]]:
    """Temporal within-file split with a boundary gap.

    Windows ending at/before the cut go to memory (train); windows starting at/
    after cut+gap go to test. Windows overlapping the [cut, cut+gap) buffer are
    dropped so no train window shares any timepoint with any test window (and a
    ``gap``-wide band separates them, preventing near-duplicate overlap leakage).
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("per_file_ratio must be in (0,1).")
    if gap < 0:
        raise ValueError("gap must be >= 0.")

    cut = int(file_len * ratio)
    train: list[TimeSeriesSample] = []
    test: list[TimeSeriesSample] = []
    for wnd in windows:
        ws = int(wnd.metadata["window_start"])
        we = int(wnd.metadata["window_end"])
        if we <= cut:
            train.append(wnd)
        elif ws >= cut + gap:
            test.append(wnd)
        # else: window straddles the [cut, cut+gap) buffer -> dropped
    return train, test


def _sample_windows(
    windows: list[TimeSeriesSample],
    n: Optional[int],
    balance: bool,
    rng: random.Random,
) -> list[TimeSeriesSample]:
    """Optionally subsample windows within a region.

    Sampling is deterministic given ``rng``.

    - balance=False: n=None keeps all windows; otherwise a random n-subset.
    - balance=True: draw an equal number of normal/anomaly windows by
      downsampling the majority class to the minority. With n=None the count per
      class is the minority-class size (so the result is 2*min(#pos, #neg),
      50/50). With n set, each class contributes up to n//2. If only one class is
      present, falls back to plain (optionally capped) sampling.
    """
    if not windows:
        return windows
    if not balance:
        if n is None or n >= len(windows):
            return list(windows)
        return rng.sample(windows, n)

    pos = [w for w in windows if int(w.y) == 1]
    neg = [w for w in windows if int(w.y) == 0]
    m = min(len(pos), len(neg))

    def _take(pool: list[TimeSeriesSample], k: int) -> list[TimeSeriesSample]:
        k = max(0, min(k, len(pool)))
        return rng.sample(pool, k) if k < len(pool) else list(pool)

    if m == 0:
        # Only one class in this region -> nothing to balance against.
        if n is None or n >= len(windows):
            return list(windows)
        return rng.sample(windows, n)

    per_class = m if n is None else min(max(1, n // 2), m)
    return _take(pos, per_class) + _take(neg, per_class)


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
    # Split configuration.
    #   "by_file"  : whole files go to train or test (original behavior).
    #   "per_file" : split each file's timeline into memory/test regions with a
    #                boundary gap (matches within-run / SKAB-style evaluation).
    split_mode: str = "by_file"
    per_file_gap: int = 0          # timepoints left unused between memory & test (per_file only)
    train_samples_per_file: Optional[int] = None
    test_samples_per_file: Optional[int] = None
    balance_train: bool = False
    balance_test: bool = False
    sample_seed: int = 42

    @property
    def task_type(self) -> TaskType:
        return TaskType.ANOMALY_WINDOW

    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> AnomalyWindowDatasetBundle:
        w = int(kwargs.get("window_size", self.window_size))
        s = int(kwargs.get("stride", self.stride))
        r = str(kwargs.get("rule", self.rule))
        th = float(kwargs.get("ratio_threshold", self.ratio_threshold))
        split_mode = str(kwargs.get("split_mode", self.split_mode))

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
        train_windows: list[TimeSeriesSample] = []
        test_windows: list[TimeSeriesSample] = []

        if split_mode == "per_file":
            gap = int(kwargs.get("per_file_gap", self.per_file_gap))
            n_train = kwargs.get("train_samples_per_file", self.train_samples_per_file)
            n_test = kwargs.get("test_samples_per_file", self.test_samples_per_file)
            n_train = None if n_train is None else int(n_train)
            n_test = None if n_test is None else int(n_test)
            balance_train = bool(kwargs.get("balance_train", self.balance_train))
            balance_test = bool(kwargs.get("balance_test", self.balance_test))
            rng = random.Random(int(kwargs.get("sample_seed", self.sample_seed)))

            for artifact in artifacts:
                windows = artifact.to_window_samples(window_size=w, stride=s, rule=r, ratio_threshold=th)
                if not windows:
                    continue
                tr, te = _per_file_window_split(windows, artifact.length, ratio, gap)
                train_windows.extend(_sample_windows(tr, n_train, balance_train, rng))
                test_windows.extend(_sample_windows(te, n_test, balance_test, rng))
        else:
            train_artifacts, test_artifacts = _split_samples(artifacts, ratio)
            for artifact in train_artifacts:
                train_windows.extend(
                    artifact.to_window_samples(window_size=w, stride=s, rule=r, ratio_threshold=th)
                )
            for artifact in test_artifacts:
                test_windows.extend(
                    artifact.to_window_samples(window_size=w, stride=s, rule=r, ratio_threshold=th)
                )

        cap = kwargs.get("max_samples_per_split")
        if cap is not None:
            cap = int(cap)
            if cap >= 0:
                train_windows = train_windows[:cap]
                test_windows = test_windows[:cap]

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
                "split_mode": split_mode,
                "per_file_gap": int(kwargs.get("per_file_gap", self.per_file_gap)) if split_mode == "per_file" else None,
                "train_ratio": ratio,
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
