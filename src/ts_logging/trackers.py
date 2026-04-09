# src/logging/trackers.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from utils.time import now_utc_iso


class MetricTracker:
    """
    Lightweight tracker for experiment metrics.

    Intended use
    ------------
    - evaluation results
    - ablation summaries
    - agent-level metrics
    - pipeline-level metrics

    Output files
    ------------
    - metrics.jsonl  : append-only event-style metric records
    - latest_metrics.json : latest snapshot for convenience
    """

    def __init__(
        self,
        logs_dir: str | Path,
        jsonl_filename: str = "metrics.jsonl",
        latest_filename: str = "latest_metrics.json",
        auto_flush: bool = True,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.logs_dir / jsonl_filename
        self.latest_path = self.logs_dir / latest_filename
        self.auto_flush = auto_flush

        self._fp = open(self.jsonl_path, "a", encoding="utf-8")

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def flush(self) -> None:
        self._fp.flush()

    def __enter__(self) -> "MetricTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def log_metrics(
        self,
        metrics: dict[str, Any],
        split: Optional[str] = None,
        stage: Optional[str] = None,
        step: Optional[int] = None,
        experiment_name: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log one metric record.

        Parameters
        ----------
        metrics:
            Dict of metric_name -> metric_value
        split:
            e.g. train / val / test
        stage:
            e.g. selector / memory_build / inference / evaluation
        step:
            Optional numeric step counter
        experiment_name:
            Optional experiment identifier
        extra:
            Optional extra metadata
        """
        if not isinstance(metrics, dict) or len(metrics) == 0:
            raise ValueError("metrics must be a non-empty dict.")

        record: dict[str, Any] = {
            "timestamp": now_utc_iso(),
            "metrics": metrics,
        }

        if split is not None:
            record["split"] = split
        if stage is not None:
            record["stage"] = stage
        if step is not None:
            record["step"] = step
        if experiment_name is not None:
            record["experiment_name"] = experiment_name
        if extra:
            if not isinstance(extra, dict):
                raise TypeError("extra must be a dict or None.")
            record.update(extra)

        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.auto_flush:
            self.flush()

        # Also save latest snapshot for convenience
        with open(self.latest_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    def log_scalar(
        self,
        name: str,
        value: Any,
        split: Optional[str] = None,
        stage: Optional[str] = None,
        step: Optional[int] = None,
        experiment_name: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Convenience wrapper for logging one scalar metric.
        """
        self.log_metrics(
            metrics={name: value},
            split=split,
            stage=stage,
            step=step,
            experiment_name=experiment_name,
            extra=extra,
        )


class NullTracker:
    """
    No-op tracker useful for optional tracking.
    """

    def log_metrics(self, *args, **kwargs) -> None:
        return None

    def log_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def flush(self) -> None:
        return None