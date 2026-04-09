# src/logging/event_log.py

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from utils.time import now_utc_iso


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common Python / dataclass / NumPy objects into JSON-serializable forms.
    """
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    return obj


class EventLogger:
    """
    Structured event logger that writes JSONL records.

    Intended use
    ------------
    - pipeline stage tracing
    - agent start/end tracing
    - retrieval summaries
    - reasoning summaries
    - debug-friendly execution replay

    Notes
    -----
    - This logger is for structured machine-readable events, not human log text.
    - One line = one JSON object.
    """

    def __init__(
        self,
        logs_dir: str | Path,
        filename: str = "events.jsonl",
        auto_flush: bool = True,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.logs_dir / filename
        self.auto_flush = auto_flush
        self._fp = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def flush(self) -> None:
        self._fp.flush()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def log_event(
        self,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write one structured event.

        Parameters
        ----------
        event_type:
            Short event name, e.g.:
            - pipeline_start
            - agent_start
            - retrieval_end
            - reasoning_end
        payload:
            Optional dict payload.
        kwargs:
            Extra top-level fields merged into the record.
        """
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string.")

        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict or None.")

        record = {
            "timestamp": now_utc_iso(),
            "event_type": event_type,
            "payload": _to_jsonable(payload),
        }
        if kwargs:
            record.update(_to_jsonable(kwargs))

        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.auto_flush:
            self.flush()

    def log_agent_start(
        self,
        agent_name: str,
        query_id: str | None = None,
        sample_id: str | None = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "agent_name": agent_name,
            "query_id": query_id,
            "sample_id": sample_id,
        }
        if extra:
            payload.update(extra)
        self.log_event("agent_start", payload=payload)

    def log_agent_end(
        self,
        agent_name: str,
        query_id: str | None = None,
        sample_id: str | None = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "agent_name": agent_name,
            "query_id": query_id,
            "sample_id": sample_id,
        }
        if extra:
            payload.update(extra)
        self.log_event("agent_end", payload=payload)

    def log_pipeline_start(
        self,
        pipeline_name: str,
        sample_id: str | None = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "pipeline_name": pipeline_name,
            "sample_id": sample_id,
        }
        if extra:
            payload.update(extra)
        self.log_event("pipeline_start", payload=payload)

    def log_pipeline_end(
        self,
        pipeline_name: str,
        sample_id: str | None = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "pipeline_name": pipeline_name,
            "sample_id": sample_id,
        }
        if extra:
            payload.update(extra)
        self.log_event("pipeline_end", payload=payload)