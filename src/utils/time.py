# src/utils/time.py

from __future__ import annotations

import time as _time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def now_utc_iso() -> str:
    """
    Return the current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def now_local_iso() -> str:
    """
    Return the current local time in ISO-8601 format.
    """
    return datetime.now().astimezone().isoformat()


def timestamp_for_path() -> str:
    """
    Return a compact timestamp string suitable for filenames.
    Example: 20260409_143522
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@contextmanager
def timed_block() -> Iterator[dict[str, float]]:
    """
    Measure elapsed wall-clock time for a code block.

    Example
    -------
    >>> with timed_block() as t:
    ...     do_work()
    >>> print(t["elapsed_sec"])
    """
    info: dict[str, float] = {}
    start = _time.perf_counter()
    try:
        yield info
    finally:
        end = _time.perf_counter()
        info["elapsed_sec"] = end - start


class Timer:
    """
    Simple reusable timer.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def start(self) -> None:
        self._start = _time.perf_counter()
        self._end = None

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.stop() called before start().")
        self._end = _time.perf_counter()
        return self.elapsed_sec

    @property
    def elapsed_sec(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        end = self._end if self._end is not None else _time.perf_counter()
        return end - self._start