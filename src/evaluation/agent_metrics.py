from __future__ import annotations

from typing import Any, Iterable


def count_agent_outputs(outputs: Iterable[Any]) -> dict[str, int]:
    """Return simple counts by output class name for agent diagnostics."""
    counts: dict[str, int] = {}
    for output in outputs:
        key = output.__class__.__name__
        counts[key] = counts.get(key, 0) + 1
    return counts


def average_confidence(outputs: Iterable[Any]) -> float | None:
    """Average confidence attributes when present."""
    values = []
    for output in outputs:
        confidence = getattr(output, "confidence", None)
        if confidence is not None:
            values.append(float(confidence))
    if not values:
        return None
    return float(sum(values) / len(values))
