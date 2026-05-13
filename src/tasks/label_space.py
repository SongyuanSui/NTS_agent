from __future__ import annotations

from typing import Any, Iterable


def normalize_label_space(labels: Iterable[Any] | None) -> list[str]:
    """Return a stable, duplicate-free string label space."""
    if labels is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def ensure_label_in_space(label: Any, label_space: Iterable[Any] | None) -> Any:
    """Validate a label against a non-empty label space."""
    labels = normalize_label_space(label_space)
    if not labels:
        return label

    candidate = str(label).strip() if isinstance(label, str) else label
    if str(candidate) not in labels:
        raise ValueError(f"label {candidate!r} is not in label_space {labels!r}.")
    return candidate


def binary_anomaly_label_space() -> list[str]:
    return ["normal", "anomaly"]
