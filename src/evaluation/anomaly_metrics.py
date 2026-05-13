from __future__ import annotations

from typing import Any, Iterable


def binary_anomaly_metrics(true_labels: Iterable[Any], predicted_labels: Iterable[Any]) -> dict[str, float]:
    true_binary = [_to_binary(label) for label in true_labels]
    pred_binary = [_to_binary(label) for label in predicted_labels]
    if len(true_binary) != len(pred_binary):
        raise ValueError("true_labels and predicted_labels must have the same length.")
    if not true_binary:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp = sum(t == 1 and p == 1 for t, p in zip(true_binary, pred_binary))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_binary, pred_binary))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_binary, pred_binary))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_binary, pred_binary))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(true_binary)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _to_binary(label: Any) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return 1 if int(label) == 1 else 0
    text = str(label).strip().lower()
    if text in {"1", "true", "yes", "anomaly", "anomalous"}:
        return 1
    if text in {"0", "false", "no", "normal", "nominal"}:
        return 0
    raise ValueError(f"Cannot convert label to binary anomaly value: {label!r}")
