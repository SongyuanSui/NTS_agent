from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def accuracy_score(true_labels: Iterable[Any], predicted_labels: Iterable[Any]) -> float:
    true_list = list(true_labels)
    pred_list = list(predicted_labels)
    _validate_equal_length(true_list, pred_list)
    if not true_list:
        return 0.0
    return float(sum(t == p for t, p in zip(true_list, pred_list)) / len(true_list))


def macro_f1_score(true_labels: Iterable[Any], predicted_labels: Iterable[Any]) -> float:
    true_list = list(true_labels)
    pred_list = list(predicted_labels)
    _validate_equal_length(true_list, pred_list)
    labels = sorted({*true_list, *pred_list}, key=str)
    if not labels:
        return 0.0

    f1_values = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true_list, pred_list))
        fp = sum(t != label and p == label for t, p in zip(true_list, pred_list))
        fn = sum(t == label and p != label for t, p in zip(true_list, pred_list))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(sum(f1_values) / len(f1_values))


def confusion_counts(true_labels: Iterable[Any], predicted_labels: Iterable[Any]) -> dict[str, dict[str, int]]:
    true_list = list(true_labels)
    pred_list = list(predicted_labels)
    _validate_equal_length(true_list, pred_list)
    counts = Counter((str(t), str(p)) for t, p in zip(true_list, pred_list))
    return {f"{t}->{p}": {"true": t, "predicted": p, "count": count} for (t, p), count in counts.items()}


def _validate_equal_length(true_list: list[Any], pred_list: list[Any]) -> None:
    if len(true_list) != len(pred_list):
        raise ValueError("true_labels and predicted_labels must have the same length.")
