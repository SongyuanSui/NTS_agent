"""Retrieval evaluation metrics and utility functions."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from retrieval.schemas import RetrievedSet


def evaluate_retrieved_set(
    retrieved_set: RetrievedSet,
    true_label: Any,
    k: int,
) -> dict[str, float]:
    """
    Evaluate a single RetrievedSet against a true label.

    Parameters
    ----------
    retrieved_set : RetrievedSet
        The retrieval result containing scores and examples
    true_label : Any
        The ground truth label to match against
    k : int
        The top-k threshold for accuracy computation

    Returns
    -------
    dict with keys:
        - top_k_accuracy: 1 if true_label in top-k retrieved labels, else 0
        - precision_at_k: ratio of top-k retrieved examples matching true_label
    """
    if not isinstance(retrieved_set, RetrievedSet):
        raise TypeError(f"Expected RetrievedSet, got {type(retrieved_set).__name__}")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Get labels from top-k retrieved examples
    top_k_examples = retrieved_set.examples[:k]
    top_k_labels = [ex.label for ex in top_k_examples]

    # Compute top_k_accuracy
    top_k_accuracy = 1.0 if true_label in top_k_labels else 0.0

    # Compute precision_at_k
    num_matches = sum(1 for label in top_k_labels if label == true_label)
    precision_at_k = num_matches / k if k > 0 else 0.0

    return {
        "top_k_accuracy": top_k_accuracy,
        "precision_at_k": precision_at_k,
    }


def evaluate_retrieved_sets_by_channel(
    retrieved_sets: dict[int, RetrievedSet],
    true_label: Any,
    k: int,
) -> dict[int, dict[str, float]]:
    """
    Evaluate a multi-channel retrieval result.

    Parameters
    ----------
    retrieved_sets : dict[int, RetrievedSet]
        Dict mapping channel_id to retrieval results
    true_label : Any
        The ground truth label
    k : int
        The top-k threshold

    Returns
    -------
    dict mapping channel_id to per-channel metrics
    """
    results = {}
    for channel_id, retrieved_set in retrieved_sets.items():
        results[channel_id] = evaluate_retrieved_set(
            retrieved_set=retrieved_set,
            true_label=true_label,
            k=k,
        )
    return results


def average_channel_metrics(per_channel: dict[int, dict[str, float]]) -> dict[str, float]:
    """
    Average metrics across channels.

    Parameters
    ----------
    per_channel : dict[int, dict[str, float]]
        Per-channel metric results

    Returns
    -------
    dict with averaged metrics (top_k_accuracy, precision_at_k)
    """
    if not per_channel:
        raise ValueError("per_channel must be non-empty")

    metrics_list = list(per_channel.values())
    num_channels = len(metrics_list)

    avg_topk = sum(m["top_k_accuracy"] for m in metrics_list) / num_channels
    avg_precision = sum(m["precision_at_k"] for m in metrics_list) / num_channels

    return {
        "top_k_accuracy": float(avg_topk),
        "precision_at_k": float(avg_precision),
    }


def compute_topk_accuracy_and_precision_at_k(
    true_labels: Iterable[Any],
    predicted_labels: Iterable[Any],
    k: int,
) -> dict[str, float]:
    """
    Compute top-k accuracy and precision@k for a batch of predictions.

    For a single query, if true_label is in the top-k predicted labels:
    - top_k_accuracy += 1
    - precision_at_k += (number of matches in top-k) / k

    Parameters
    ----------
    true_labels : Iterable[Any]
        Ground truth labels for each query
    predicted_labels : Iterable[Iterable[Any]]
        Predicted label lists (each list should be ordered by relevance)
    k : int
        The top-k threshold

    Returns
    -------
    dict with keys:
        - top_k_accuracy: average across all queries
        - precision_at_k: average across all queries
    """
    true_labels = list(true_labels)
    predicted_labels = list(predicted_labels)

    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have same length")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    topk_accs = []
    precisions = []

    for true_label, pred_labels in zip(true_labels, predicted_labels):
        pred_labels = list(pred_labels)[:k]  # Take only top-k

        # Top-k accuracy: 1 if true_label in top-k
        topk_acc = 1.0 if true_label in pred_labels else 0.0
        topk_accs.append(topk_acc)

        # Precision@k: count of matches / k
        num_matches = sum(1 for label in pred_labels if label == true_label)
        precision = num_matches / k if k > 0 else 0.0
        precisions.append(precision)

    n = len(true_labels)
    return {
        "top_k_accuracy": float(sum(topk_accs) / n),
        "precision_at_k": float(sum(precisions) / n),
    }
