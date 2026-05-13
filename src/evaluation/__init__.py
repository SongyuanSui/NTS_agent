"""Evaluation module for metrics computation and result analysis."""

from evaluation.agent_metrics import average_confidence, count_agent_outputs
from evaluation.anomaly_metrics import binary_anomaly_metrics
from evaluation.classification_metrics import accuracy_score, confusion_counts, macro_f1_score
from evaluation.evaluators import RetrievalEvaluator
from evaluation.metrics_base import MetricResult
from evaluation.prediction_metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from evaluation.retrieval_metrics import (
    compute_topk_accuracy_and_precision_at_k,
    evaluate_retrieved_set,
    evaluate_retrieved_sets_by_channel,
)

__all__ = [
    "MetricResult",
    "accuracy_score",
    "average_confidence",
    "binary_anomaly_metrics",
    "compute_topk_accuracy_and_precision_at_k",
    "confusion_counts",
    "count_agent_outputs",
    "evaluate_retrieved_set",
    "evaluate_retrieved_sets_by_channel",
    "macro_f1_score",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "RetrievalEvaluator",
]
