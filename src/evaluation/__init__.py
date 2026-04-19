"""Evaluation module for metrics computation and result analysis."""

from evaluation.evaluators import RetrievalEvaluator
from evaluation.metrics_base import MetricResult
from evaluation.retrieval_metrics import (
    compute_topk_accuracy_and_precision_at_k,
    evaluate_retrieved_set,
    evaluate_retrieved_sets_by_channel,
)

__all__ = [
    "MetricResult",
    "compute_topk_accuracy_and_precision_at_k",
    "evaluate_retrieved_set",
    "evaluate_retrieved_sets_by_channel",
    "RetrievalEvaluator",
]
