"""Evaluator implementations for different task types."""

from __future__ import annotations

from typing import Any, Optional

from agents.schemas import RetrievalOutput
from core.interfaces import BaseEvaluatorInterface
from evaluation.retrieval_metrics import (
    average_channel_metrics,
    evaluate_retrieved_sets_by_channel,
)


class RetrievalEvaluator(BaseEvaluatorInterface):
    """
    Evaluator for retrieval outputs.

    Computes top-k accuracy and precision@k metrics for retrieved candidates.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)

    def validate_input(self, data: Any) -> None:
        """
        Validate that data is a RetrievalOutput.

        Raises
        ------
        TypeError
            If data is not a RetrievalOutput
        """
        if not isinstance(data, RetrievalOutput):
            raise TypeError(f"Expected RetrievalOutput, got {type(data).__name__}")

    def evaluate(
        self,
        data: RetrievalOutput,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate one RetrievalOutput.

        Parameters
        ----------
        data : RetrievalOutput
            The retrieval output to evaluate
        context : dict, optional
            Must contain 'true_label' (ground truth) and 'k' (top-k threshold)

        Returns
        -------
        dict with keys:
            - query_id: str
            - per_channel: dict[channel_id, metrics]
            - macro: dict[top_k_accuracy, precision_at_k]

        Raises
        ------
        ValueError
            If context is missing required keys
        TypeError
            If data is not a RetrievalOutput
        """
        self.validate_input(data)

        if context is None or "true_label" not in context or "k" not in context:
            raise ValueError("context must contain 'true_label' and 'k'")

        true_label = context["true_label"]
        k = context["k"]

        per_channel = evaluate_retrieved_sets_by_channel(
            retrieved_sets=data.retrieved_sets,
            true_label=true_label,
            k=k,
        )
        macro = average_channel_metrics(per_channel)

        return {
            "query_id": data.query_id,
            "per_channel": per_channel,
            "macro": macro,
        }
