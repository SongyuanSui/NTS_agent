from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import AggregationInput, AggregationOutput
from core.registry import AGENT_REGISTRY


@AGENT_REGISTRY.decorator("aggregation_agent")
class AggregationAgent(BaseAgent):
	"""
	Aggregation agent that combines channel-level decisions into a final prediction.

	Aggregates channel decisions from the reasoner into a single prediction
	using configurable aggregation strategies (majority vote, weighted avg, etc.).
	"""

	def validate_input(self, input_data: Any) -> None:
		if not isinstance(input_data, AggregationInput):
			raise TypeError(
				f"{self.name}: input_data must be AggregationInput, "
				f"but got {type(input_data).__name__}."
			)

	def _run_impl(
		self,
		input_data: AggregationInput,
		context: Optional[dict[str, Any]] = None,
	) -> AggregationOutput:
		query = input_data.query
		channel_decisions = input_data.channel_decisions

		if not channel_decisions:
			return AggregationOutput(
				query_id=query.query_id,
				prediction=None,
				confidence=0.0,
				reasoning="No channel decisions to aggregate",
				metadata={
					"sample_id": query.sample.sample_id,
					"num_channels": 0,
					"aggregation_method": self.get_config("aggregation_method", "majority_vote"),
				},
			)

		method = str(self.get_config("aggregation_method", "majority_vote"))
		final_prediction, confidence, reasoning = self._aggregate_decisions(
			channel_decisions=channel_decisions,
			method=method,
		)

		self.log_info(
			context,
			"AggregationAgent '%s': aggregated %d channel decisions into prediction",
			self.name,
			len(channel_decisions),
		)

		return AggregationOutput(
			query_id=query.query_id,
			prediction=final_prediction,
			confidence=confidence,
			reasoning=reasoning,
			metadata={
				"sample_id": query.sample.sample_id,
				"num_channels": len(channel_decisions),
				"aggregation_method": method,
			},
		)

	def _aggregate_decisions(
		self,
		channel_decisions: list[Any],
		method: str,
	) -> tuple[Any, float, str]:
		"""Aggregate channel decisions using specified method."""
		if method == "majority_vote":
			return self._majority_vote_aggregation(channel_decisions)
		elif method == "weighted_average":
			return self._weighted_average_aggregation(channel_decisions)
		elif method == "unanimous":
			return self._unanimous_aggregation(channel_decisions)
		elif method == "max_confidence":
			return self._max_confidence_aggregation(channel_decisions)
		else:
			return self._majority_vote_aggregation(channel_decisions)

	def _majority_vote_aggregation(self, decisions: list[Any]) -> tuple[Any, float, str]:
		"""Aggregate using majority voting."""
		from collections import Counter

		predictions = [d.prediction for d in decisions if d.prediction is not None]

		if not predictions:
			return None, 0.0, "All channel predictions are None"

		prediction_counts = Counter(predictions)
		best_prediction = prediction_counts.most_common(1)[0][0]
		max_count = prediction_counts[best_prediction]
		total = len(decisions)

		confidence = float(max_count) / float(total)
		reasoning = f"Majority vote: {best_prediction} ({max_count}/{total} channels)"

		return best_prediction, confidence, reasoning

	def _weighted_average_aggregation(self, decisions: list[Any]) -> tuple[Any, float, str]:
		"""Aggregate using weighted average of confidences."""
		from collections import Counter

		prediction_confidences: dict[Any, list[float]] = {}

		for decision in decisions:
			if decision.prediction is None:
				continue

			prediction = decision.prediction
			confidence = decision.confidence if decision.confidence is not None else 0.5

			if prediction not in prediction_confidences:
				prediction_confidences[prediction] = []
			prediction_confidences[prediction].append(confidence)

		if not prediction_confidences:
			return None, 0.0, "No valid predictions"

		avg_confidences = {
			pred: sum(confs) / len(confs)
			for pred, confs in prediction_confidences.items()
		}

		best_prediction = max(avg_confidences.keys(), key=lambda p: avg_confidences[p])
		final_confidence = float(avg_confidences[best_prediction])

		reasoning = f"Weighted average: {best_prediction} (avg_confidence={final_confidence:.3f})"

		return best_prediction, final_confidence, reasoning

	def _unanimous_aggregation(self, decisions: list[Any]) -> tuple[Any, float, str]:
		"""Aggregate requiring all channels to agree."""
		predictions = [d.prediction for d in decisions if d.prediction is not None]

		if not predictions:
			return None, 0.0, "No valid predictions"

		if len(set(predictions)) == 1:
			best_prediction = predictions[0]
			avg_confidence = sum(d.confidence for d in decisions if d.confidence is not None) / len(
				decisions
			)
			reasoning = f"Unanimous agreement: {best_prediction}"
			return best_prediction, float(avg_confidence), reasoning
		else:
			avg_confidence = sum(d.confidence for d in decisions if d.confidence is not None) / len(
				decisions
			)
			from collections import Counter

			prediction_counts = Counter(predictions)
			best = prediction_counts.most_common(1)[0][0]
			reasoning = "No unanimous agreement, using best prediction"
			return best, float(avg_confidence), reasoning

	def _max_confidence_aggregation(self, decisions: list[Any]) -> tuple[Any, float, str]:
		"""Select prediction with highest confidence."""
		valid_decisions = [d for d in decisions if d.prediction is not None]

		if not valid_decisions:
			return None, 0.0, "No valid predictions"

		best_decision = max(valid_decisions, key=lambda d: d.confidence if d.confidence else 0.0)

		reasoning = f"Max confidence: {best_decision.prediction} (confidence={best_decision.confidence:.3f})"

		return (
			best_decision.prediction,
			float(best_decision.confidence) if best_decision.confidence is not None else 0.0,
			reasoning,
		)
