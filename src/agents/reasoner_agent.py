from __future__ import annotations

from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.schemas import ReasonerChannelDecision, ReasonerInput, ReasonerOutput
from core.registry import AGENT_REGISTRY


@AGENT_REGISTRY.decorator("reasoner_agent")
class ReasonerAgent(BaseAgent):
	"""
	Reasoning agent that makes decisions based on retrieved results.

	For each channel, analyzes retrieved neighbors and produces a prediction,
	confidence score, and optional reasoning.
	"""

	def validate_input(self, input_data: Any) -> None:
		if not isinstance(input_data, ReasonerInput):
			raise TypeError(
				f"{self.name}: input_data must be ReasonerInput, "
				f"but got {type(input_data).__name__}."
			)

	def _run_impl(
		self,
		input_data: ReasonerInput,
		context: Optional[dict[str, Any]] = None,
	) -> ReasonerOutput:
		query = input_data.query
		task_spec = input_data.task_spec
		retrieved_sets = input_data.retrieved_sets

		channel_decisions: list[ReasonerChannelDecision] = []
		decision_method = str(self.get_config("decision_method", "majority_vote"))

		for channel_id, retrieved_set in retrieved_sets.items():
			decision = self._make_channel_decision(
				channel_id=channel_id,
				retrieved_set=retrieved_set,
				task_spec=task_spec,
				method=decision_method,
			)
			channel_decisions.append(decision)

		self.log_info(
			context,
			"ReasonerAgent '%s': made decisions for %d channels",
			self.name,
			len(channel_decisions),
		)

		return ReasonerOutput(
			query_id=query.query_id,
			channel_decisions=channel_decisions,
			metadata={
				"sample_id": query.sample.sample_id,
				"num_channels": len(channel_decisions),
				"decision_method": decision_method,
				"task_type": task_spec.task_type.value,
			},
		)

	def _make_channel_decision(
		self,
		channel_id: int,
		retrieved_set: Any,
		task_spec: Any,
		method: str,
	) -> ReasonerChannelDecision:
		"""Make a decision for a single channel based on retrieved neighbors."""
		if not retrieved_set.examples:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="No neighbors retrieved",
				metadata={"num_neighbors": 0},
			)

		if method == "majority_vote":
			return self._majority_vote_decision(channel_id, retrieved_set)
		elif method == "weighted_vote":
			return self._weighted_vote_decision(channel_id, retrieved_set)
		elif method == "top_1":
			return self._top_1_decision(channel_id, retrieved_set)
		else:
			return self._majority_vote_decision(channel_id, retrieved_set)

	def _majority_vote_decision(self, channel_id: int, retrieved_set: Any) -> ReasonerChannelDecision:
		"""Make decision using majority voting among neighbors."""
		from collections import Counter

		labels = retrieved_set.labels
		label_counts = Counter(labels)
		most_common_label = label_counts.most_common(1)[0][0] if label_counts else None

		if most_common_label is None:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="All neighbors have None label",
			)

		max_count = label_counts[most_common_label]
		total_count = len(labels)
		confidence = float(max_count) / float(total_count) if total_count > 0 else 0.0

		reasoning = f"Majority vote: {most_common_label} ({max_count}/{total_count} neighbors)"

		return ReasonerChannelDecision(
			channel_id=channel_id,
			prediction=most_common_label,
			confidence=confidence,
			reasoning=reasoning,
			metadata={
				"num_neighbors": len(labels),
				"vote_counts": dict(label_counts),
			},
		)

	def _weighted_vote_decision(self, channel_id: int, retrieved_set: Any) -> ReasonerChannelDecision:
		"""Make decision using weighted voting based on retrieval scores."""
		label_scores: dict[Any, float] = {}

		for example in retrieved_set.examples:
			label = example.label
			score = float(example.score.value)

			if example.score.higher_is_better:
				weight = score
			else:
				weight = 1.0 / (1.0 + score)

			if label not in label_scores:
				label_scores[label] = 0.0
			label_scores[label] += weight

		if not label_scores:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="No valid labels in neighbors",
			)

		best_label = max(label_scores.keys(), key=lambda l: label_scores[l])
		total_weight = sum(label_scores.values())
		confidence = float(label_scores[best_label]) / float(total_weight) if total_weight > 0 else 0.0

		reasoning = f"Weighted vote: {best_label} (weight={label_scores[best_label]:.3f})"

		return ReasonerChannelDecision(
			channel_id=channel_id,
			prediction=best_label,
			confidence=confidence,
			reasoning=reasoning,
			metadata={
				"num_neighbors": len(retrieved_set.examples),
				"label_scores": {str(k): float(v) for k, v in label_scores.items()},
			},
		)

	def _top_1_decision(self, channel_id: int, retrieved_set: Any) -> ReasonerChannelDecision:
		"""Make decision based on top-1 neighbor."""
		if not retrieved_set.examples:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="No neighbors retrieved",
			)

		top_example = retrieved_set.examples[0]
		confidence_val = 1.0 / (1.0 + float(top_example.score.value))

		reasoning = f"Top-1 neighbor: {top_example.sample_id} (label={top_example.label})"

		return ReasonerChannelDecision(
			channel_id=channel_id,
			prediction=top_example.label,
			confidence=float(confidence_val),
			reasoning=reasoning,
			metadata={
				"top_1_sample_id": top_example.sample_id,
				"top_1_score": float(top_example.score.value),
			},
		)
