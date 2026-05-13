from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from retrieval.schemas import RetrievedSet, RetrievalScore


ScoreFusionFn = Callable[[list[float], list[bool]], float]


def _reciprocal_rank_fusion(scores: list[float], higher_is_better: list[bool], k: float = 60.0) -> float:
	"""Reciprocal Rank Fusion: 1 / (k + rank)."""
	rank = 1
	fused = 0.0
	for score, is_better in zip(scores, higher_is_better):
		if is_better:
			fused += 1.0 / (k + rank)
		else:
			fused -= 1.0 / (k + rank)
		rank += 1
	return fused


def _weighted_sum_fusion(scores: list[float], weights: list[float]) -> float:
	"""Weighted sum of scores."""
	total = 0.0
	for score, weight in zip(scores, weights):
		total += float(score) * float(weight)
	return total


def _normalized_weighted_sum_fusion(scores: list[float], higher_is_better: list[bool], weights: list[float]) -> float:
	"""Weighted sum with normalization by direction."""
	normalized = []
	for score, is_better, weight in zip(scores, higher_is_better, weights):
		s = float(score)
		if is_better:
			normalized.append(s * float(weight))
		else:
			normalized.append(-s * float(weight))
	return sum(normalized)


def _max_fusion(scores: list[float], higher_is_better: list[bool]) -> float:
	"""Maximum score (with direction awareness)."""
	if not scores:
		return 0.0
	adjusted = [s if is_better else -s for s, is_better in zip(scores, higher_is_better)]
	return max(adjusted)


def _min_fusion(scores: list[float], higher_is_better: list[bool]) -> float:
	"""Minimum score (with direction awareness)."""
	if not scores:
		return 0.0
	adjusted = [s if is_better else -s for s, is_better in zip(scores, higher_is_better)]
	return min(adjusted)


def _average_fusion(scores: list[float], higher_is_better: list[bool]) -> float:
	"""Average of scores (with direction awareness)."""
	if not scores:
		return 0.0
	adjusted = [s if is_better else -s for s, is_better in zip(scores, higher_is_better)]
	return sum(adjusted) / len(adjusted)


_FUSION_FUNCS: dict[str, Any] = {
	"rrf": _reciprocal_rank_fusion,
	"weighted_sum": _weighted_sum_fusion,
	"normalized_weighted_sum": _normalized_weighted_sum_fusion,
	"max": _max_fusion,
	"min": _min_fusion,
	"average": _average_fusion,
}


def fuse_retrieved_sets(
	retrieved_sets: list[RetrievedSet],
	method: str = "average",
	weights: Optional[list[float]] = None,
	top_k: Optional[int] = None,
) -> RetrievedSet:
	"""
	Fuse multiple retrieved sets into a single ranked list.

	Parameters
	----------
	retrieved_sets : list[RetrievedSet]
		Multiple retrieved sets to fuse.
	method : str
		Fusion method: 'rrf', 'weighted_sum', 'max', 'min', 'average'.
	weights : list[float], optional
		Weights for each retrieved set (used by weighted methods).
	top_k : int, optional
		Number of top results to return. If None, returns all.

	Returns
	-------
	RetrievedSet
		Fused and ranked retrieved set.
	"""
	if not retrieved_sets:
		raise ValueError("retrieved_sets must be non-empty")

	if method not in _FUSION_FUNCS:
		raise ValueError(f"Unknown fusion method: {method}")

	query_id = retrieved_sets[0].query_id

	sample_to_scores: dict[str, list[float]] = {}
	sample_to_higher_is_better: dict[str, list[bool]] = {}
	sample_info: dict[str, tuple] = {}

	for retrieved_set in retrieved_sets:
		for example in retrieved_set.examples:
			sample_id = example.sample_id
			if sample_id not in sample_to_scores:
				sample_to_scores[sample_id] = []
				sample_to_higher_is_better[sample_id] = []
				sample_info[sample_id] = (example.label, example.channel_id, example.payload, example.metadata)

			sample_to_scores[sample_id].append(float(example.score.value))
			sample_to_higher_is_better[sample_id].append(example.score.higher_is_better)

	fused_scores = {}
	for sample_id in sample_to_scores.keys():
		scores = sample_to_scores[sample_id]
		is_better = sample_to_higher_is_better[sample_id]

		if method == "rrf":
			fused = _reciprocal_rank_fusion(scores, is_better)
		elif method == "weighted_sum":
			if weights is None:
				weights = [1.0 / len(retrieved_sets)] * len(retrieved_sets)
			w = [weights[i % len(weights)] for i in range(len(scores))]
			fused = _weighted_sum_fusion(scores, w)
		elif method == "normalized_weighted_sum":
			if weights is None:
				weights = [1.0 / len(retrieved_sets)] * len(retrieved_sets)
			w = [weights[i % len(weights)] for i in range(len(scores))]
			fused = _normalized_weighted_sum_fusion(scores, is_better, w)
		elif method == "max":
			fused = _max_fusion(scores, is_better)
		elif method == "min":
			fused = _min_fusion(scores, is_better)
		else:
			fused = _average_fusion(scores, is_better)

		fused_scores[sample_id] = fused

	sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
	if top_k is not None:
		sorted_items = sorted_items[:top_k]

	from retrieval.schemas import RetrievedExample

	examples = []
	for sample_id, fused_score in sorted_items:
		label, channel_id, payload, metadata = sample_info[sample_id]
		score = RetrievalScore(
			value=float(fused_score),
			higher_is_better=True,
			score_name=f"fused_{method}",
		)
		examples.append(
			RetrievedExample(
				sample_id=sample_id,
				label=label,
				channel_id=channel_id,
				representation_type=retrieved_sets[0].examples[0].representation_type if retrieved_sets[0].examples else None,
				score=score,
				payload=payload,
				metadata=metadata,
			)
		)

	return RetrievedSet(
		query_id=query_id,
		examples=examples,
		retrieval_mode=f"fused_{method}",
		metadata={
			"num_retrieved_sets": len(retrieved_sets),
			"fusion_method": method,
			"fused_count": len(examples),
		},
	)


def combine_scores(
	scores: list[float],
	higher_is_better: list[bool],
	method: str = "average",
) -> float:
	"""Combine multiple scores into a single value."""
	if method not in ["max", "min", "average"]:
		raise ValueError(f"Unknown combination method: {method}")

	if method == "max":
		return _max_fusion(scores, higher_is_better)
	elif method == "min":
		return _min_fusion(scores, higher_is_better)
	else:
		return _average_fusion(scores, higher_is_better)
