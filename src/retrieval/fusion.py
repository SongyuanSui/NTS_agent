from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from retrieval.schemas import RetrievedSet, RetrievalScore
from utils.retrieval_compat import unwrap_payload


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


def _view_key_for_mode(retrieval_mode: str) -> str | None:
	mode = str(retrieval_mode).lower()
	if mode == "ts":
		return "ts_view"
	if mode == "text":
		return "summary_view"
	if mode == "stat":
		return "statistic_view"
	return None


def fuse_retrieved_sets(
	retrieved_sets: list[RetrievedSet],
	method: str = "average",
	weights: Optional[list[float]] = None,
	top_k: Optional[int] = None,
	memory_bank: Any = None,
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

	memory_lookup: dict[tuple[str, int], Any] = {}
	if memory_bank is not None:
		entries = memory_bank.get_all() if hasattr(memory_bank, "get_all") else list(memory_bank)
		for entry in entries:
			sample_id = str(getattr(entry, "sample_id", ""))
			channel_id = int(getattr(entry, "channel_id", -1))
			if sample_id and channel_id >= 0:
				memory_lookup[(sample_id, channel_id)] = entry

	# key by (sample_id, channel_id) to preserve channel provenance
	sample_to_scores: dict[tuple[str, int], list[float]] = {}
	sample_to_higher_is_better: dict[tuple[str, int], list[bool]] = {}
	sample_info: dict[tuple[str, int], tuple] = {}
	sample_views: dict[tuple[str, int], dict[str, Any]] = {}
	sample_retrieval_modes: dict[tuple[str, int], set[str]] = {}

	for retrieved_set in retrieved_sets:
		view_key = _view_key_for_mode(retrieved_set.retrieval_mode)
		for example in retrieved_set.examples:
			sample_id = str(example.sample_id)
			channel_id = int(example.channel_id) if example.channel_id is not None else -1
			key = (sample_id, channel_id)
			if key not in sample_to_scores:
				sample_to_scores[key] = []
				sample_to_higher_is_better[key] = []
				sample_info[key] = (example.label, channel_id, unwrap_payload(example.payload), example.metadata)
				sample_views[key] = {}
				sample_retrieval_modes[key] = set()

			sample_retrieval_modes[key].add(retrieved_set.retrieval_mode)
			if view_key is not None and example.payload is not None:
				sample_views[key][view_key] = unwrap_payload(example.payload)

			sample_to_scores[key].append(float(example.score.value))
			sample_to_higher_is_better[key].append(example.score.higher_is_better)

	fused_scores = {}
	for key in sample_to_scores.keys():
		scores = sample_to_scores[key]
		is_better = sample_to_higher_is_better[key]

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

		fused_scores[key] = fused

	sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
	if top_k is not None:
		sorted_items = sorted_items[:top_k]

	from retrieval.schemas import RetrievedExample

	examples = []
	for key, fused_score in sorted_items:
		sample_id, channel_id = key
		label, _, payload, metadata = sample_info[key]
		merged_views = dict(sample_views.get(key, {}))
		memory_entry = memory_lookup.get((sample_id, int(channel_id)))
		if memory_entry is not None:
			if "ts_view" not in merged_views and getattr(memory_entry, "ts_view", None) is not None:
				merged_views["ts_view"] = getattr(memory_entry, "ts_view")
			if "summary_view" not in merged_views and getattr(memory_entry, "summary_view", None) is not None:
				merged_views["summary_view"] = getattr(memory_entry, "summary_view")
			if "statistic_view" not in merged_views and getattr(memory_entry, "statistic_view", None) is not None:
				merged_views["statistic_view"] = getattr(memory_entry, "statistic_view")
		merged_metadata = dict(metadata or {})
		if merged_views:
			# preserve provenance by storing channel_id alongside each view
			prov_views = {}
			for k, v in merged_views.items():
				prov_views[k] = {"channel_id": channel_id, "view": v}
			merged_metadata["retrieval_views"] = prov_views
			merged_metadata["retrieval_modes"] = sorted(sample_retrieval_modes.get(key, set()))
			payload = prov_views
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
				metadata=merged_metadata,
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
