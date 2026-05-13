from __future__ import annotations

from typing import Any, Callable, Optional

from retrieval.schemas import RetrievedSet, RetrievedExample, RetrievalScore


FilterFn = Callable[[RetrievedExample], bool]


class RetrievedSetProcessor:
	"""Utility class for post-processing retrieved sets."""

	@staticmethod
	def topk(retrieved_set: RetrievedSet, k: int) -> RetrievedSet:
		"""Keep only top-k results."""
		if k <= 0:
			raise ValueError("k must be positive")
		return retrieved_set.topk(k)

	@staticmethod
	def filter_by_score(
		retrieved_set: RetrievedSet,
		min_score: Optional[float] = None,
		max_score: Optional[float] = None,
	) -> RetrievedSet:
		"""Filter results by score range."""
		filtered = []
		for example in retrieved_set.examples:
			score_val = float(example.score.value)

			if min_score is not None and score_val < min_score:
				continue
			if max_score is not None and score_val > max_score:
				continue

			filtered.append(example)

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=filtered,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "filter": "score"},
		)

	@staticmethod
	def filter_by_label(
		retrieved_set: RetrievedSet,
		allowed_labels: list[Any],
	) -> RetrievedSet:
		"""Keep only results with allowed labels."""
		allowed_set = set(allowed_labels)
		filtered = [ex for ex in retrieved_set.examples if ex.label in allowed_set]

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=filtered,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "filter": "label"},
		)

	@staticmethod
	def filter_by_channel(
		retrieved_set: RetrievedSet,
		channel_ids: list[int],
	) -> RetrievedSet:
		"""Keep only results from allowed channels."""
		channel_set = set(int(cid) for cid in channel_ids)
		filtered = [ex for ex in retrieved_set.examples if ex.channel_id in channel_set]

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=filtered,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "filter": "channel"},
		)

	@staticmethod
	def filter_by_sample_id(
		retrieved_set: RetrievedSet,
		exclude_sample_ids: list[str],
	) -> RetrievedSet:
		"""Exclude results with specific sample IDs."""
		exclude_set = set(exclude_sample_ids)
		filtered = [ex for ex in retrieved_set.examples if ex.sample_id not in exclude_set]

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=filtered,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "filter": "sample_id"},
		)

	@staticmethod
	def filter_custom(
		retrieved_set: RetrievedSet,
		predicate: FilterFn,
	) -> RetrievedSet:
		"""Filter using custom predicate function."""
		filtered = [ex for ex in retrieved_set.examples if predicate(ex)]

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=filtered,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "filter": "custom"},
		)

	@staticmethod
	def deduplicate(
		retrieved_set: RetrievedSet,
		by: str = "sample_id",
	) -> RetrievedSet:
		"""Remove duplicate results based on a field."""
		if by == "sample_id":
			seen = set()
			deduped = []
			for ex in retrieved_set.examples:
				if ex.sample_id not in seen:
					seen.add(ex.sample_id)
					deduped.append(ex)
		elif by == "label":
			seen = set()
			deduped = []
			for ex in retrieved_set.examples:
				if ex.label not in seen:
					seen.add(ex.label)
					deduped.append(ex)
		else:
			raise ValueError(f"Unknown deduplication field: {by}")

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=deduped,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "deduplicated_by": by},
		)

	@staticmethod
	def rerank_by_label_frequency(
		retrieved_set: RetrievedSet,
		preserve_order: bool = False,
	) -> RetrievedSet:
		"""Re-rank results by label frequency (most frequent labels first)."""
		from collections import Counter

		label_counts = Counter(ex.label for ex in retrieved_set.examples)

		def label_score(example: RetrievedExample) -> tuple:
			if preserve_order:
				return (-label_counts[example.label], retrieved_set.examples.index(example))
			return (-label_counts[example.label],)

		reranked = sorted(retrieved_set.examples, key=label_score)

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=reranked,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "reranked_by": "label_frequency"},
		)

	@staticmethod
	def rerank_by_score(
		retrieved_set: RetrievedSet,
		reverse: Optional[bool] = None,
	) -> RetrievedSet:
		"""Re-rank results by score value."""
		if reverse is None:
			if retrieved_set.examples:
				reverse = retrieved_set.examples[0].score.higher_is_better
			else:
				reverse = False

		reranked = sorted(retrieved_set.examples, key=lambda ex: float(ex.score.value), reverse=reverse)

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=reranked,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "reranked_by": "score"},
		)

	@staticmethod
	def add_ranks(retrieved_set: RetrievedSet) -> RetrievedSet:
		"""Add rank information to each result."""
		examples = []
		for rank, example in enumerate(retrieved_set.examples, 1):
			metadata = {**example.metadata, "rank": rank}
			new_example = RetrievedExample(
				sample_id=example.sample_id,
				label=example.label,
				channel_id=example.channel_id,
				representation_type=example.representation_type,
				score=example.score,
				payload=example.payload,
				metadata=metadata,
			)
			examples.append(new_example)

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=examples,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "ranks_added": True},
		)

	@staticmethod
	def normalize_scores(
		retrieved_set: RetrievedSet,
		method: str = "minmax",
	) -> RetrievedSet:
		"""Normalize scores in the result set."""
		if not retrieved_set.examples:
			return retrieved_set

		scores = [float(ex.score.value) for ex in retrieved_set.examples]

		if method == "minmax":
			min_score = min(scores)
			max_score = max(scores)
			denom = max_score - min_score if max_score > min_score else 1.0
			normalized = [(s - min_score) / denom for s in scores]
		elif method == "zscore":
			import numpy as np
			mean = np.mean(scores)
			std = np.std(scores)
			std = 1.0 if std == 0 else std
			normalized = [(s - mean) / std for s in scores]
		else:
			raise ValueError(f"Unknown normalization method: {method}")

		examples = []
		for example, norm_score in zip(retrieved_set.examples, normalized):
			new_score = RetrievalScore(
				value=float(norm_score),
				higher_is_better=example.score.higher_is_better,
				score_name=f"{example.score.score_name}_{method}",
				metadata={**example.score.metadata, "normalized": True},
			)
			new_example = RetrievedExample(
				sample_id=example.sample_id,
				label=example.label,
				channel_id=example.channel_id,
				representation_type=example.representation_type,
				score=new_score,
				payload=example.payload,
				metadata=example.metadata,
			)
			examples.append(new_example)

		return RetrievedSet(
			query_id=retrieved_set.query_id,
			examples=examples,
			retrieval_mode=retrieved_set.retrieval_mode,
			metadata={**retrieved_set.metadata, "scores_normalized": method},
		)


def apply_postprocessing_chain(
	retrieved_set: RetrievedSet,
	operations: list[tuple[str, dict[str, Any]]],
) -> RetrievedSet:
	"""
	Apply a chain of post-processing operations.

	Parameters
	----------
	retrieved_set : RetrievedSet
		Initial retrieved set.
	operations : list[tuple[str, dict]]
		List of (operation_name, operation_kwargs) pairs.

	Returns
	-------
	RetrievedSet
		Processed result set.
	"""
	result = retrieved_set
	processor = RetrievedSetProcessor()

	for op_name, op_kwargs in operations:
		if op_name == "topk":
			result = processor.topk(result, **op_kwargs)
		elif op_name == "filter_score":
			result = processor.filter_by_score(result, **op_kwargs)
		elif op_name == "filter_label":
			result = processor.filter_by_label(result, **op_kwargs)
		elif op_name == "filter_channel":
			result = processor.filter_by_channel(result, **op_kwargs)
		elif op_name == "deduplicate":
			result = processor.deduplicate(result, **op_kwargs)
		elif op_name == "rerank_score":
			result = processor.rerank_by_score(result, **op_kwargs)
		elif op_name == "rerank_frequency":
			result = processor.rerank_by_label_frequency(result, **op_kwargs)
		elif op_name == "add_ranks":
			result = processor.add_ranks(result)
		elif op_name == "normalize_scores":
			result = processor.normalize_scores(result, **op_kwargs)
		else:
			raise ValueError(f"Unknown operation: {op_name}")

	return result
