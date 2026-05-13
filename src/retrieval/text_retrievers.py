from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from core.enums import RepresentationType, RetrievalMode
from core.schemas import QueryInstance
from retrieval.retriever_base import BaseRetriever
from retrieval.schemas import RetrievedExample, RetrievedSet, RetrievalScore


SimilarityFn = Callable[[str, str], float]


def _string_similarity_ratio(s1: str, s2: str) -> float:
	"""Simple similarity ratio based on common characters."""
	s1_set = set(s1.lower())
	s2_set = set(s2.lower())
	if not s1_set and not s2_set:
		return 1.0
	intersection = len(s1_set & s2_set)
	union = len(s1_set | s2_set)
	return intersection / union if union > 0 else 0.0


def _text_embedding_similarity(s1: str, s2: str) -> float:
	"""
	Placeholder for embedding-based similarity.
	In production, this should use actual embeddings (e.g., sentence-transformers).
	"""
	v1 = _simple_text_vector(s1)
	v2 = _simple_text_vector(s2)
	dot = np.dot(v1, v2)
	norm1 = np.linalg.norm(v1)
	norm2 = np.linalg.norm(v2)
	if norm1 == 0 or norm2 == 0:
		return 0.0
	return float(dot / (norm1 * norm2))


def _simple_text_vector(text: str) -> np.ndarray:
	"""Create a simple numeric vector from text."""
	text = text.lower()
	chars = {}
	for ch in text:
		if ch.isalpha():
			chars[ch] = chars.get(ch, 0) + 1

	if not chars:
		return np.array([0.0])

	sorted_chars = sorted(chars.keys())
	return np.array([float(chars[ch]) for ch in sorted_chars], dtype=float)


_SIMILARITY_FUNCS: dict[str, SimilarityFn] = {
	"ratio": _string_similarity_ratio,
	"embedding": _text_embedding_similarity,
}


class TextSummaryRetriever(BaseRetriever):
	"""Retriever over text summaries (SUMMARY representation)."""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
		enabled: bool = True,
	) -> None:
		super().__init__(name=name, config=config, enabled=enabled)

	def _score_impl(self, query: QueryInstance, candidate: Any) -> float:
		method = str(self.get_config("method", "embedding"))
		similarity_fn = self._resolve_similarity(method)

		query_text = self._extract_query_text(query=query)
		candidate_text = self._extract_candidate_text(candidate=candidate)

		similarity = similarity_fn(query_text, candidate_text)
		return float(1.0 - similarity)

	def _retrieve_impl(
		self,
		query: QueryInstance,
		memory_bank: Any,
		top_k: int,
		context: Optional[dict[str, Any]] = None,
	) -> RetrievedSet:
		method = str(self.get_config("method", "embedding"))
		similarity_fn = self._resolve_similarity(method)

		query_text = self._extract_query_text(query=query)
		candidates = self._collect_candidates(memory_bank=memory_bank)

		if len(candidates) == 0:
			return RetrievedSet(
				query_id=query.query_id,
				examples=[],
				retrieval_mode=RetrievalMode.TEXT.value,
				metadata={"method": method},
			)

		if top_k > len(candidates):
			top_k = len(candidates)

		scores = []
		for candidate_sample_id, candidate_text, candidate_label, candidate_channel_id, candidate_payload, candidate_metadata in candidates:
			if candidate_sample_id == str(query.sample.sample_id):
				continue

			similarity = similarity_fn(query_text, candidate_text)
			distance = 1.0 - similarity
			scores.append((distance, candidate_sample_id, candidate_label, candidate_channel_id, candidate_payload, candidate_metadata))

		scores.sort(key=lambda x: x[0])
		scores = scores[:top_k]

		examples: list[RetrievedExample] = []
		for distance, sample_id, label, channel_id, payload, metadata in scores:
			score = RetrievalScore(
				value=float(distance),
				higher_is_better=False,
				score_name=f"text_{method}",
			)
			examples.append(
				RetrievedExample(
					sample_id=sample_id,
					label=label,
					channel_id=channel_id,
					representation_type=RepresentationType.SUMMARY,
					score=score,
					payload=payload,
					metadata=metadata,
				)
			)

		return RetrievedSet(
			query_id=query.query_id,
			examples=examples,
			retrieval_mode=RetrievalMode.TEXT.value,
			metadata={
				"method": method,
				"gallery_size": len(candidates),
				"top_k": len(examples),
			},
		)

	def _resolve_similarity(self, method: str) -> SimilarityFn:
		if method not in _SIMILARITY_FUNCS:
			raise ValueError(f"Unsupported similarity method '{method}'")
		return _SIMILARITY_FUNCS[method]

	def _extract_query_text(self, query: QueryInstance) -> str:
		summary_payload = query.metadata.get("summary_view")
		if summary_payload is None:
			summary_payload = query.metadata.get("text")

		if not isinstance(summary_payload, str):
			summary_payload = str(summary_payload) if summary_payload is not None else ""

		return summary_payload

	def _collect_candidates(self, memory_bank: Any) -> list[tuple]:
		entries = memory_bank.get_all() if hasattr(memory_bank, "get_all") else list(memory_bank)
		candidates = []

		for entry in entries:
			if not hasattr(entry, "summary_view") or entry.summary_view is None:
				continue

			summary_text = str(entry.summary_view)

			candidates.append((
				entry.sample_id,
				summary_text,
				entry.label,
				entry.channel_id,
				entry.summary_view,
				entry.metadata,
			))

		return candidates

	def _extract_candidate_text(self, candidate: Any) -> str:
		if isinstance(candidate, dict):
			text_payload = candidate.get("summary_view", candidate.get("text", candidate.get("payload")))
		else:
			text_payload = getattr(candidate, "summary_view", None)
			if text_payload is None:
				text_payload = getattr(candidate, "text", None)
			if text_payload is None:
				text_payload = getattr(candidate, "payload", None)

		if text_payload is None:
			raise ValueError("Candidate must have summary_view or text or payload")

		return str(text_payload)
