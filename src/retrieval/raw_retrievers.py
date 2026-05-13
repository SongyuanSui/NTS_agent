from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from core.enums import RepresentationType, RetrievalMode
from core.schemas import QueryInstance
from retrieval.retriever_base import BaseRetriever
from retrieval.schemas import RetrievedExample, RetrievedSet, RetrievalScore
from retrieval.scoring import l2_distance, cosine_distance


DistanceFn = Callable[[np.ndarray, np.ndarray], float]


_DISTANCE_FUNCS: dict[str, DistanceFn] = {
	"cosine": cosine_distance,
	"l2": l2_distance,
	"dtw": None,
}


class RawSeriesRetriever(BaseRetriever):
	"""KNN retriever over raw time series (TS representation)."""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
		enabled: bool = True,
	) -> None:
		super().__init__(name=name, config=config, enabled=enabled)

	def _score_impl(self, query: QueryInstance, candidate: Any) -> float:
		distance_name = str(self.get_config("distance", "l2"))
		distance_fn = self._resolve_distance(distance_name)

		query_series = self._extract_query_series(query=query)
		candidate_series = self._extract_candidate_series(candidate=candidate)

		return distance_fn(query_series, candidate_series)

	def _retrieve_impl(
		self,
		query: QueryInstance,
		memory_bank: Any,
		top_k: int,
		context: Optional[dict[str, Any]] = None,
	) -> RetrievedSet:
		distance_name = str(self.get_config("distance", "l2"))
		distance_fn = self._resolve_distance(distance_name)
		if distance_fn is None:
			raise ValueError(f"Distance '{distance_name}' not implemented")

		query_series = self._extract_query_series(query=query)
		candidates = self._collect_candidates(memory_bank=memory_bank)

		if len(candidates) == 0:
			return RetrievedSet(
				query_id=query.query_id,
				examples=[],
				retrieval_mode=RetrievalMode.TS.value,
				metadata={"distance": distance_name},
			)

		if top_k > len(candidates):
			top_k = len(candidates)

		dists = []
		for candidate_sample_id, candidate_series, candidate_label, candidate_channel_id, candidate_payload, candidate_metadata in candidates:
			if candidate_sample_id == str(query.sample.sample_id):
				continue

			dist = distance_fn(query_series, candidate_series)
			dists.append((dist, candidate_sample_id, candidate_label, candidate_channel_id, candidate_payload, candidate_metadata))

		dists.sort(key=lambda x: x[0])
		dists = dists[:top_k]

		examples: list[RetrievedExample] = []
		for dist, sample_id, label, channel_id, payload, metadata in dists:
			score = RetrievalScore(
				value=float(dist),
				higher_is_better=False,
				score_name=distance_name,
			)
			examples.append(
				RetrievedExample(
					sample_id=sample_id,
					label=label,
					channel_id=channel_id,
					representation_type=RepresentationType.TS,
					score=score,
					payload=payload,
					metadata=metadata,
				)
			)

		return RetrievedSet(
			query_id=query.query_id,
			examples=examples,
			retrieval_mode=RetrievalMode.TS.value,
			metadata={
				"distance": distance_name,
				"gallery_size": len(candidates),
				"top_k": len(examples),
			},
		)

	def _resolve_distance(self, name: str) -> Optional[DistanceFn]:
		if name not in _DISTANCE_FUNCS:
			raise ValueError(f"Unsupported distance '{name}'")
		return _DISTANCE_FUNCS[name]

	def _extract_query_series(self, query: QueryInstance) -> np.ndarray:
		ts_payload = query.metadata.get("ts_view")
		if ts_payload is None:
			ts_payload = query.sample.x

		arr = np.asarray(ts_payload, dtype=float)
		if arr.ndim > 1:
			arr = arr.reshape(-1)
		return arr

	def _collect_candidates(self, memory_bank: Any) -> list[tuple]:
		entries = memory_bank.get_all() if hasattr(memory_bank, "get_all") else list(memory_bank)
		candidates = []

		for entry in entries:
			if not hasattr(entry, "ts_view") or entry.ts_view is None:
				continue

			ts_series = np.asarray(entry.ts_view, dtype=float)
			if ts_series.ndim > 1:
				ts_series = ts_series.reshape(-1)

			candidates.append((
				entry.sample_id,
				ts_series,
				entry.label,
				entry.channel_id,
				entry.ts_view,
				entry.metadata,
			))

		return candidates

	def _extract_candidate_series(self, candidate: Any) -> np.ndarray:
		if isinstance(candidate, dict):
			ts_payload = candidate.get("ts_view", candidate.get("payload"))
		else:
			ts_payload = getattr(candidate, "ts_view", None)
			if ts_payload is None:
				ts_payload = getattr(candidate, "payload", None)

		if ts_payload is None:
			raise ValueError("Candidate must have ts_view or payload")

		arr = np.asarray(ts_payload, dtype=float)
		if arr.ndim > 1:
			arr = arr.reshape(-1)
		return arr
