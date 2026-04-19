from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from core.enums import RepresentationType, RetrievalMode
from core.schemas import QueryInstance
from retrieval.retriever_base import BaseRetriever
from retrieval.schemas import RetrievedExample, RetrievedSet, RetrievalScore
from retrieval.scoring import (
    apply_normalization,
    cosine_distance,
    l2_distance,
    weighted_l2_distance,
)


DistanceFn = Callable[..., float]


_DISTANCE_FUNCS: dict[str, DistanceFn] = {
    "cosine": cosine_distance,
    "l2": l2_distance,
    "weighted_l2": weighted_l2_distance,
}


@dataclass(slots=True)
class _CandidateRow:
    sample_id: str
    label: Any
    channel_id: int
    vector: np.ndarray
    payload: Any
    metadata: dict[str, Any]


class StatKNNRetriever(BaseRetriever):
    """KNN retriever over statistic vectors, adapted from SF_Exp retrieval flow."""

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, config=config, enabled=enabled)

    def _score_impl(self, query: QueryInstance, candidate: Any) -> float:
        distance_name = str(self.get_config("distance", "cosine"))
        distance_fn = self._resolve_distance(distance_name)
        weights = self._resolve_weights()

        query_vector, feature_order = self._extract_query_vector(query=query, context=None)
        candidate_row = self._candidate_to_row(candidate=candidate, feature_order=feature_order)

        if distance_name == "weighted_l2":
            if weights is None:
                raise ValueError("weighted_l2 requires 'weights' in retriever config")
            return distance_fn(query_vector, candidate_row.vector, weights)

        return distance_fn(query_vector, candidate_row.vector)

    def _retrieve_impl(
        self,
        query: QueryInstance,
        memory_bank: Any,
        top_k: int,
        context: Optional[dict[str, Any]] = None,
    ) -> RetrievedSet:
        distance_name = str(self.get_config("distance", "cosine"))
        normalize = str(self.get_config("normalize", "none"))
        distance_fn = self._resolve_distance(distance_name)
        weights = self._resolve_weights()

        query_vector, feature_order = self._extract_query_vector(query=query, context=context)
        candidate_rows = self._collect_candidates(
            memory_bank=memory_bank,
            feature_order=feature_order,
            exclude_sample_id=str(query.sample.sample_id),
        )

        if len(candidate_rows) == 0:
            return RetrievedSet(
                query_id=query.query_id,
                examples=[],
                retrieval_mode=RetrievalMode.STAT.value,
                metadata={"distance": distance_name, "normalize": normalize},
            )

        X_gallery = np.vstack([row.vector for row in candidate_rows])
        X_query = query_vector.reshape(1, -1)

        X_gallery_n, X_query_n = apply_normalization(X_gallery, X_query, normalize)
        q = X_query_n[0]

        if top_k > len(candidate_rows):
            top_k = len(candidate_rows)

        dists = []
        for idx, row in enumerate(candidate_rows):
            g = X_gallery_n[idx]
            if distance_name == "weighted_l2":
                if weights is None:
                    raise ValueError("weighted_l2 requires 'weights' in retriever config")
                dist = distance_fn(q, g, weights)
            else:
                dist = distance_fn(q, g)
            dists.append(dist)

        sorted_indices = np.argsort(np.asarray(dists, dtype=float))[:top_k]
        examples: list[RetrievedExample] = []
        for idx in sorted_indices:
            row = candidate_rows[int(idx)]
            score = RetrievalScore(
                value=float(dists[int(idx)]),
                higher_is_better=False,
                score_name=distance_name,
                metadata={"normalized": normalize != "none"},
            )
            examples.append(
                RetrievedExample(
                    sample_id=row.sample_id,
                    label=row.label,
                    channel_id=row.channel_id,
                    representation_type=RepresentationType.STATISTIC,
                    score=score,
                    payload=row.payload,
                    metadata=dict(row.metadata),
                )
            )

        return RetrievedSet(
            query_id=query.query_id,
            examples=examples,
            retrieval_mode=RetrievalMode.STAT.value,
            metadata={
                "distance": distance_name,
                "normalize": normalize,
                "gallery_size": len(candidate_rows),
                "top_k": len(examples),
            },
        )

    def _resolve_distance(self, name: str) -> DistanceFn:
        if name not in _DISTANCE_FUNCS:
            supported = ", ".join(sorted(_DISTANCE_FUNCS.keys()))
            raise ValueError(f"Unsupported distance '{name}'. Supported: {supported}")
        return _DISTANCE_FUNCS[name]

    def _resolve_weights(self) -> Optional[np.ndarray]:
        weights = self.get_config("weights")
        if weights is None:
            return None
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if np.any(arr < 0.0):
            raise ValueError("weights must be non-negative")
        return arr

    def _extract_query_vector(
        self,
        query: QueryInstance,
        context: Optional[dict[str, Any]],
    ) -> tuple[np.ndarray, Optional[list[str]]]:
        if context and "query_stat_vector" in context:
            vector = np.asarray(context["query_stat_vector"], dtype=float).reshape(-1)
            return vector, None

        if context and "query_stat_dict" in context:
            return self._dict_to_vector(context["query_stat_dict"], None)

        stat_payload = query.metadata.get("statistic_view")
        if stat_payload is None:
            stat_payload = query.metadata.get("stat_vector")

        if stat_payload is None:
            raise ValueError(
                "Query statistic representation is missing. "
                "Provide context['query_stat_vector'] or query.metadata['statistic_view']"
            )

        if isinstance(stat_payload, dict):
            return self._dict_to_vector(stat_payload, None)

        return np.asarray(stat_payload, dtype=float).reshape(-1), None

    def _collect_candidates(
        self,
        memory_bank: Any,
        feature_order: Optional[list[str]],
        exclude_sample_id: Optional[str],
    ) -> list[_CandidateRow]:
        entries = memory_bank.get_all() if hasattr(memory_bank, "get_all") else list(memory_bank)
        rows: list[_CandidateRow] = []
        for entry in entries:
            row = self._candidate_to_row(entry, feature_order)
            if exclude_sample_id is not None and row.sample_id == exclude_sample_id:
                continue
            rows.append(row)
        return rows

    def _candidate_to_row(self, candidate: Any, feature_order: Optional[list[str]]) -> _CandidateRow:
        if isinstance(candidate, dict):
            sample_id = str(candidate.get("sample_id", ""))
            label = candidate.get("label")
            channel_id = int(candidate.get("channel_id", 0))
            payload = candidate.get("payload")
            stat_payload = candidate.get("statistic_view", candidate.get("stat_vector", payload))
            metadata = candidate.get("metadata", {})
        else:
            sample_id = str(getattr(candidate, "sample_id", ""))
            label = getattr(candidate, "label", None)
            channel_id = int(getattr(candidate, "channel_id", 0))
            payload = getattr(candidate, "payload", None)
            stat_payload = getattr(candidate, "statistic_view", None)
            if stat_payload is None:
                stat_payload = getattr(candidate, "stat_vector", payload)
            metadata = getattr(candidate, "metadata", {}) or {}

        if not sample_id:
            raise ValueError("Candidate must provide a non-empty sample_id")

        if isinstance(stat_payload, dict):
            vector, _ = self._dict_to_vector(stat_payload, feature_order)
        else:
            vector = np.asarray(stat_payload, dtype=float).reshape(-1)

        return _CandidateRow(
            sample_id=sample_id,
            label=label,
            channel_id=channel_id,
            vector=vector,
            payload=payload if payload is not None else stat_payload,
            metadata=dict(metadata),
        )

    def _dict_to_vector(
        self,
        payload: dict[str, Any],
        feature_order: Optional[list[str]],
    ) -> tuple[np.ndarray, list[str]]:
        if feature_order is None:
            feature_order = sorted(str(k) for k in payload.keys())

        values = []
        for key in feature_order:
            if key not in payload:
                raise ValueError(f"Missing feature '{key}' in statistic payload")
            values.append(float(payload[key]))
        return np.asarray(values, dtype=float), feature_order
