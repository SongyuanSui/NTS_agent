from __future__ import annotations

from typing import Any, Optional

from core.schemas import QueryInstance
from retrieval.retriever_base import BaseRetriever
from retrieval.schemas import RetrievedSet
from retrieval.fusion import fuse_retrieved_sets


class HybridRetriever(BaseRetriever):
	"""
	Hybrid retriever that combines multiple retrievers and fuses their results.

	Supports combining TS, text, and statistical retrievers.
	"""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
		enabled: bool = True,
	) -> None:
		super().__init__(name=name, config=config, enabled=enabled)
		self.retrievers: dict[str, BaseRetriever] = {}

	def add_retriever(self, key: str, retriever: BaseRetriever) -> None:
		"""Register a retriever with a key."""
		if not isinstance(retriever, BaseRetriever):
			raise TypeError(f"retriever must be a BaseRetriever, got {type(retriever).__name__}")
		self.retrievers[key] = retriever

	def remove_retriever(self, key: str) -> None:
		"""Unregister a retriever."""
		if key in self.retrievers:
			del self.retrievers[key]

	def get_retriever(self, key: str) -> Optional[BaseRetriever]:
		"""Get a registered retriever by key."""
		return self.retrievers.get(key)

	def _score_impl(self, query: QueryInstance, candidate: Any) -> float:
		raise NotImplementedError(
			"HybridRetriever does not support single-candidate scoring. Use retrieve() instead."
		)

	def _retrieve_impl(
		self,
		query: QueryInstance,
		memory_bank: Any,
		top_k: int,
		context: Optional[dict[str, Any]] = None,
	) -> RetrievedSet:
		context = self.normalize_context(context)

		retrieved_sets: list[RetrievedSet] = []
		active_retrievers = {k: v for k, v in self.retrievers.items() if v.enabled}

		if not active_retrievers:
			raise ValueError("No active retrievers registered in HybridRetriever")

		for key, retriever in active_retrievers.items():
			try:
				result_set = retriever.retrieve(
					query=query,
					memory_bank=memory_bank,
					top_k=top_k,
					context=context,
				)
				retrieved_sets.append(result_set)
			except Exception as e:
				if context.get("skip_failures"):
					continue
				raise RuntimeError(f"Retriever '{key}' failed: {e}")

		if not retrieved_sets:
			raise ValueError("All retrievers failed or no results returned")

		fusion_method = str(self.get_config("fusion_method", "average"))
		fusion_weights = self.get_config("fusion_weights")

		fused = fuse_retrieved_sets(
			retrieved_sets=retrieved_sets,
			method=fusion_method,
			weights=fusion_weights,
			top_k=top_k,
			memory_bank=memory_bank,
		)

		return fused

	def describe(self) -> str:
		return (
			f"{self.__class__.__name__}("
			f"name={self.name}, "
			f"enabled={self.enabled}, "
			f"num_retrievers={len(self.retrievers)}, "
			f"retrievers={list(self.retrievers.keys())}"
			f")"
		)


class WeightedHybridRetriever(HybridRetriever):
	"""Hybrid retriever with configurable weights for each component."""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
		enabled: bool = True,
	) -> None:
		super().__init__(name=name, config=config, enabled=enabled)
		self.retriever_weights: dict[str, float] = {}

	def set_retriever_weight(self, key: str, weight: float) -> None:
		"""Set weight for a retriever."""
		if key not in self.retrievers:
			raise KeyError(f"Retriever '{key}' not found")
		self.retriever_weights[key] = float(weight)

	def _retrieve_impl(
		self,
		query: QueryInstance,
		memory_bank: Any,
		top_k: int,
		context: Optional[dict[str, Any]] = None,
	) -> RetrievedSet:
		context = self.normalize_context(context)

		retrieved_sets: list[RetrievedSet] = []
		weights: list[float] = []

		active_retrievers = {k: v for k, v in self.retrievers.items() if v.enabled}

		if not active_retrievers:
			raise ValueError("No active retrievers registered")

		for key, retriever in active_retrievers.items():
			try:
				result_set = retriever.retrieve(
					query=query,
					memory_bank=memory_bank,
					top_k=top_k,
					context=context,
				)
				retrieved_sets.append(result_set)
				weight = self.retriever_weights.get(key, 1.0 / len(active_retrievers))
				weights.append(weight)
			except Exception as e:
				if context.get("skip_failures"):
					continue
				raise RuntimeError(f"Retriever '{key}' failed: {e}")

		if not retrieved_sets:
			raise ValueError("All retrievers failed or no results returned")

		fusion_method = str(self.get_config("fusion_method", "weighted_sum"))

		fused = fuse_retrieved_sets(
			retrieved_sets=retrieved_sets,
			method=fusion_method,
			weights=weights,
			top_k=top_k,
			memory_bank=memory_bank,
		)

		return fused
