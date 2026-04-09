# src/retrieval/retriever_base.py

from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from core.schemas import QueryInstance
from retrieval.schemas import RetrievedSet
from core.interfaces import BaseRetrieverInterface


class BaseRetriever(BaseRetrieverInterface, ABC):
    """
    Reusable base class for all retrievers.

    Responsibilities
    ----------------
    A concrete retriever is responsible for:
    1. validating a standardized query object
    2. scoring one candidate against the query
    3. retrieving top-k candidates from a memory bank

    Design goals
    ------------
    - keep the interface stable across TS / text / statistic / hybrid retrieval
    - make retrievers easy to swap in ablation experiments
    - avoid imposing view-specific assumptions in the base class
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, config=config)
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def validate_query(self, query: QueryInstance) -> None:
        """
        Default query validation.

        Concrete retrievers may override this if they require more specific
        assumptions, such as:
        - at least one decomposed channel
        - a particular representation payload
        - a particular task type
        """
        if not isinstance(query, QueryInstance):
            raise TypeError(
                f"{self.name}: query must be a QueryInstance, "
                f"but got {type(query).__name__}."
            )

    def validate_top_k(self, top_k: int) -> None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"{self.name}: top_k must be a positive integer.")

    def validate_memory_bank(self, memory_bank: Any) -> None:
        """
        Lightweight default memory-bank validation.

        We intentionally keep this loose because your memory layer may evolve.
        At minimum, the object should not be None and should expose either:
        - get_all()
        - or be directly iterable
        """
        if memory_bank is None:
            raise ValueError(f"{self.name}: memory_bank must not be None.")

        if not hasattr(memory_bank, "get_all") and not hasattr(memory_bank, "__iter__"):
            raise TypeError(
                f"{self.name}: memory_bank must provide get_all() or be iterable."
            )

    def normalize_context(self, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if context is None:
            return {}
        if not isinstance(context, dict):
            raise TypeError(f"{self.name}: context must be a dict or None.")
        return context

    def describe(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"enabled={self.enabled}, "
            f"config_keys={sorted(self.config.keys())}"
            f")"
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        if key not in self.config:
            raise KeyError(f"{self.name}: required config key '{key}' is missing.")
        return self.config[key]

    def pre_retrieve(
        self,
        query: QueryInstance,
        memory_bank: Any,
        top_k: int,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Shared pre-retrieval validation hook.
        """
        if not self.enabled:
            raise RuntimeError(f"{self.name}: retriever is disabled.")

        self.validate_query(query)
        self.validate_memory_bank(memory_bank)
        self.validate_top_k(top_k)

    def post_retrieve(
        self,
        output: RetrievedSet,
        context: Optional[dict[str, Any]] = None,
    ) -> RetrievedSet:
        """
        Shared post-retrieval hook.

        Subclasses may override this for:
        - normalization
        - reranking
        - output validation
        """
        return output

    def retrieve(
        self,
        query: QueryInstance,
        memory_bank: Any,
        top_k: int,
        context: Optional[dict[str, Any]] = None,
    ) -> RetrievedSet:
        """
        Standard retrieval entrypoint.

        Execution flow:
        1. normalize context
        2. pre_retrieve(...)
        3. _retrieve_impl(...)
        4. post_retrieve(...)
        """
        context = self.normalize_context(context)
        self.pre_retrieve(query=query, memory_bank=memory_bank, top_k=top_k, context=context)
        output = self._retrieve_impl(
            query=query,
            memory_bank=memory_bank,
            top_k=top_k,
            context=context,
        )
        return self.post_retrieve(output=output, context=context)

    def score(self, query: QueryInstance, candidate: Any) -> float:
        """
        Public score entrypoint.

        Concrete retrievers should override _score_impl(...) instead of score(...),
        unless they truly need a custom public execution path.
        """
        self.validate_query(query)
        return self._score_impl(query=query, candidate=candidate)

    def _score_impl(self, query: QueryInstance, candidate: Any) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _score_impl(...)."
        )

    def _retrieve_impl(
        self,
        query: QueryInstance,
        memory_bank: Any,
        top_k: int,
        context: Optional[dict[str, Any]] = None,
    ) -> RetrievedSet:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _retrieve_impl(...)."
        )