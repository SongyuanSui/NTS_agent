# src/retrieval/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from core.enums import RepresentationType


Metadata = dict[str, Any]


@dataclass(slots=True)
class RetrievalScore:
    """
    Score object for one query-candidate pair.

    Notes
    -----
    - `value` is the raw score produced by a retriever.
    - `higher_is_better` specifies the semantic direction of the score.
      Example:
        * cosine similarity -> True
        * distance -> False
    """

    value: float
    higher_is_better: bool
    score_name: str = ""
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("RetrievalScore.value must be finite.")

        if not isinstance(self.higher_is_better, bool):
            raise TypeError("higher_is_better must be a bool.")

        if not isinstance(self.score_name, str):
            raise TypeError("score_name must be a string.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RetrievedExample:
    """
    One retrieved candidate.

    payload:
        Usually a lightweight object representing the retrieved content for the
        specific retrieval view, such as:
        - np.ndarray for TS retrieval
        - str for text retrieval
        - dict / np.ndarray for statistic retrieval
    """

    sample_id: str
    label: Optional[Any]
    channel_id: int
    representation_type: RepresentationType
    score: RetrievalScore
    payload: Any = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")

        if not isinstance(self.channel_id, int) or self.channel_id < 0:
            raise ValueError("channel_id must be a non-negative integer.")

        if not isinstance(self.representation_type, RepresentationType):
            raise TypeError("representation_type must be a RepresentationType.")

        if not isinstance(self.score, RetrievalScore):
            raise TypeError("score must be a RetrievalScore.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RetrievedSet:
    """
    Retrieved result for one query.

    Examples are assumed to be ordered according to the retriever's final ranking.
    """

    query_id: str
    examples: list[RetrievedExample]
    retrieval_mode: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")

        if not isinstance(self.examples, list):
            self.examples = list(self.examples)

        for example in self.examples:
            if not isinstance(example, RetrievedExample):
                raise TypeError("examples must contain RetrievedExample objects only.")

        if not isinstance(self.retrieval_mode, str) or not self.retrieval_mode:
            raise ValueError("retrieval_mode must be a non-empty string.")

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def is_empty(self) -> bool:
        return len(self.examples) == 0

    @property
    def labels(self) -> list[Any]:
        return [example.label for example in self.examples]

    @property
    def sample_ids(self) -> list[str]:
        return [example.sample_id for example in self.examples]

    def topk(self, k: int) -> "RetrievedSet":
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        return RetrievedSet(
            query_id=self.query_id,
            examples=self.examples[:k],
            retrieval_mode=self.retrieval_mode,
            metadata=dict(self.metadata),
        )