from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.enums import TaskType
from core.schemas import QueryInstance, TaskSpec, TimeSeriesSample
from retrieval.scoring import apply_normalization, cosine_distance, l2_distance, weighted_l2_distance
from retrieval.stat_retrievers import StatKNNRetriever


def _make_query(query_id: str = "q1") -> QueryInstance:
    sample = TimeSeriesSample(sample_id="query_sample", x=np.array([1.0, 2.0, 3.0]))
    task_spec = TaskSpec(task_type=TaskType.CLASSIFICATION, label_space=["0", "1"])
    return QueryInstance(query_id=query_id, sample=sample, task_spec=task_spec, metadata={})


def test_scoring_functions_basic_behavior() -> None:
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])

    assert cosine_distance(x, x) == pytest.approx(0.0)
    assert cosine_distance(x, y) == pytest.approx(1.0)
    assert l2_distance(x, y) == pytest.approx(np.sqrt(2.0))
    assert weighted_l2_distance(x, y, np.array([1.0, 2.0])) == pytest.approx(np.sqrt(3.0))


def test_apply_normalization_zscore_shapes() -> None:
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_query = np.array([[2.0, 3.0]])

    X_train_n, X_query_n = apply_normalization(X_train, X_query, method="zscore")
    assert X_train_n.shape == (2, 2)
    assert X_query_n.shape == (1, 2)


def test_stat_knn_retriever_returns_top_k_sorted_by_distance() -> None:
    query = _make_query()
    retriever = StatKNNRetriever(config={"distance": "l2", "normalize": "none"})

    memory = [
        {"sample_id": "a", "label": 0, "channel_id": 0, "stat_vector": np.array([0.0, 0.0])},
        {"sample_id": "b", "label": 1, "channel_id": 0, "stat_vector": np.array([2.0, 2.0])},
        {"sample_id": "c", "label": 1, "channel_id": 0, "stat_vector": np.array([5.0, 5.0])},
    ]

    out = retriever.retrieve(
        query=query,
        memory_bank=memory,
        top_k=2,
        context={"query_stat_vector": np.array([1.0, 1.0])},
    )

    assert len(out.examples) == 2
    assert out.sample_ids == ["a", "b"]
    assert out.examples[0].score.value <= out.examples[1].score.value


def test_stat_knn_retriever_supports_weighted_l2() -> None:
    query = _make_query()
    retriever = StatKNNRetriever(
        config={"distance": "weighted_l2", "normalize": "none", "weights": [10.0, 1.0]}
    )

    memory = [
        {"sample_id": "a", "label": 0, "channel_id": 0, "stat_vector": np.array([0.9, 0.0])},
        {"sample_id": "b", "label": 1, "channel_id": 0, "stat_vector": np.array([0.0, 0.2])},
    ]

    out = retriever.retrieve(
        query=query,
        memory_bank=memory,
        top_k=1,
        context={"query_stat_vector": np.array([0.0, 0.0])},
    )

    assert out.sample_ids == ["b"]


def test_stat_knn_retriever_excludes_same_sample_id() -> None:
    query = _make_query()
    query.sample.sample_id = "dup"

    retriever = StatKNNRetriever(config={"distance": "l2", "normalize": "none"})
    memory = [
        {"sample_id": "dup", "label": 0, "channel_id": 0, "stat_vector": np.array([0.0, 0.0])},
        {"sample_id": "other", "label": 1, "channel_id": 0, "stat_vector": np.array([1.0, 1.0])},
    ]

    out = retriever.retrieve(
        query=query,
        memory_bank=memory,
        top_k=2,
        context={"query_stat_vector": np.array([0.1, 0.1])},
    )

    assert out.sample_ids == ["other"]


def test_stat_knn_retriever_rejects_unsupported_distance() -> None:
    query = _make_query()
    retriever = StatKNNRetriever(config={"distance": "invalid_distance"})

    memory = [{"sample_id": "a", "label": 0, "channel_id": 0, "stat_vector": np.array([0.0, 0.0])}]
    with pytest.raises(ValueError):
        retriever.retrieve(
            query=query,
            memory_bank=memory,
            top_k=1,
            context={"query_stat_vector": np.array([0.0, 0.0])},
        )
