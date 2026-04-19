from __future__ import annotations

from typing import Optional

import numpy as np


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance: 1 - cosine similarity."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(x, y) / denom)


def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    return float(np.linalg.norm(x - y))


def weighted_l2_distance(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Euclidean distance: sqrt(sum(w * (x-y)^2))."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if x.shape != y.shape or x.shape != weights.shape:
        raise ValueError(
            f"x, y, and weights must have same shape, got {x.shape}, {y.shape}, {weights.shape}"
        )
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    return float(np.sqrt(np.sum(weights * np.square(x - y))))


def apply_normalization(
    X_train: np.ndarray,
    X_query: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit normalization stats on gallery vectors and transform both gallery/query vectors.

    Supported methods: none, zscore, robust, log1p_robust
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)

    if method == "none":
        return X_train, X_query

    if method == "log1p_robust":
        X_train = np.sign(X_train) * np.log1p(np.abs(X_train))
        X_query = np.sign(X_query) * np.log1p(np.abs(X_query))
        method = "robust"

    if method == "zscore":
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0)
        sigma = np.where(sigma == 0.0, 1.0, sigma)
        return (X_train - mu) / sigma, (X_query - mu) / sigma

    if method == "robust":
        median = np.median(X_train, axis=0)
        q1 = np.percentile(X_train, 25, axis=0)
        q3 = np.percentile(X_train, 75, axis=0)
        iqr = q3 - q1
        iqr = np.where(iqr == 0.0, 1.0, iqr)
        return (X_train - median) / iqr, (X_query - median) / iqr

    raise ValueError(f"Unknown normalization method: {method}")
