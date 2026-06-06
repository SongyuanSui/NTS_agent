from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.schemas import RepresentationRecord
from utils.retrieval_compat import unwrap_payload


def normalize_array(
	data: np.ndarray,
	method: str = "minmax",
	axis: Optional[int] = None,
	epsilon: float = 1e-8,
) -> np.ndarray:
	"""
	Normalize an array using specified method.

	Parameters
	----------
	data : np.ndarray
		Input array to normalize.
	method : str
		Normalization method: 'minmax', 'zscore', 'l2', or 'none'.
	axis : int, optional
		Axis along which to normalize. If None, normalize globally.
	epsilon : float
		Small value to avoid division by zero.

	Returns
	-------
	np.ndarray
		Normalized array.
	"""
	data = np.asarray(data, dtype=float)

	if method == "none":
		return data
	elif method == "minmax":
		return _normalize_minmax(data, axis=axis, epsilon=epsilon)
	elif method == "zscore":
		return _normalize_zscore(data, axis=axis, epsilon=epsilon)
	elif method == "l2":
		return _normalize_l2(data, axis=axis, epsilon=epsilon)
	else:
		raise ValueError(f"Unknown normalization method: {method}")


def _normalize_minmax(
	data: np.ndarray,
	axis: Optional[int] = None,
	epsilon: float = 1e-8,
) -> np.ndarray:
	"""Min-max normalization to [0, 1]."""
	min_val = np.min(data, axis=axis, keepdims=True)
	max_val = np.max(data, axis=axis, keepdims=True)
	denom = max_val - min_val + epsilon
	return (data - min_val) / denom


def _normalize_zscore(
	data: np.ndarray,
	axis: Optional[int] = None,
	epsilon: float = 1e-8,
) -> np.ndarray:
	"""Z-score normalization."""
	mean = np.mean(data, axis=axis, keepdims=True)
	std = np.std(data, axis=axis, keepdims=True)
	return (data - mean) / (std + epsilon)


def _normalize_l2(
	data: np.ndarray,
	axis: Optional[int] = None,
	epsilon: float = 1e-8,
) -> np.ndarray:
	"""L2 normalization."""
	norm = np.linalg.norm(data, axis=axis, keepdims=True)
	return data / (norm + epsilon)


def normalize_records(
	records: list[RepresentationRecord],
	method: str = "none",
	payload_key: Optional[str] = None,
) -> list[RepresentationRecord]:
	"""
	Normalize payloads in representation records.

	Parameters
	----------
	records : list[RepresentationRecord]
		List of representation records.
	method : str
		Normalization method.
	payload_key : str, optional
		If the payload is a dict, normalize the values of this key.
		If None, the entire payload is normalized if it's numeric.

	Returns
	-------
	list[RepresentationRecord]
		New records with normalized payloads.
	"""
	if method == "none":
		return records

	normalized_records = []
	for record in records:
		payload = unwrap_payload(record.payload)

		if isinstance(payload, dict):
			if payload_key is None:
				try:
					values = np.array(list(payload.values()), dtype=float)
					normalized = normalize_array(values, method=method)
					normalized_payload = {k: float(v) for k, v in zip(payload.keys(), normalized)}
				except (TypeError, ValueError):
					normalized_payload = payload
			else:
				if payload_key in payload:
					try:
						val = float(payload[payload_key])
						normalized_val = float(normalize_array(np.array([val]), method=method)[0])
						normalized_payload = {**payload, payload_key: normalized_val}
					except (TypeError, ValueError):
						normalized_payload = payload
				else:
					normalized_payload = payload
		elif isinstance(payload, (list, np.ndarray)):
			try:
				arr = np.asarray(payload, dtype=float)
				normalized_arr = normalize_array(arr, method=method)
				normalized_payload = normalized_arr.tolist() if isinstance(payload, list) else normalized_arr
			except (TypeError, ValueError):
				normalized_payload = payload
		else:
			normalized_payload = payload

		normalized_records.append(
			RepresentationRecord(
				rep_type=record.rep_type,
				payload=normalized_payload,
				metadata={**record.metadata, "normalization_method": method},
			)
		)

	return normalized_records


def scale_array(
	data: np.ndarray,
	scale_factor: float = 1.0,
) -> np.ndarray:
	"""Scale an array by a constant factor."""
	return np.asarray(data, dtype=float) * scale_factor


def clip_array(
	data: np.ndarray,
	min_val: Optional[float] = None,
	max_val: Optional[float] = None,
) -> np.ndarray:
	"""Clip array values to a specified range."""
	return np.clip(np.asarray(data, dtype=float), min_val, max_val)
