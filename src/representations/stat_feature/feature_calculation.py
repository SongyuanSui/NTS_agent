from __future__ import annotations

import json
import pickle as pkl
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tsfel

from core.schemas import TimeSeriesSample


CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# ========= Helper functions for feature generation and pruning =========
def _extract_feature_names_from_group_file(group_file: Path) -> list[str]:
	feature_names: list[str] = []

	if group_file.suffix.lower() == ".json":
		with open(group_file, "r", encoding="utf-8") as f:
			payload = json.load(f)

		if isinstance(payload, list):
			feature_names = [item for item in payload if isinstance(item, str)]
		elif isinstance(payload, dict):
			raw_features = payload.get("features", [])
			feature_names = [item for item in raw_features if isinstance(item, str)]
	else:
		with open(group_file, "r", encoding="utf-8") as f:
			for line in f:
				text = line.strip()
				if not text or text.startswith("==="):
					continue
				if ". " in text and text.split(". ", 1)[0].isdigit():
					text = text.split(". ", 1)[1]
				feature_names.append(text)

	return [name.strip() for name in feature_names if name.strip()]


def _prune_fft_mean_coefficients(features: pd.DataFrame) -> pd.DataFrame:
	keys_to_drop = []
	for column_name in features.columns:
		if isinstance(column_name, str) and column_name.startswith("0_FFT mean coefficient"):
			keys_to_drop.append(column_name)
	if keys_to_drop:
		features = features.drop(columns=keys_to_drop)
	return features


def _prune_by_group_files(
	features: pd.DataFrame,
	file_name_list: Optional[Iterable[str]],
) -> pd.DataFrame:
	if not file_name_list:
		return features

	allowed: set[str] = set()
	for file_name in file_name_list:
		group_file = CONFIG_DIR / file_name
		if not group_file.exists():
			raise FileNotFoundError(f"Feature group file not found: {group_file}")

		for feature_name in _extract_feature_names_from_group_file(group_file):
			allowed.add(feature_name)

	if not allowed:
		return features

	keep_columns = [column for column in features.columns if isinstance(column, str) and column in allowed]
	return features.loc[:, keep_columns]

# Extract the time series
def _to_univariate_array(
	sample_or_series: TimeSeriesSample | pd.Series | np.ndarray | list[float],
	channel_id: int = 0,
) -> np.ndarray:
	if isinstance(sample_or_series, TimeSeriesSample):
		x = sample_or_series.x
		if x.ndim == 1:
			return x
		if channel_id < 0 or channel_id >= x.shape[1]:
			raise ValueError(
				f"channel_id out of range for sample '{sample_or_series.sample_id}': "
				f"got {channel_id}, valid range is [0, {x.shape[1] - 1}]"
			)
		return x[:, channel_id]

	if isinstance(sample_or_series, pd.Series):
		return sample_or_series.to_numpy(dtype=float)

	arr = np.asarray(sample_or_series, dtype=float)
	if arr.ndim != 1:
		raise ValueError(f"Expected 1D time series, got array with ndim={arr.ndim}")
	return arr

# KEY Function to generate features for a single sample or series, with optional pruning by group files
def generate_single_statistical_features(
	sample_or_series: TimeSeriesSample | pd.Series | np.ndarray | list[float],
	file_name_list: Optional[Iterable[str]] = None,
	channel_id: int = 0,
	prune_fft_mean_coefficients: bool = True,
) -> pd.DataFrame:
	ts = _to_univariate_array(sample_or_series, channel_id=channel_id)
	cfg = tsfel.get_features_by_domain()
	features = tsfel.time_series_features_extractor(cfg, pd.Series(ts))

	if prune_fft_mean_coefficients:
		features = _prune_fft_mean_coefficients(features)

	if file_name_list:
		features = _prune_by_group_files(features, file_name_list)

	return features


def generate_statistical_prompts(
	data: np.ndarray | list[TimeSeriesSample],
	save_path: Optional[str | Path] = None,
	file_name_list: Optional[Iterable[str]] = None,
	sample_names: Optional[list[str]] = None,
	channel_id: int = 0,
	prune_fft_mean_coefficients: bool = True,
	verbose: bool = True,
) -> dict | None:
	"""
	Generate statistical features for a batch of univariate time series.

	Inputs:
	- np.ndarray with shape (m, n)
	- list[TimeSeriesSample] (each sample can be uni/multi-variate; channel_id selects channel)
	"""
	if isinstance(data, list):
		if not data:
			raise ValueError("data list is empty")

		batch = np.array([_to_univariate_array(item, channel_id=channel_id) for item in data], dtype=float)
		inferred_names = [item.sample_id for item in data]
	else:
		batch = np.asarray(data, dtype=float)
		inferred_names = None

	if batch.ndim != 2:
		raise ValueError("data must be a 2D array with shape (m, n)")

	num_samples, series_length = batch.shape
	if sample_names is None:
		sample_names = inferred_names or [f"sample_{i}" for i in range(num_samples)]
	elif len(sample_names) != num_samples:
		raise ValueError("sample_names length must match the number of samples")

	all_statistical_features: list[np.ndarray] = []
	processed_sample_names: list[str] = []
	stat_feature_names: list[str] = []

	if verbose:
		print("Generating statistical features...")
		print(f"Input data shape: {batch.shape}")

	for sample_idx in range(num_samples):
		sample_name = sample_names[sample_idx]
		try:
			statistical_features = generate_single_statistical_features(
				batch[sample_idx],
				file_name_list=file_name_list,
				prune_fft_mean_coefficients=prune_fft_mean_coefficients,
			)

			all_statistical_features.append(statistical_features.iloc[0].to_numpy(dtype=float))
			processed_sample_names.append(sample_name)

			if not stat_feature_names:
				stat_feature_names = [str(name) for name in statistical_features.columns]

			if verbose:
				print(f"Processed {sample_name}: {len(statistical_features.columns)} features")

		except Exception as exc:
			if verbose:
				print(f"Error processing {sample_name}: {exc}")

	if not all_statistical_features:
		if verbose:
			print("No samples were successfully processed")
		return None

	feature_matrix = np.array(all_statistical_features)
	results = {
		"statistical_features": feature_matrix,
		"sample_names": processed_sample_names,
		"stat_feature_names": stat_feature_names,
		"original_data": batch,
		"metadata": {
			"n_samples": len(processed_sample_names),
			"series_length": series_length,
			"n_statistical_features": feature_matrix.shape[1],
			"channel_id": channel_id,
		},
	}

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		with open(save_path, "wb") as f:
			pkl.dump(results, f)
		if verbose:
			print(f"Saved statistical features to: {save_path}")

	return results


def load_statistical_features(prompt_file: str | Path) -> dict:
	with open(prompt_file, "rb") as f:
		return pkl.load(f)
