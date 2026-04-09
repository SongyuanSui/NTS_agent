# src/memory/artifacts.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.schemas import ChannelSelectorOutput
from utils.io import ensure_dir, read_json, write_json
from utils.json_utils import to_jsonable


def infer_dataset_name(dataset_path: str | Path) -> str:
    """
    Infer dataset name from a dataset directory path.

    Examples
    --------
    datasets/classification/HandMovementDirection -> HandMovementDirection
    /abs/path/to/datasets/anomaly/MSL -> MSL
    """
    path = Path(dataset_path)
    name = path.name.strip()

    if not name:
        raise ValueError(f"Cannot infer dataset name from path: {dataset_path}")

    return name


def normalize_experiment_name(experiment_name: str) -> str:
    """
    Normalize user-provided experiment name for safe filesystem usage.

    Rules
    -----
    - trim spaces
    - replace whitespace with underscores
    - keep only [A-Za-z0-9_-]
    """
    if not isinstance(experiment_name, str):
        raise TypeError("experiment_name must be a string.")

    name = experiment_name.strip()
    if not name:
        raise ValueError("experiment_name must be a non-empty string.")

    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)

    if not name:
        raise ValueError("experiment_name became empty after normalization.")

    return name


def make_memory_run_name(dataset_name: str, experiment_name: str) -> str:
    """
    Build the canonical memory run name:
        <dataset_name>_<experiment_name>
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")

    dataset_name = dataset_name.strip()
    experiment_name = normalize_experiment_name(experiment_name)

    return f"{dataset_name}_{experiment_name}"


def get_memory_root(outputs_root: str | Path = "outputs") -> Path:
    """
    Return the root directory for memory artifacts.
    """
    return Path(outputs_root) / "memory"


def get_memory_run_dir(
    outputs_root: str | Path,
    dataset_name: str,
    experiment_name: str,
    create: bool = False,
) -> Path:
    """
    Return the canonical run directory:
        outputs/memory/<dataset_name>_<experiment_name>/
    """
    run_name = make_memory_run_name(dataset_name, experiment_name)
    run_dir = get_memory_root(outputs_root) / run_name

    if create:
        ensure_dir(run_dir)

    return run_dir


def get_logs_dir(run_dir: str | Path, create: bool = False) -> Path:
    """
    Return the logs directory under a run dir.
    """
    path = Path(run_dir) / "logs"
    if create:
        ensure_dir(path)
    return path


def get_selected_channels_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "selected_channels.json"


def get_memory_bank_path(run_dir: str | Path, filename: str = "memory_bank.jsonl") -> Path:
    return Path(run_dir) / filename


def get_index_ts_path(run_dir: str | Path, filename: str = "index_ts.pkl") -> Path:
    return Path(run_dir) / filename


def get_index_text_path(run_dir: str | Path, filename: str = "index_text.pkl") -> Path:
    return Path(run_dir) / filename


def get_index_stat_path(run_dir: str | Path, filename: str = "index_stat.pkl") -> Path:
    return Path(run_dir) / filename


def get_build_meta_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "build_meta.json"


def ensure_run_dir(
    outputs_root: str | Path,
    dataset_name: str,
    experiment_name: str,
) -> Path:
    """
    Create and return the canonical run directory and its logs subdirectory.
    """
    run_dir = get_memory_run_dir(
        outputs_root=outputs_root,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        create=True,
    )
    get_logs_dir(run_dir, create=True)
    return run_dir


def save_selected_channels(
    selector_output: ChannelSelectorOutput,
    run_dir: str | Path,
    dataset_name: str,
    experiment_name: str,
    task_type: str,
    dataset_path: str | Path | None = None,
) -> Path:
    """
    Save channel selector output to:
        <run_dir>/selected_channels.json
    """
    if not isinstance(selector_output, ChannelSelectorOutput):
        raise TypeError("selector_output must be a ChannelSelectorOutput.")

    run_dir = ensure_dir(run_dir)

    payload = {
        "dataset_name": dataset_name,
        "experiment_name": normalize_experiment_name(experiment_name),
        "task_type": task_type,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "selected_channel_ids": selector_output.selected_channel_ids,
        "ranked_channel_ids": selector_output.ranked_channel_ids,
        "channel_scores": {str(k): float(v) for k, v in selector_output.channel_scores.items()},
        "score_details": {str(k): v for k, v in selector_output.score_details.items()},
        "selection_applied": selector_output.selection_applied,
        "metadata": selector_output.metadata,
    }

    payload = to_jsonable(payload)
    return write_json(get_selected_channels_path(run_dir), payload, indent=2, ensure_ascii=False)


def load_selected_channels(path: str | Path) -> dict[str, Any]:
    """
    Load selected_channels.json and normalize key fields.

    Returns
    -------
    dict with normalized fields:
    - selected_channel_ids: list[int]
    - ranked_channel_ids: list[int]
    - channel_scores: dict[int, float]
    - score_details: dict[int, dict]
    """
    data = read_json(path)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dict JSON content in {path}, got {type(data).__name__}."
        )

    data["selected_channel_ids"] = [int(x) for x in data.get("selected_channel_ids", [])]
    data["ranked_channel_ids"] = [int(x) for x in data.get("ranked_channel_ids", [])]

    raw_scores = data.get("channel_scores", {})
    if not isinstance(raw_scores, dict):
        raise TypeError("channel_scores must be a dict in selected_channels.json.")
    data["channel_scores"] = {int(k): float(v) for k, v in raw_scores.items()}

    raw_details = data.get("score_details", {})
    if not isinstance(raw_details, dict):
        raise TypeError("score_details must be a dict in selected_channels.json.")
    normalized_details: dict[int, dict[str, Any]] = {}
    for k, v in raw_details.items():
        if not isinstance(v, dict):
            raise TypeError("score_details values must be dict objects.")
        normalized_details[int(k)] = v
    data["score_details"] = normalized_details

    return data


def save_build_meta(
    run_dir: str | Path,
    dataset_name: str,
    experiment_name: str,
    task_type: str,
    dataset_path: str | Path,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """
    Save a lightweight build meta file for experiment traceability.
    """
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "experiment_name": normalize_experiment_name(experiment_name),
        "run_name": make_memory_run_name(dataset_name, experiment_name),
        "task_type": task_type,
        "dataset_path": str(dataset_path),
    }

    if extra_meta is not None:
        if not isinstance(extra_meta, dict):
            raise TypeError("extra_meta must be a dict or None.")
        payload.update(extra_meta)

    payload = to_jsonable(payload)
    return write_json(get_build_meta_path(run_dir), payload, indent=2, ensure_ascii=False)


def resolve_run_dir_from_dataset_path(
    dataset_path: str | Path,
    experiment_name: str,
    outputs_root: str | Path = "outputs",
    create: bool = False,
) -> Path:
    """
    Convenience helper:
        dataset_path -> dataset_name -> run_dir
    """
    dataset_name = infer_dataset_name(dataset_path)
    return get_memory_run_dir(
        outputs_root=outputs_root,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        create=create,
    )