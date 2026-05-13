from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.enums import TaskType  # noqa: E402
from core.schemas import TaskSpec  # noqa: E402
from data.dataset_registry import get_dataset_loader  # noqa: E402
from utils.json_utils import to_jsonable  # noqa: E402


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file as a dict."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configs/*.yaml files.") from exc

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge updates into base."""
    result = dict(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config_stack(path: str | Path) -> dict[str, Any]:
    """
    Load a config and recursively merge paths listed under top-level defaults.

    Later files override earlier files; the current file overrides defaults.
    """
    cfg = load_yaml(path)
    merged: dict[str, Any] = {}
    for default_path in cfg.get("defaults", []) or []:
        merged = deep_update(merged, load_config_stack(default_path))

    cfg_no_defaults = {k: v for k, v in cfg.items() if k != "defaults"}
    return deep_update(merged, cfg_no_defaults)


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def load_dataset_from_config(
    cfg: dict[str, Any],
    dataset_loader: str | None = None,
    dataset_name: str | None = None,
    base_dir: str | None = None,
    max_samples_per_split: int | None = None,
) -> Any:
    data_cfg = dict(cfg.get("data", {}))
    loader_name = dataset_loader or data_cfg.get("dataset_loader")
    dataset = dataset_name or data_cfg.get("dataset_name")
    root = base_dir or data_cfg.get("base_dir")

    if not loader_name:
        raise ValueError("dataset_loader is required.")
    if not dataset:
        raise ValueError("dataset_name is required.")
    if not root:
        raise ValueError("base_dir is required.")

    kwargs = dict(cfg.get("loader_params", {}))
    if "remap_labels" in data_cfg:
        kwargs["remap_labels"] = data_cfg["remap_labels"]

    cap = max_samples_per_split
    if cap is None:
        cap = data_cfg.get("max_samples_per_split")
    if cap is not None:
        kwargs["max_samples_per_split"] = int(cap)
    
    loader = get_dataset_loader(str(loader_name))
    return loader.load(dataset_name=str(dataset), base_dir=str(root), **kwargs)


def task_spec_from_config(cfg: dict[str, Any], fallback_task_type: TaskType | str) -> TaskSpec:
    task_cfg = dict(cfg.get("task", {}))
    task_spec_cfg = dict(task_cfg.get("task_spec", {}))
    if "task_type" not in task_spec_cfg:
        task_spec_cfg["task_type"] = fallback_task_type
    return TaskSpec(**task_spec_cfg)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--dataset-loader", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)


def resolve_config(path: str | None, default_path: str) -> dict[str, Any]:
    return load_config_stack(path or default_path)


def print_json(payload: Any) -> None:
    print(json.dumps(to_jsonable(payload), ensure_ascii=True, indent=2))
