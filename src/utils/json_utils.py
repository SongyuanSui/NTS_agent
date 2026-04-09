# src/utils/json_utils.py

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """
    Convert common Python / dataclass / NumPy objects into JSON-serializable forms.

    Supported conversions
    ---------------------
    - dataclass -> dict
    - numpy scalar -> Python scalar
    - numpy array -> list
    - dict / list / tuple / set -> recursively converted
    """
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    return obj


def pretty_jsonable_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convenience wrapper for dict-like objects.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict.")
    return to_jsonable(data)