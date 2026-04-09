# src/core/types.py

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np


PathLike: TypeAlias = str | Path
Metadata: TypeAlias = dict[str, Any]
ConfigDict: TypeAlias = dict[str, Any]

ArrayLike1D: TypeAlias = np.ndarray
ArrayLike2D: TypeAlias = np.ndarray

ChannelId: TypeAlias = int
SampleId: TypeAlias = str
QueryId: TypeAlias = str
LabelType: TypeAlias = Any