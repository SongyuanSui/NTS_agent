# src/utils/seed.py

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for Python and NumPy.

    Parameters
    ----------
    seed:
        Global random seed.
    deterministic:
        Reserved flag for future extension (e.g., torch deterministic mode).
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Placeholder for future extensions, e.g. torch/cudnn.
    _ = deterministic