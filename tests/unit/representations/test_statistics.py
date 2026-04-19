from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.schemas import TimeSeriesSample
import representations.statistics as stats


def test_compute_statistics_for_sample_returns_float_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = pd.DataFrame([{"f1": 1, "f2": 2.5}])
    monkeypatch.setattr(stats, "generate_single_statistical_features", lambda *args, **kwargs: fake_df)

    sample = TimeSeriesSample(sample_id="s1", x=np.array([1.0, 2.0, 3.0]))
    out = stats.compute_statistics_for_sample(sample, feature_groups=["statistics.json"])

    assert out == {"f1": 1.0, "f2": 2.5}


def test_compute_statistics_for_batch_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def _fake_generate_statistical_prompts(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(stats, "generate_statistical_prompts", _fake_generate_statistical_prompts)

    samples = [TimeSeriesSample(sample_id="s1", x=np.array([1.0, 2.0, 3.0]))]
    out = stats.compute_statistics_for_batch(
        samples,
        feature_groups=["statistics.json"],
        channel_id=1,
        verbose=True,
    )

    assert out == {"ok": True}
    assert captured["data"] == samples
    assert captured["file_name_list"] == ["statistics.json"]
    assert captured["channel_id"] == 1
    assert captured["verbose"] is True
