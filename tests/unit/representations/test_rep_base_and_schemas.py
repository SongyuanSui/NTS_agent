from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.enums import RepresentationType
from core.schemas import TimeSeriesSample
from representations.schemas import RepresentationInput, RepresentationOutput
import representations.statistics as stats


def test_representation_input_validates_channel_id() -> None:
    sample = TimeSeriesSample(sample_id="s1", x=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        RepresentationInput(samples=[sample], channel_id=-1)


def test_representation_output_normalizes_rep_type() -> None:
    out = RepresentationOutput(rep_type="statistic")
    assert out.rep_type == RepresentationType.STATISTIC
    assert out.num_records == 0


def test_statistics_representation_run_returns_typed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_compute_statistics_for_batch(**kwargs):
        return {
            "sample_names": ["s1"],
            "stat_feature_names": ["f1", "f2"],
            "statistical_features": np.array([[1.0, 2.5]], dtype=float),
            "metadata": {"n_samples": 1},
        }

    monkeypatch.setattr(stats, "compute_statistics_for_batch", _fake_compute_statistics_for_batch)

    sample = TimeSeriesSample(sample_id="s1", x=np.array([1.0, 2.0, 3.0]))
    rep = stats.StatisticsRepresentation(name="stat_rep")
    out = rep.run(RepresentationInput(samples=[sample]))

    assert out.rep_type == RepresentationType.STATISTIC
    assert out.num_records == 1
    assert out.records[0].metadata["sample_id"] == "s1"
    assert out.records[0].payload == {"f1": 1.0, "f2": 2.5}
