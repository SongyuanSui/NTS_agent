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
import representations.stat_feature.feature_calculation as fc


def test_extract_feature_names_from_group_file_json_and_txt(tmp_path: Path) -> None:
    json_file = tmp_path / "g.json"
    json_file.write_text('{"features": ["0_Mean", "0_Variance", "", 1]}', encoding="utf-8")

    txt_file = tmp_path / "g.txt"
    txt_file.write_text("1. 0_Mean\n=== header ===\n0_Variance\n", encoding="utf-8")

    assert fc._extract_feature_names_from_group_file(json_file) == ["0_Mean", "0_Variance"]
    assert fc._extract_feature_names_from_group_file(txt_file) == ["0_Mean", "0_Variance"]


def test_to_univariate_array_supports_channel_selection() -> None:
    sample = TimeSeriesSample(
        sample_id="s1",
        x=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
    )

    selected = fc._to_univariate_array(sample, channel_id=1)
    assert np.allclose(selected, np.array([10.0, 20.0, 30.0]))

    with pytest.raises(ValueError):
        fc._to_univariate_array(sample, channel_id=2)




@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generate_single_statistical_features_applies_pruning_and_group_filter(
    tmp_path: Path,
) -> None:
    """Integration test: verify pruning and group filtering with real TSFEL."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "statistics.json").write_text(
        '{"features": ["0_Mean", "0_Variance"]}',
        encoding="utf-8",
    )

    # Monkey-patch CONFIG_DIR to use temp configs
    original_config_dir = fc.CONFIG_DIR
    try:
        fc.CONFIG_DIR = config_dir

        # Use longer time series to satisfy TSFEL's minimum length requirements
        ts = np.linspace(1.0, 100.0, 256)
        out = fc.generate_single_statistical_features(
            ts,
            file_name_list=["statistics.json"],
            prune_fft_mean_coefficients=True,
        )

        # Verify shape and that only whitelisted features remain
        assert out.shape[0] == 1
        assert set(out.columns) == {"0_Mean", "0_Variance"}

        # Verify values are numeric and reasonable
        for col in out.columns:
            val = float(out.iloc[0][col])
            assert isinstance(val, float)
            assert not np.isnan(val)
    finally:
        fc.CONFIG_DIR = original_config_dir


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generate_statistical_prompts_accepts_time_series_sample_list() -> None:
    """Integration test: verify batch processing with real TSFEL and TimeSeriesSample."""
    samples = [
        TimeSeriesSample(sample_id="a", x=np.linspace(1.0, 50.0, 256)),
        TimeSeriesSample(sample_id="b", x=np.linspace(2.0, 100.0, 256)),
    ]

    out = fc.generate_statistical_prompts(samples, verbose=False)

    assert out is not None
    assert out["sample_names"] == ["a", "b"]
    assert out["statistical_features"].shape[0] == 2  # 2 samples
    assert out["statistical_features"].shape[1] > 0   # some features extracted
    assert out["metadata"]["n_samples"] == 2
    assert out["metadata"]["channel_id"] == 0

    # Verify feature values are all numeric and valid
    for row in out["statistical_features"]:
        for val in row:
            assert isinstance(val, (float, np.floating))
            assert not np.isnan(val)


def test_generate_statistical_prompts_rejects_non_2d_array() -> None:
    with pytest.raises(ValueError):
        fc.generate_statistical_prompts(np.array([1.0, 2.0, 3.0]), verbose=False)
