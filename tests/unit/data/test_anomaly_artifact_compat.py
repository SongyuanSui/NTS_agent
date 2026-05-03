from __future__ import annotations

import numpy as np
from pathlib import Path

from data.collate import collate_samples_to_array
from data.loaders.anomaly_loader import SKABAnomalySequenceLoader


def _write_skab_csv(path: Path, anomalies: list[int]) -> None:
    import pandas as pd

    frame = pd.DataFrame({
        "Time[s]": list(range(len(anomalies))),
        "Volume Flow RateRMS": [10.0 + i for i in range(len(anomalies))],
        "anomaly": anomalies,
        "changepoint": [0] * len(anomalies),
    })
    frame.to_csv(path, index=False, sep=";")


def test_bundle_is_collatable(tmp_path: Path) -> None:
    root = tmp_path / "skab"
    root.mkdir()
    _write_skab_csv(root / "a.csv", [0, 1, 0, 0])
    _write_skab_csv(root / "b.csv", [0, 0, 0, 0])

    loader = SKABAnomalySequenceLoader()
    bundle = loader.load(dataset_name="skab", base_dir=tmp_path)

    # ensure there are samples
    assert len(bundle.train.samples) + len(bundle.test.samples) >= 1

    samples = bundle.train.samples
    X, y = collate_samples_to_array(samples)

    # X should be a numeric ndarray and match number of samples
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(samples)
    # no NaNs and numeric dtype
    assert not np.isnan(X).any()
    assert np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, np.integer)
    assert len(y) == X.shape[0]

    # metadata should include feature column information and match channel count
    sample0 = samples[0]
    assert "feature_columns" in sample0.metadata
    feature_cols = sample0.metadata["feature_columns"]
    assert isinstance(feature_cols, list)
    assert len(feature_cols) == sample0.x.shape[1]
