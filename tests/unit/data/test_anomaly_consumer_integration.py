from __future__ import annotations

import numpy as np
from pathlib import Path

from representations.schemas import RepresentationInput
from representations.rep_base import BaseRepresentation
from data.loaders.anomaly_loader import SKABAnomalySequenceLoader


class DummyRep(BaseRepresentation):
    @property
    def rep_type(self):
        return "dummy"

    def transform(self, input_data: RepresentationInput, context=None):
        # produce per-sample mean feature (handles both (T,) and (T,C))
        feats = []
        for s in input_data.samples:
            arr = np.asarray(s.x, dtype=float)
            if arr.ndim == 1:
                feats.append(np.array([arr.mean()], dtype=np.float32))
            else:
                feats.append(arr.mean(axis=0).astype(np.float32))
        reps = np.stack(feats, axis=0)
        return type("R", (), {"representations": reps})


def test_bundle_consumable_by_representation(tmp_path: Path) -> None:
    root = tmp_path / "skab"
    root.mkdir()

    # create two files
    import pandas as pd

    frame = pd.DataFrame({
        "Time[s]": [0, 1, 2, 3],
        "Volume Flow RateRMS": [10.0, 11.0, 12.0, 13.0],
        "anomaly": [0, 1, 0, 0],
        "changepoint": [0, 0, 0, 0],
    })
    frame.to_csv(root / "a.csv", index=False, sep=";")
    frame.to_csv(root / "b.csv", index=False, sep=";")

    loader = SKABAnomalySequenceLoader()
    bundle = loader.load(dataset_name="skab", base_dir=tmp_path)

    # run dummy rep on train samples
    rep = DummyRep()
    out = rep.run(RepresentationInput(samples=bundle.train.samples))

    # representations should have same first dim as sample count
    assert hasattr(out, "representations")
    reps = out.representations
    assert reps.shape[0] == len(bundle.train.samples)
    assert not np.isnan(reps).any()

    # metadata should include feature column info and match channel count
    s0 = bundle.train.samples[0]
    assert "feature_columns" in s0.metadata
    assert isinstance(s0.metadata["feature_columns"], list)
    assert len(s0.metadata["feature_columns"]) == s0.x.shape[1]
