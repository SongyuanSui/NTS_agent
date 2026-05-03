from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.loaders.anomaly_loader import SKABAnomalySequenceLoader, SKABAnomalyWindowLoader


def _write_skab_csv(path: Path, anomalies: list[int]) -> None:
    frame = pd.DataFrame(
        {
            "datetime": [
                f"2020-03-09 10:14:{33 + idx:02d}" for idx in range(len(anomalies))
            ],
            "Accelerometer1RMS": [1.0 + idx for idx in range(len(anomalies))],
            "Accelerometer2RMS": [2.0 + idx for idx in range(len(anomalies))],
            "Current": [3.0 + idx for idx in range(len(anomalies))],
            "Pressure": [4.0 + idx for idx in range(len(anomalies))],
            "Temperature": [5.0 + idx for idx in range(len(anomalies))],
            "Thermocouple": [6.0 + idx for idx in range(len(anomalies))],
            "Voltage": [7.0 + idx for idx in range(len(anomalies))],
            "Volume Flow RateRMS": [8.0 + idx for idx in range(len(anomalies))],
            "anomaly": anomalies,
            "changepoint": [0] * len(anomalies),
        }
    )
    frame.to_csv(path, index=False, sep=";")


def test_skab_sequence_loader_uses_any_rule(tmp_path: Path) -> None:
    dataset_root = tmp_path / "demo"
    dataset_root.mkdir()
    _write_skab_csv(dataset_root / "a.csv", [0, 1, 0])
    _write_skab_csv(dataset_root / "b.csv", [0, 0, 0])

    loader = SKABAnomalySequenceLoader()
    bundle = loader.load(dataset_name="demo", base_dir=tmp_path)

    assert len(bundle.train.samples) == 1
    assert len(bundle.test.samples) == 1
    assert bundle.train.samples[0].y == 1
    assert bundle.test.samples[0].y == 0


def test_skab_window_loader_builds_window_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "demo"
    dataset_root.mkdir()
    _write_skab_csv(dataset_root / "a.csv", [0, 1, 0, 0])
    _write_skab_csv(dataset_root / "b.csv", [0, 0, 0, 0])

    loader = SKABAnomalyWindowLoader()
    bundle = loader.load(dataset_name="demo", base_dir=tmp_path, window_size=2, stride=1)

    assert bundle.window_size == 2
    assert bundle.stride == 1
    assert len(bundle.train.samples) == 3
    assert [sample.y for sample in bundle.train.samples] == [1, 1, 0]
    assert [sample.y for sample in bundle.test.samples] == [0, 0, 0]


def test_skab_hierarchical_structure_with_subdirs(tmp_path: Path) -> None:
    """Test loading SKAB data organized in hierarchical subdirectories (valve1, valve2, other, etc.)."""
    dataset_root = tmp_path / "skab"
    dataset_root.mkdir()
    
    # Create subdirectories mimicking real SKAB structure
    (dataset_root / "anomaly-free").mkdir()
    (dataset_root / "valve1").mkdir()
    (dataset_root / "valve2").mkdir()
    (dataset_root / "other").mkdir()
    
    # Write CSV files in each subdirectory
    _write_skab_csv(dataset_root / "anomaly-free" / "anomaly-free.csv", [0, 0, 0])
    _write_skab_csv(dataset_root / "valve1" / "1.csv", [0, 1, 0])
    _write_skab_csv(dataset_root / "valve1" / "2.csv", [1, 1, 1])
    _write_skab_csv(dataset_root / "valve2" / "1.csv", [0, 1, 1])
    _write_skab_csv(dataset_root / "other" / "1.csv", [1, 0, 0])
    
    loader = SKABAnomalySequenceLoader()
    
    # Test 1: Load all files (default behavior)
    bundle_all = loader.load(dataset_name="skab", base_dir=tmp_path)
    total_samples = len(bundle_all.train.samples) + len(bundle_all.test.samples)
    assert total_samples == 5, f"Expected 5 total samples, got {total_samples}"
    
    # Verify sample_ids include subdirectory information (using __ separator)
    all_sample_ids = [s.sample_id for split in [bundle_all.train, bundle_all.test] for s in split.samples]
    assert any("valve1__" in sid for sid in all_sample_ids), "Should have samples from valve1 (format: valve1__<num>)"
    assert any("valve2__" in sid for sid in all_sample_ids), "Should have samples from valve2 (format: valve2__<num>)"
    assert any("other__" in sid for sid in all_sample_ids), "Should have samples from other (format: other__<num>)"
    
    # Test 2: Load only valve1 and valve2 (exclude anomaly-free and other)
    bundle_valves = loader.load(
        dataset_name="skab", 
        base_dir=tmp_path,
        subdirs=["valve1", "valve2"]
    )
    total_valve_samples = len(bundle_valves.train.samples) + len(bundle_valves.test.samples)
    assert total_valve_samples == 3, f"Expected 3 samples from valve dirs, got {total_valve_samples}"
    
    # Verify no samples from anomaly-free or other
    valve_sample_ids = [s.sample_id for split in [bundle_valves.train, bundle_valves.test] for s in split.samples]
    assert all("valve" in sid for sid in valve_sample_ids), "All samples should be from valve dirs (format: valve*__<num>)"
    
    # Test 3: Load only anomaly-free
    bundle_clean = loader.load(
        dataset_name="skab",
        base_dir=tmp_path,
        subdirs=["anomaly-free"]
    )
    # When there's only 1 file, _split_samples returns (samples, samples) for edge case
    # so we get 2 samples (1 train + 1 test pointing to same artifact)
    assert len(bundle_clean.train.samples) >= 1
    assert len(bundle_clean.test.samples) >= 1
    
    clean_sample_ids = [s.sample_id for split in [bundle_clean.train, bundle_clean.test] for s in split.samples]
    assert all("anomaly-free__" in sid for sid in clean_sample_ids), "Should have format: anomaly-free__<filename>"