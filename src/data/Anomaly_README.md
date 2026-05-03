# Hierarchical Anomaly Dataset Loading Framework

This document consolidates the design, implementation, and usage guidelines for the hierarchical anomaly dataset loading framework, with a focus on the SKAB dataset.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [SKAB Dataset Organization](#skab-dataset-organization)
4. [Sample ID Format](#sample-id-format)
5. [Usage Examples](#usage-examples)
6. [Implementation Details](#implementation-details)
7. [API Reference](#api-reference)
8. [Quick Reference](#quick-reference)

---

## Overview

The anomaly dataset loading framework supports hierarchical directory structures, enabling flexible organization and selective loading of multi-type anomaly data. Originally designed for SKAB, the architecture generalizes to other hierarchical anomaly datasets.

### Key Features

- ✅ **Recursive directory scanning** - Automatically discovers CSV files in nested subdirectories
- ✅ **Selective subdirectory loading** - Choose specific anomaly types (e.g., "valve1", "valve2")
- ✅ **Unique sample IDs** - Prevents name collisions between files in different directories
- ✅ **Cross-platform compatibility** - Unified ID format across Windows, Linux, and macOS
- ✅ **Backward compatible** - Existing flat-structure datasets continue to work unchanged

---

## Architecture

### Three-Layer Design

The framework follows a **three-layer adapter pattern**:

```
CSV Files (hierarchical)
    ↓ Layer 1: Recursive Glob
    ↓ (find all .csv files recursively)
    ↓
All Files Found
    ↓ Layer 2: Subdirectory Filter
    ↓ (optional: select specific subdirs)
    ↓
Filtered Files
    ↓ Layer 3: Relative Path ID
    ↓ (generate unique IDs with __ separator)
    ↓
AnomalySequenceArtifact
    ↓ (aggregate point labels → sequence label)
    ↓
TimeSeriesSample
    ↓ (train/test split)
    ↓
DatasetBundle
```

### Layer 1: Recursive Directory Scanning

**Old approach** (flat only):
```python
files = root.glob("*.csv")  # Only top-level
```

**New approach** (hierarchical support):
```python
files = root.rglob("**/*.csv")  # Recursive search
# Finds: anomaly-free/file.csv, valve1/1.csv, valve1/2.csv, ...
```

### Layer 2: Subdirectory Filtering

**Filter logic**:
```python
if subdirs is not None:
    subdirs_set = set(subdirs)  # {"valve1", "valve2"}
    files = [
        f for f in files
        if f.parent.relative_to(root).parts 
        and f.parent.relative_to(root).parts[0] in subdirs_set
    ]
```

**How it works**:
```python
file = Path("datasets/skab/valve1/1.csv")
root = Path("datasets/skab")

# Extract first-level subdirectory
subdir = file.parent.relative_to(root).parts[0]  # "valve1"

# Check if in desired set
if subdir in {"valve1", "valve2"}:
    keep_file = True  # ✓ Include
```

### Layer 3: Unique Sample ID Generation

**Why `__` instead of `/`?**

The `/` character has problems:
- ❌ Interpreted as path separator on Unix/Linux/Mac
- ❌ Cannot be used in filenames
- ❌ May need escaping in databases/JSON
- ❌ Causes issues with string operations like `split("/")`

The `__` separator is better:
- ✅ No special meaning in any OS
- ✅ Can be used directly in filenames
- ✅ No escaping needed anywhere
- ✅ Safe string operations
- ✅ Low collision risk (would need folder named `__`)

**Implementation**:
```python
if root_dir is not None:
    relative_path = csv_path.relative_to(Path(root_dir))
    # E.g., Path("valve1/1.csv")
    
    sample_id = str(relative_path.with_suffix(""))
    # E.g., "valve1/1"
    
    sample_id = sample_id.replace("\\", "/")  # Windows → Unix
    # E.g., "valve1/1"
    
    sample_id = sample_id.replace("/", "__")  # Final conversion
    # E.g., "valve1__1" ✅
```

---

## SKAB Dataset Organization

### Recommended Directory Structure

```
NTS_agent/datasets/
└── skab/
    ├── anomaly-free/
    │   └── anomaly-free.csv              # Normal mode (no anomalies)
    ├── valve1/                           # Valve closed at pump inlet
    │   ├── 1.csv
    │   ├── 2.csv
    │   ├── ...
    │   └── 16.csv
    ├── valve2/                           # Valve closed at pump outlet
    │   ├── 1.csv
    │   ├── 2.csv
    │   ├── 3.csv
    │   └── 4.csv
    └── other/                            # Other experimental conditions
        ├── 1.csv                         # Fluid leaks and additions
        ├── 2.csv                         # Fluid leaks and additions
        ├── ...
        ├── 13.csv                        # Two-phase flow (cavitation)
        └── 14.csv                        # High-temperature water supply
```

### Why This Structure?

1. **Semantic Grouping** - Each subdirectory represents a specific anomaly type or experimental condition
2. **Flexibility** - Load all anomalies, specific types, or separations of normal/anomalous data
3. **Extensibility** - Other datasets can use the same pattern
4. **Traceability** - Subdirectory information is preserved in sample IDs

### Data Format

SKAB CSV files use semicolon (`;`) delimiter:

```
datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS;anomaly;changepoint
2020-03-09 10:14:33;0.75;0.81;0.36;40.61;35.59;39.78;220.07;0.59;0;0
2020-03-09 10:14:34;0.76;0.82;0.36;40.62;35.60;39.79;220.08;0.60;0;0
```

- **anomaly**: Point-level anomaly label (0 = normal, 1 = anomalous)
- **changepoint**: Change point marker (usually 0)
- Other columns: Sensor measurements from the hydraulic system

### SKAB Statistics

| Anomaly Type | Files | Description |
|--------------|-------|-------------|
| `anomaly-free` | 1 | Normal mode operation |
| `valve1` | 16 | Pump inlet valve closure |
| `valve2` | 4 | Pump outlet valve closure |
| `other` | 14 | Other anomaly types |
| **Total** | **~35** | Varies with duplicates |

---

## Sample ID Format

### Naming Convention

The sample ID is generated from the relative path with `/` replaced by `__`:

```
File Path                           → Sample ID
──────────────────────────────────────────────────────
anomaly-free/anomaly-free.csv     → anomaly-free__anomaly-free
valve1/1.csv                      → valve1__1
valve2/3.csv                      → valve2__3
other/14.csv                      → other__14
valve_1/1.csv (underscore in name) → valve_1__1
```

### Why `__` Is Superior to `/`

| Factor | `/` (old) | `__` (new) |
|--------|-----------|-----------|
| **System Special Char** | ❌ Path separator | ✅ None |
| **Filename Compatible** | ❌ Not allowed | ✅ Yes |
| **Database Safe** | ⚠️ Needs escaping | ✅ Direct use |
| **String Operations** | ⚠️ split("/") unsafe | ✅ Safe parsing |
| **Cross-platform** | ⚠️ Windows `\` issues | ✅ Unified |
| **Collision Risk** | ⚠️ High (system level) | ✅ Low |

### Reverse Conversion

To recover the original path from a sample ID:

```python
def reverse_sample_id(sample_id: str) -> str:
    return sample_id.replace("__", "/")

# Examples:
reverse_sample_id("valve1__1")                    # → "valve1/1"
reverse_sample_id("anomaly-free__anomaly-free")   # → "anomaly-free/anomaly-free"
```

---

## Usage Examples

### Example 1: Load All Files

```python
from pathlib import Path
from data.loaders.anomaly_loader import SKABAnomalySequenceLoader

loader = SKABAnomalySequenceLoader()
bundle = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets")
    # Defaults: loads all CSV files recursively
)

# Result:
# - ~35 samples total (all subdirectories)
# - Split into train/test (50/50 by default)
print(f"Train samples: {len(bundle.train.samples)}")
print(f"Test samples: {len(bundle.test.samples)}")
```

### Example 2: Load Specific Anomaly Types

```python
# Load only valve-related anomalies (exclude normal and other)
bundle = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    subdirs=["valve1", "valve2"]
)

# Result:
# - 20 samples (16 from valve1 + 4 from valve2)
# - Useful for comparing different failure modes
```

### Example 3: Separate Normal and Anomalous Data

```python
# Load only normal data (for negative examples)
normal_bundle = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    subdirs=["anomaly-free"]
)

# Load all anomalies
anomaly_bundle = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    subdirs=["valve1", "valve2", "other"]
)

# Use case: One-class classification
# Train on normal_bundle, test on anomaly_bundle
```

### Example 4: Window-Level Anomaly Detection

```python
from data.loaders.anomaly_loader import SKABAnomalyWindowLoader

loader = SKABAnomalyWindowLoader()
bundle = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    subdirs=["valve1"],           # Optional: load only valve1
    window_size=60,               # Temporal window length
    stride=10,                    # Sliding window step
    rule="any"                    # Aggregation: any point anomalous → window anomalous
)

# Result:
# - Multiple windows per file
# - Each window has a binary label
```

### Example 5: Custom Label Aggregation

```python
# Load with different aggregation rules

# "any" - if ANY point is anomalous, sequence is anomalous
bundle_any = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    agg_rule="any"
)

# "all" - if ALL points are anomalous, sequence is anomalous
bundle_all = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    agg_rule="all"
)

# "ratio" - if ratio of anomalous points >= threshold, sequence is anomalous
bundle_ratio = loader.load(
    dataset_name="skab",
    base_dir=Path("datasets"),
    agg_rule="ratio",
    ratio_threshold=0.1  # 10% of points must be anomalous
)
```

---

## Implementation Details

### Key Classes

#### `AnomalySequenceArtifact`
Intermediate representation of a single anomaly sequence file.

**Attributes**:
- `sample_id: str` - Unique identifier (e.g., "valve1__1")
- `x: np.ndarray` - Feature matrix (T × C)
- `point_labels: np.ndarray` - Point-level labels (T,)
- `metadata: dict` - Provenance and processing info

**Methods**:
- `aggregate_label(rule, ratio_threshold)` - Aggregate point labels to sequence label
- `to_sequence_sample()` - Convert to TimeSeriesSample
- `to_window_samples()` - Generate windowed samples

#### `SKABAnomalySequenceLoader`
Loads SKAB data as sequence-level binary classification task.

**Key Parameters**:
- `dataset_name` - Dataset name (e.g., "skab")
- `base_dir` - Parent directory path
- `subdirs` - Optional list of subdirectories to load
- `agg_rule` - Label aggregation rule ("any", "all", "ratio")
- `train_ratio` - Training set ratio (default 0.5)

#### `SKABAnomalyWindowLoader`
Extends `SKABAnomalySequenceLoader` for window-level tasks.

**Additional Parameters**:
- `window_size` - Window size (default 60)
- `stride` - Sliding window stride (default 10)
- `rule` - Window label aggregation rule

### Data Flow

```
CSV File (e.g., valve1/1.csv)
         ↓
pd.read_csv(sep=";")
         ↓
Extract features and point labels
         ↓
Create AnomalySequenceArtifact(sample_id="valve1__1", x, point_labels)
         ↓
Aggregate point_labels → sequence_label (using agg_rule)
         ↓
Create TimeSeriesSample(sample_id="valve1__1", x, y=sequence_label)
         ↓
Train/test split via _split_samples()
         ↓
AnomalySequenceDatasetBundle(train_samples, test_samples, metadata)
```

### Subdirectory Filtering Algorithm

```python
# Step 1: Find all CSV files recursively
files = sorted(root.rglob("**/*.csv"))

# Step 2: Filter by first-level subdirectory (optional)
if subdirs is not None:
    subdirs_set = set(subdirs)
    
    filtered = []
    for f in files:
        # Compute relative path from root
        rel_path = f.parent.relative_to(root)
        
        # Extract first-level directory name
        if rel_path.parts:
            first_subdir = rel_path.parts[0]
            
            # Include if in specified subdirs
            if first_subdir in subdirs_set:
                filtered.append(f)
    
    files = filtered

# Step 3: Load each file
artifacts = [
    load_anomaly_sequence_artifact_from_csv(
        csv_path=csv_path,
        root_dir=root  # For ID generation
    )
    for csv_path in files
]
```

### Sample ID Generation Algorithm

```python
def generate_sample_id(csv_path, root_dir):
    # Input: Path("datasets/skab/valve1/1.csv"), Path("datasets/skab")
    
    # Step 1: Compute relative path
    relative_path = csv_path.relative_to(root_dir)
    # Result: Path("valve1/1.csv")
    
    # Step 2: Remove extension
    sample_id = str(relative_path.with_suffix(""))
    # Result: "valve1/1"
    
    # Step 3: Normalize path separators (Windows)
    sample_id = sample_id.replace("\\", "/")
    # Result: "valve1/1" (on Windows: "valve1\1" → "valve1/1")
    
    # Step 4: Replace "/" with "__"
    sample_id = sample_id.replace("/", "__")
    # Result: "valve1__1"
    
    return sample_id
```

---

## API Reference

### Module: `src/data/adapters/anomaly_adapter.py`

#### `load_anomaly_sequence_artifact_from_csv()`

```python
def load_anomaly_sequence_artifact_from_csv(
    csv_path: str | Path,
    dataset_name: str,
    label_col: str = "anomaly",
    time_col: Optional[str] = "datetime",
    drop_columns: Optional[list[str]] = None,
    root_dir: Optional[str | Path] = None,
) -> AnomalySequenceArtifact:
    """
    Load a single CSV file into an AnomalySequenceArtifact.
    
    Args:
        csv_path: Path to the CSV file
        dataset_name: Name of the dataset
        label_col: Column name for anomaly labels (default "anomaly")
        time_col: Column name for datetime (will be dropped)
        drop_columns: Additional columns to drop
        root_dir: Root directory for computing relative paths in sample_id
                  If provided, generates ID like "valve1__1"
                  If None, uses only filename stem
    
    Returns:
        AnomalySequenceArtifact with unique sample_id and parsed data
    
    Raises:
        KeyError: If label_col not found in CSV
        ValueError: If feature shape is invalid
    """
```

#### `load_anomaly_sequence_artifacts_from_dir()`

```python
def load_anomaly_sequence_artifacts_from_dir(
    base_dir: str | Path,
    dataset_name: str,
    label_col: str = "anomaly",
    time_col: Optional[str] = "datetime",
    csv_glob: str = "**/*.csv",
    drop_columns: Optional[list[str]] = None,
    max_files: Optional[int] = None,
    subdirs: Optional[list[str]] = None,
) -> list[AnomalySequenceArtifact]:
    """
    Load all anomaly sequence artifacts from a hierarchical directory.
    
    Supports both flat and hierarchical structures:
    - Flat: base_dir/dataset_name/*.csv
    - Hierarchical: base_dir/dataset_name/subdir/*.csv
    
    Args:
        base_dir: Parent directory (e.g., "datasets")
        dataset_name: Dataset name (e.g., "skab")
        label_col: Column name for anomaly labels
        time_col: Column name for datetime
        csv_glob: Glob pattern for CSV discovery (default "**/*.csv" for recursive)
        drop_columns: Additional columns to drop
        max_files: Maximum number of files to load (None = unlimited)
        subdirs: List of subdirectories to load. If None, loads all.
                 Example: ["valve1", "valve2"] loads only those directories
    
    Returns:
        List of AnomalySequenceArtifact objects with unique sample IDs
    
    Raises:
        FileNotFoundError: If base_dir/dataset_name doesn't exist or no CSV files found
    """
```

### Module: `src/data/loaders/anomaly_loader.py`

#### `SKABAnomalySequenceLoader.load()`

```python
def load(
    self,
    dataset_name: str,
    base_dir: str | Path,
    label_col: str = "anomaly",
    time_col: Optional[str] = "datetime",
    csv_glob: str = "**/*.csv",
    drop_columns: Optional[list[str]] = None,
    train_ratio: Optional[float] = None,
    max_files: Optional[int] = None,
    subdirs: Optional[list[str]] = None,
    **kwargs: Any,
) -> AnomalySequenceDatasetBundle:
    """
    Load SKAB anomaly sequence dataset.
    
    Args:
        dataset_name: Dataset name (e.g., "skab")
        base_dir: Parent directory containing dataset
        label_col: Column name for anomaly labels (default "anomaly")
        time_col: Column name for datetime (default "datetime")
        csv_glob: Glob pattern for finding CSV files (default "**/*.csv" for recursive)
        drop_columns: Additional columns to drop
        train_ratio: Ratio of training samples (default 0.5)
        max_files: Maximum number of files to load
        subdirs: List of subdirectories to load (e.g., ["valve1", "valve2"])
        **kwargs: Additional parameters:
            - agg_rule: Label aggregation rule ("any", "all", "ratio") (default "any")
            - ratio_threshold: Threshold for "ratio" rule (default 0.1)
    
    Returns:
        AnomalySequenceDatasetBundle with train and test splits
    
    Raises:
        FileNotFoundError: If dataset directory or CSV files not found
    """
```

#### `SKABAnomalyWindowLoader.load()`

```python
def load(
    self,
    dataset_name: str,
    base_dir: str | Path,
    **kwargs: Any
) -> AnomalyWindowDatasetBundle:
    """
    Load SKAB anomaly window dataset.
    
    Args:
        dataset_name: Dataset name (e.g., "skab")
        base_dir: Parent directory containing dataset
        **kwargs: Load options and window parameters:
            - window_size: Window size (default 60)
            - stride: Sliding window stride (default 10)
            - rule: Aggregation rule 'any'/'all'/'ratio' (default 'any')
            - ratio_threshold: Threshold for 'ratio' rule (default 0.1)
            - subdirs: List of subdirectories to load
            - Other parameters same as SKABAnomalySequenceLoader
    
    Returns:
        AnomalyWindowDatasetBundle with windowed samples
    """
```

---

## Quick Reference

### Common Parameter Combinations

```python
from data.loaders.anomaly_loader import SKABAnomalySequenceLoader, SKABAnomalyWindowLoader
from pathlib import Path

loader = SKABAnomalySequenceLoader()

# ✓ All data
bundle = loader.load(dataset_name="skab", base_dir=Path("datasets"))

# ✓ Only valve anomalies
bundle = loader.load(dataset_name="skab", base_dir=Path("datasets"), 
                     subdirs=["valve1", "valve2"])

# ✓ Only normal data
bundle = loader.load(dataset_name="skab", base_dir=Path("datasets"),
                     subdirs=["anomaly-free"])

# ✓ Custom aggregation
bundle = loader.load(dataset_name="skab", base_dir=Path("datasets"),
                     agg_rule="ratio", ratio_threshold=0.2)

# ✓ Windows
window_loader = SKABAnomalyWindowLoader()
bundle = window_loader.load(dataset_name="skab", base_dir=Path("datasets"),
                            window_size=120, stride=30)
```

### Debugging

```python
# Inspect sample IDs
all_samples = bundle.train.samples + bundle.test.samples
sample_ids = [s.sample_id for s in all_samples]
print(sample_ids)
# Output: ['valve1__1', 'valve2__3', 'anomaly-free__anomaly-free', ...]

# Recover original paths
from data.README import reverse_sample_id
original_paths = [reverse_sample_id(sid) for sid in sample_ids]
print(original_paths)
# Output: ['valve1/1', 'valve2/3', 'anomaly-free/anomaly-free', ...]

# Check metadata
sample = all_samples[0]
print(sample.metadata['source_file'])
# Output: /full/path/to/datasets/skab/valve1/1.csv
```

### Testing

```bash
# Run all anomaly loader tests
cd /data1/zx57/NTS_agent
conda run -n nts_agent python -m pytest tests/unit/data/test_anomaly_loader.py -v

# Run all data module tests
conda run -n nts_agent python -m pytest tests/unit/data/ -v
```

### Verification Checklist

- [ ] Create `datasets/skab/` directory structure
- [ ] Download/copy SKAB CSV files to respective subdirectories
- [ ] Verify CSV files use semicolon (`;`) delimiter
- [ ] Verify CSV columns include: `anomaly`, `Volume Flow RateRMS`, etc.
- [ ] Run tests: `pytest tests/unit/data/test_anomaly_loader.py -v`
- [ ] Verify test output shows "valve1__", "valve2__", "anomaly-free__" in sample IDs

---

## Design Decisions

### Why Recursive Glob?
SKAB has multiple subdirectories with organized experimental data. Recursive globbing enables automatic discovery without requiring users to specify directory structures.

### Why Subdirectory Filtering?
Different research questions may require different anomaly subsets. Users may want to:
- Study valve-specific failures (subdirs=["valve1", "valve2"])
- Compare normal vs. anomalous (subdirs=["anomaly-free"] vs. subdirs=["other"])
- Isolate specific experimental conditions (subdirs=["valve1", "other"])

### Why `__` for Path Separator?
- `/` has system-level meaning and causes issues across platforms
- `_` alone is too common in folder names (e.g., "valve_1_test")
- `__` (double underscore) is a distinctive separator with low collision risk
- `__` can be directly used in filenames, JSON, databases without escaping

### Why Preserve Metadata?
The original file path is stored in `metadata['source_file']` for:
- Debugging and tracing back to source
- Reproducibility of experiments
- Organizing results by anomaly type post-hoc

---

## Future Extensions

1. **Configuration File Support** - Define train/test splits via JSON/YAML config
2. **Dataset Registry** - Predefined patterns for common anomaly types
3. **Multi-Dataset Integration** - Support for NAB, SMAP/MSL, TSB-AD, etc.
4. **Metadata Management** - Enhanced tracking of anomaly characteristics
5. **Balanced Sampling** - Ensure equal representation of all anomaly types

---

## Related Files

| File | Purpose |
|------|---------|
| `src/data/adapters/anomaly_adapter.py` | CSV parsing and artifact generation |
| `src/data/loaders/anomaly_loader.py` | Sequence and window-level loaders |
| `src/data/schemas.py` | Bundle and split data structures |
| `tests/unit/data/test_anomaly_loader.py` | Comprehensive tests |
| `tests/unit/data/test_dataset_registry.py` | Registry tests |

---

## Contact & Support

For issues, questions, or contributions related to the anomaly loading framework:
- Check existing tests in `tests/unit/data/test_anomaly_loader.py`
- Review docstrings in `src/data/adapters/anomaly_adapter.py`
- Examine usage examples in this README
