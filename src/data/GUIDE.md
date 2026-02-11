# Data Module Guide

This module handles dataset generation, loading, preprocessing, and providing data loaders for training.

## Module Structure

```
src/data/
├── GUIDE.md              # This file
├── __init__.py           # Exports: DataHandler
├── handler.py            # DataHandler: main data management class
└── gen_n_load.py         # Dataset generation and loading utilities
```

## DataHandler (`handler.py`)

Central class for managing all data operations in the pipeline.

### Lifecycle

```
1. __init__(cfg)          # Create handler with config
2. load()                 # Load/generate raw data
3. split_and_rescale(bm)  # Split and normalize to embedding range
4. get_classification_loaders()  # Create loaders for classifier training
5. get_discrimination_loaders()  # Create loaders for GAN training
```

### Initialization

```python
from src.data import DataHandler

datahandler = DataHandler(cfg.dataset)
```

**Key Attributes After Init:**
- `cfg`: DatasetConfig
- `data`, `labels`: None (until `load()`)
- `classification`, `discrimination`: None (until respective getter)
- `data_dim`, `num_cls`, `total_size`: None (until `load()`)

### Loading Data

```python
datahandler.load()
# Populates: data, labels, data_dim, num_cls, total_size
```

This calls `gen_n_load.load_dataset()` which:
1. Checks if dataset exists in `.datasets/`
2. If not, generates/downloads it
3. Returns `LabelledDataset` dataclass

### Split and Rescale

```python
datahandler.split_and_rescale(bornmachine, scaler_name="minmax")
```

**Steps:**
1. Get input range from BornMachine's embedding
2. Split data into train/valid/test (using `cfg.split` ratios)
3. Fit scaler on training data only (avoid data leakage!)
4. Transform all splits to embedding range
5. Compute per-class statistics (mean, covariance) if `data_dim < 100`

**After this call:**
```python
datahandler.data["train"]   # torch.FloatTensor
datahandler.data["valid"]   # torch.FloatTensor
datahandler.data["test"]    # torch.FloatTensor
datahandler.labels["train"] # torch.LongTensor
datahandler.means           # List[torch.Tensor] per class
datahandler.covs            # List[torch.Tensor] per class
datahandler.input_range     # (0.0, 1.0) for Fourier
```

### Classification Loaders

```python
datahandler.get_classification_loaders()

# Access loaders:
for data, labels in datahandler.classification["train"]:
    # data: (batch_size, data_dim)
    # labels: (batch_size,)
```

**Properties:**
- Batch size: 64 (hardcoded)
- Train: shuffled, drop_last=True
- Valid/Test: no shuffle, no drop_last

### Discrimination Loaders

```python
datahandler.get_discrimination_loaders(batch_size=16)

# Access loaders:
for naturals in datahandler.discrimination["train"]:
    # naturals: (batch_size, num_classes, data_dim)
```

**Purpose:** For GAN training, data is organized by class (samples per class).

**Properties:**
- Batch size: configurable
- Data shape: `(num_spc, num_classes, data_dim)`

## Dataset Generation (`gen_n_load.py`)

### Supported Datasets

Currently implemented 2D toy datasets:

| Name | Description | Generator |
|------|-------------|-----------|
| `moons` | Two interleaving half circles | `sklearn.datasets.make_moons` |
| `circles` | Concentric circles | `sklearn.datasets.make_circles` |
| `spirals` | Two spirals | Custom implementation |

### Dataset Configuration

```yaml
# configs/dataset/moons_4k.yaml
name: "moons_4k"
gen_dow_kwargs:
  name: "moons_4k"
  size: 2000           # Per class
  seed: 42             # Data generation seed (independent of tracking.seed)
  noise: 0.1           # Noise level
  circ_factor: null    # Only for circles
  dow_link: null       # For downloads
split: [0.7, 0.15, 0.15]
split_seed: 42         # Train/valid/test split seed (independent of tracking.seed)
overwrite: false       # Set to true to regenerate dataset on each run
```

**Seed independence**: `gen_dow_kwargs.seed` and `split_seed` are fully independent of
`tracking.seed`. Changing `tracking.seed` does not affect data generation or splits —
it only affects training randomness (model init, DataLoader shuffling, PGD, sampling).

**Note on `overwrite`**: When `overwrite: true`, the dataset is regenerated even if it already exists on disk. This is useful for seed sweep experiments where each trial needs fresh data with different random seeds.

### Loading Process

```python
def load_dataset(cfg: DatasetConfig) -> LabelledDataset:
    # 1. Parse dataset name
    canonical, variant = _parse_dataset_name(cfg.gen_dow_kwargs.name)
    # e.g., "moons_4k" -> ("moons", "moons_4k")

    # 2. Check if exists
    dataset_file = f".datasets/{canonical}/{variant}.npz"

    # 3. Generate if missing
    if not os.path.exists(dataset_file):
        _generate_or_download(cfg.gen_dow_kwargs, path)

    # 4. Load and return
    data = np.load(dataset_file)
    return LabelledDataset(...)
```

### Adding a New Dataset

**For 2D Toy Data:**

1. Add to canonical list:
```python
_TWO_DIM_DATA = ["moons", "circles", "spirals", "mynewdata"]
```

2. Implement generator:
```python
def _two_dim_generator(cfg):
    ...
    elif canonical == "mynewdata":
        X, t = generate_my_data(cfg.size, cfg.noise)
        return X, t
```

3. Create config file:
```yaml
# configs/dataset/mynewdata_1k.yaml
name: "mynewdata_1k"
gen_dow_kwargs:
  name: "mynewdata_1k"
  size: 500
  seed: 42
  noise: 0.05
  circ_factor: null
  dow_link: null
split: [0.7, 0.15, 0.15]
split_seed: 42
```

**For Downloaded Datasets (MNIST, etc.):**

1. Add to canonical list:
```python
_NIST_DATA = ["mnist", "fashionmnist"]
```

2. Implement download/preprocess in `_generate_or_download()`:
```python
elif canonical in _NIST_DATA:
    download_and_preprocess_mnist(cfg, path)
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  .datasets/moons/moons_4k.npz                                   │
│        │                                                         │
│        ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ DataHandler.load()                                           │
│  │ data: (N, D) numpy                                           │
│  │ labels: (N,) numpy                                           │
│  └────────┬────────┘                                            │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────┐           │
│  │ DataHandler.split_and_rescale(bm)               │           │
│  │                                                  │           │
│  │ 1. train_test_split (70/15/15)                  │           │
│  │ 2. MinMaxScaler fit on train                    │           │
│  │ 3. Transform to [0, 1] (for Fourier)            │           │
│  │ 4. Convert to torch tensors                     │           │
│  │ 5. Compute class statistics                     │           │
│  │                                                  │           │
│  │ data["train"]: (N_train, D) torch.FloatTensor   │           │
│  │ labels["train"]: (N_train,) torch.LongTensor    │           │
│  └────────┬────────┴────────────┬──────────────────┘           │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌────────────────┐    ┌──────────────────────┐                │
│  │ Classification │    │ Discrimination        │                │
│  │ Loaders        │    │ Loaders              │                │
│  │                │    │                       │                │
│  │ (B, D), (B,)   │    │ (B, C, D)            │                │
│  │ data, labels   │    │ naturals per class   │                │
│  └────────────────┘    └──────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Attributes Reference

| Attribute | Type | When Set | Description |
|-----------|------|----------|-------------|
| `cfg` | `DatasetConfig` | `__init__` | Dataset configuration |
| `data` | `ndarray` → `dict` | `load()` → `split_and_rescale()` | Raw → split data |
| `labels` | `ndarray` → `dict` | `load()` → `split_and_rescale()` | Raw → split labels |
| `data_dim` | `int` | `load()` | Number of features |
| `num_cls` | `int` | `load()` | Number of classes |
| `total_size` | `int` | `load()` | Total number of samples |
| `input_range` | `tuple` | `split_and_rescale()` | `(min, max)` for embedding |
| `scaler` | `MinMaxScaler` | `split_and_rescale()` | Fitted scaler |
| `means` | `List[Tensor]` | `split_and_rescale()` | Per-class means |
| `covs` | `List[Tensor]` | `split_and_rescale()` | Per-class covariances |
| `classification` | `dict` | `get_classification_loaders()` | DataLoaders |
| `discrimination` | `dict` | `get_discrimination_loaders()` | DataLoaders |
| `num_spc` | `List[int]` | `get_discrimination_loaders()` | Samples per class per split |

## What Breaks If You Change...

| Change | Impact |
|--------|--------|
| Split ratios | Different train/valid/test sizes |
| Scaler type | Data range may not match embedding |
| Batch size | Memory usage, gradient estimates |
| Data loading order | Reproducibility may break |
| `classified_data` deletion | Memory leak if not deleted |

## Common Issues

1. **Data not scaled correctly**: Check `input_range` matches embedding
   ```python
   print(f"Data range: [{data.min()}, {data.max()}]")
   print(f"Expected range: {datahandler.input_range}")
   ```

2. **Wrong number of features**: `n_features = data_dim + 1` (includes output tensor position)

3. **Labels not sequential**: Labels must be `0, 1, ..., num_cls-1`

4. **Memory with large datasets**: `classified_data` is deleted after `get_discrimination_loaders()`

## File Reference

| File | Lines | Key Classes/Functions |
|------|-------|----------------------|
| `handler.py` | ~195 | `DataHandler` |
| `gen_n_load.py` | ~275 | `LabelledDataset`, `load_dataset`, `_two_dim_generator` |
