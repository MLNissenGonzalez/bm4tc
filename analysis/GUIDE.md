# Analysis Module Guide

This directory contains post-experiment analysis tools for visualizing and understanding HPO experiment results.

## Data Sources

The analysis module supports **two data sources**:

### 1. Weights & Biases (wandb) - Remote

Fetches run data via the wandb API. This is the **default** and recommended approach when:
- You have internet access
- Your experiments logged to wandb
- You want the most complete metric data

**How it works:**
```
wandb API → fetch_hpo_runs() → DataFrame
```

### 2. Local outputs/ folder - Offline

Loads data directly from Hydra multirun output directories. Use this when:
- You don't have wandb access
- You want to analyze experiments that didn't log to wandb
- You're working offline

**How it works:**
```
outputs/experiment_name/
├── 0/.hydra/config.yaml     → extract config
├── 0/wandb/*/files/wandb-summary.json  → extract metrics
├── 1/.hydra/config.yaml
└── ...

load_local_hpo_runs() → DataFrame
```

**Note:** Local loading requires wandb summary files to be present. If wandb was disabled during training, only config data will be available.

## Module Structure

```
analysis/
├── GUIDE.md                 # This file
├── __init__.py
├── hpo_analysis.py          # Main analysis notebook (.py with #%% cells)
├── outputs/                 # Generated analysis outputs (git-ignored)
│   ├── hpo_runs.csv        # Processed HPO run data
│   ├── param_metric_correlations.csv
│   ├── metric_metric_correlations.csv
│   └── best_runs_summary.txt
└── utils/
    ├── __init__.py
    └── wandb_fetcher.py     # Data loading utilities (wandb + local)
```

## Quick Start

### 1. Configure the Analysis

Edit the configuration section at the top of `hpo_analysis.py`:

```python
# Data source: "wandb" or "local"
DATA_SOURCE = "wandb"  # Change to "local" to load from outputs/ folder

# --- WANDB SETTINGS (used if DATA_SOURCE == "wandb") ---
WANDB_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
WANDB_PROJECT = "gan_train"
EXPERIMENT_PATTERN = "lrwdbs_hpo"  # Regex pattern to match run groups
DATASET_NAME = None  # e.g., "spirals_4k", or None for all

# --- LOCAL SETTINGS (used if DATA_SOURCE == "local") ---
LOCAL_SWEEP_DIR = "outputs/lrwdbs_hpo_spirals_4k_22Jan26"
```

### 2. Run the Analysis

**In VS Code** (recommended):
```
1. Open analysis/hpo_analysis.py
2. Use "Run Cell" (Ctrl+Enter) to execute each #%% cell interactively
```

**As a script**:
```bash
cd /path/to/project
python -m analysis.hpo_analysis
```

**In Jupyter** (convert first):
```bash
pip install jupytext
jupytext --to notebook analysis/hpo_analysis.py
jupyter notebook analysis/hpo_analysis.ipynb
```

## Analysis Features

### 1. Parameter vs Metric Scatter Plots
Shows how each optimized parameter (lr, weight_decay, batch_size) correlates with each metric.
- Log scale automatically applied to lr and weight_decay
- Includes trend line

### 2. Parameter Distribution Histograms
Shows the distribution of parameter values, colored by metric performance.
Helps identify which parameter regions performed best.

### 3. Surface Plots (LR × Weight Decay)
3D surface and 2D contour plots showing metric landscapes:
- Validation accuracy
- Validation loss
- Robust accuracy (if available)

### 4. Best Runs Summary
Lists the top N runs for each metric with their hyperparameters.

### 5. Parameter-Metric Correlations
Heatmap showing Pearson correlations between (log-transformed) parameters and metrics.

### 6. Metric-Metric Correlations (NEW)
Shows how different metrics correlate with each other:
- Heatmap of all metric pairs
- Scatter matrix with correlation coefficients
- Useful for understanding trade-offs (e.g., accuracy vs robustness)

## API Reference

### WandbFetcher Class

```python
from analysis.utils import WandbFetcher

fetcher = WandbFetcher(entity="my-entity", project="my-project")

# Fetch runs with filters
runs = fetcher.fetch_runs(filters={
    "group": {"$regex": ".*hpo.*"},
    "state": "finished",
})

# Convert to DataFrame
df = fetcher.runs_to_dataframe(runs)
```

### Convenience Functions

```python
from analysis.utils import fetch_hpo_runs, load_local_hpo_runs, find_local_sweep_dirs

# From wandb
df = fetch_hpo_runs(
    entity="my-entity",
    project="gan_train",
    experiment_pattern="lrwdbs_hpo",
    dataset_name="spirals_4k",  # optional
)

# From local directory
df = load_local_hpo_runs("outputs/lrwdbs_hpo_spirals_4k_22Jan26")

# Find all sweep directories
sweeps = find_local_sweep_dirs(outputs_dir="outputs", pattern="*hpo*")
```

### Common Wandb Filters

```python
# By experiment group (exact)
filters = {"group": "lrwdbs_hpo_spirals_4k_22Jan26"}

# By group pattern (regex)
filters = {"group": {"$regex": ".*hpo.*"}}

# By state
filters = {"state": "finished"}

# By config value
filters = {"config.dataset.name": "spirals_4k"}

# Combined (AND)
filters = {
    "$and": [
        {"state": "finished"},
        {"group": {"$regex": ".*hpo.*"}},
    ]
}
```

## DataFrame Column Naming

After loading, columns are prefixed consistently:

| Type | Prefix | Example |
|------|--------|---------|
| Config values | `config/` | `config/lr`, `config/batch_size` |
| Summary metrics | `summary/` | `summary/pre/valid/acc` |
| Run metadata | none | `run_id`, `run_name`, `state` |

## Generated Outputs

After running the analysis, check `analysis/outputs/`:

| File | Description |
|------|-------------|
| `hpo_runs.csv` | All fetched runs with configs and metrics |
| `param_metric_correlations.csv` | Parameter-metric correlation matrix |
| `metric_metric_correlations.csv` | Metric-metric correlation matrix |
| `best_runs_summary.txt` | Human-readable summary of best runs |

Figures are saved to `analysis/`:

| File | Description |
|------|-------------|
| `param_metric_scatter.png` | Grid of parameter vs metric scatter plots |
| `param_hist_*.png` | Histograms colored by metric |
| `surface_lr_wd_*.png` | 3D surface plots |
| `contour_lr_wd_*.png` | 2D contour plots |
| `param_metric_correlations.png` | Parameter-metric correlation heatmap |
| `metric_metric_correlations.png` | Metric-metric correlation heatmap |
| `metric_scatter_matrix.png` | Scatter matrix of key metrics |

## Troubleshooting

### "No runs found" (wandb)
- Check `WANDB_ENTITY` and `WANDB_PROJECT` are correct
- Verify the filter pattern matches your experiment groups
- Run `wandb login` if not authenticated
- Check internet connection

### "No runs found" (local)
- Verify `LOCAL_SWEEP_DIR` path exists
- Check that directories have `.hydra/config.yaml` files
- Check that directories have numbered subdirectories (0, 1, 2, ...)

### "Column not found"
- W&B metric names depend on your experiment code
- Use `print(df.columns)` to see available columns
- Update `VALIDATION_METRICS`, `ROBUSTNESS_METRICS` lists accordingly

### Missing metrics in local loading
- Local loading only finds metrics if wandb-summary.json exists
- If wandb was disabled, only config data will be available

### Sparse surface plots
- HPO with few trials may not cover the parameter space well
- Try `method="nearest"` instead of `method="linear"` in surface plots
- Reduce `grid_resolution` for smoother interpolation

## Dependencies

Required Python packages:
- `wandb` - W&B API access (optional if using local only)
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `scipy` - Interpolation for surface plots
- `numpy` - Numerical operations
- `pyyaml` - Config file loading (for local)
- `omegaconf` - Hydra config loading (optional, for local)

## Adding New Analyses

1. Create a new `.py` file with `# %%` cells
2. Import utilities:
   ```python
   from analysis.utils import WandbFetcher, load_local_hpo_runs
   ```
3. Follow the pattern in `hpo_analysis.py`
4. Add documentation to this GUIDE.md
