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
├── hpo_analysis.py          # HPO analysis notebook (.py with #%% cells)
├── mia_analysis.py          # MIA privacy analysis notebook
├── outputs/                 # Generated analysis outputs (git-ignored)
│   ├── hpo_runs.csv        # Processed HPO run data
│   ├── param_metric_correlations.csv
│   ├── metric_metric_correlations.csv
│   ├── best_runs_summary.txt
│   └── mia/                # MIA analysis outputs
│       ├── feature_importance.png
│       ├── threshold_attacks.png
│       ├── feature_distributions.png
│       └── mia_summary.txt
└── utils/
    ├── __init__.py
    ├── wandb_fetcher.py     # Data loading utilities (wandb + local)
    ├── mia.py               # MIA evaluation classes
    └── mia_utils.py         # Config loading utilities
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

## MIA Privacy Analysis

The analysis module includes **Membership Inference Attack (MIA)** evaluation for assessing model privacy.

### What is MIA?

Membership inference attacks attempt to determine whether a specific sample was used to train a model. If an attacker can reliably distinguish training samples from test samples, it indicates the model has "memorized" training data, which is a privacy concern.

### Running MIA Analysis

**In VS Code** (recommended):
```
1. Open analysis/mia_analysis.py
2. Edit RUN_DIR to point to your trained model's output directory
3. Use "Run Cell" (Ctrl+Enter) to execute each #%% cell interactively
```

**As a script**:
```bash
cd /path/to/project
python -m analysis.mia_analysis
```

### Configuration

Edit the configuration section in `mia_analysis.py`:

```python
# Data source: "local" or "wandb"
DATA_SOURCE = "local"

# --- LOCAL SETTINGS ---
RUN_DIR = "outputs/classification_2024_01_15"  # Path to trained model

# --- MIA SETTINGS ---
MIA_FEATURES = {
    "max_prob": True,       # Maximum class probability
    "entropy": True,        # Prediction entropy
    "correct_prob": True,   # Probability of correct class
    "loss": True,           # Cross-entropy loss
    "margin": True,         # Difference between top two probabilities
    "modified_entropy": True,  # Normalized confidence
}
```

### MIA Features

The attack uses features derived from model class probabilities p(c|x):

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `max_prob` | `probs.max(dim=-1)` | Models are more confident on training data |
| `entropy` | `-sum(p * log(p))` | Lower entropy = more confident |
| `correct_prob` | `probs[i, labels[i]]` | Higher for seen samples |
| `loss` | `-log(correct_prob)` | Lower loss on training data |
| `margin` | `top_prob - second_prob` | Larger margin = more confident |
| `modified_entropy` | `1 - entropy / log(num_classes)` | Normalized confidence |

### Privacy Assessment Thresholds

| AUC-ROC | Assessment |
|---------|------------|
| < 0.55 | Excellent privacy preservation |
| 0.55 - 0.60 | Good privacy |
| 0.60 - 0.70 | Moderate leakage |
| >= 0.70 | Significant leakage |

### Output Files

After running MIA analysis, check `analysis/outputs/mia/`:

| File | Description |
|------|-------------|
| `feature_importance.png` | Bar chart of attack model coefficients |
| `threshold_attacks.png` | Per-feature AUC-ROC comparison |
| `feature_distributions.png` | Train vs test histograms |
| `mia_summary.txt` | Text summary with all metrics |

### API Reference

```python
from analysis.utils import MIAEvaluation, MIAFeatureConfig, load_run_config, find_model_checkpoint

# Load config and checkpoint
cfg = load_run_config("outputs/my_run")
checkpoint = find_model_checkpoint("outputs/my_run")
bornmachine = BornMachine.load(str(checkpoint))

# Setup data
datahandler = DataHandler(cfg.dataset)
datahandler.load()
datahandler.split_and_rescale(bornmachine)
datahandler.get_classification_loaders()

# Run MIA evaluation
feature_config = MIAFeatureConfig(max_prob=True, entropy=True, ...)
mia_eval = MIAEvaluation(feature_config=feature_config)
results = mia_eval.evaluate(
    bornmachine,
    datahandler.classification["train"],
    datahandler.classification["test"],
    device
)

# Access results
print(f"Attack AUC-ROC: {results.auc_roc}")
print(f"Privacy: {results.privacy_assessment()}")
print(results.summary())
```

---

## Adding New Analyses

1. Create a new `.py` file with `# %%` cells
2. Import utilities:
   ```python
   from analysis.utils import WandbFetcher, load_local_hpo_runs
   ```
3. Follow the pattern in `hpo_analysis.py`
4. Add documentation to this GUIDE.md
