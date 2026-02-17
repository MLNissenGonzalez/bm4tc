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
├── sweep_analysis.py        # Post-hoc sweep analysis notebook (loads models, recomputes metrics)
├── hpo_analysis.py          # HPO analysis notebook (.py with #%% cells)
├── mia_analysis.py          # MIA privacy analysis notebook
├── uq_analysis.py           # UQ analysis notebook (detection + purification)
├── visualize_distributions.py  # Distribution visualization notebook
├── config_fetcher.ipynb     # Jupyter notebook for fetching best configs
├── outputs/                 # Generated analysis outputs (git-ignored)
│   ├── <sweep_name>/       # Per-sweep output directory
│   │   ├── accuracy_histogram.png
│   │   ├── mean_accuracies_errorbars.png
│   │   ├── metric_correlations.png
│   │   ├── pareto_acc_vs_rob_*.png
│   │   ├── summary_statistics.csv
│   │   ├── sweep_analysis_summary.txt
│   │   ├── best_run_distributions.png
│   │   └── best_run_samples.png
│   ├── mia/                # MIA analysis outputs
│   │   ├── feature_importance.png
│   │   ├── threshold_attacks.png
│   │   ├── feature_distributions.png
│   │   └── mia_summary.txt
│   ├── uq/                  # UQ analysis outputs
│   │   ├── log_likelihood_histogram.png
│   │   ├── detection_rate_vs_threshold.png
│   │   ├── purification_accuracy_heatmap.png
│   │   ├── purification_before_after.png
│   │   └── uq_summary.txt
│   └── distributions/      # Distribution visualization outputs
│       └── distributions.png
└── utils/
    ├── __init__.py
    ├── wandb_fetcher.py     # Data loading utilities (wandb + local)
    ├── resolve.py           # HPO regime/param resolver
    ├── statistics.py        # Shared stats & visualization functions
    ├── evaluate.py          # Post-hoc per-model evaluation
    ├── mia.py               # MIA evaluation classes
    ├── uq.py                # UQ evaluation classes
    └── mia_utils.py         # Config loading utilities
```

## Quick Start

### 1. Configure the Analysis

Edit the configuration section at the top of `hpo_analysis.py`:

```python
# Data source: "wandb" or "local"
DATA_SOURCE = "local"  # Change to "wandb" to fetch from Weights & Biases

# --- LOCAL SETTINGS ---
LOCAL_SWEEP_DIR = "outputs/lrwd_hpo_cls_d30D18fourier_spirals_4k_2201"

# --- HPO CONFIGURATION (simplified) ---
# Training regime: "pre", "gen", "adv", "gan"
REGIME = "pre"

# Parameters to analyze (shorthand names)
PARAM_SHORTHANDS = ["lr", "weight-decay", "batch-size"]

# Auto-detect varied params (excludes single-value params)
AUTO_DETECT_VARIED = True
```

That's it! The resolver automatically:
- Maps shorthand names to full config paths
- Sets up metric columns for the regime
- Auto-detects robustness metric strengths
- Filters out parameters that didn't vary in the sweep

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

## Training Regimes

| Regime | Description | Config Prefix | Typical HPO Params |
|--------|-------------|---------------|-------------------|
| `pre` | Classification pretraining | `trainer.classification.*` | lr, weight-decay, batch-size |
| `gen` | Generative NLL training | `trainer.generative.*` | lr, weight-decay, batch-size |
| `adv` | Adversarial training | `trainer.adversarial.*` | lr, weight-decay, epsilon, trades-beta |
| `gan` | GAN-style training | `trainer.ganstyle.*` | lr, critic-lr, r-real |

## Parameter Shorthand Reference

| Shorthand | Aliases | Description |
|-----------|---------|-------------|
| `lr` | `learning-rate`, `learning_rate` | Learning rate |
| `weight-decay` | `wd`, `weight_decay` | Weight decay |
| `batch-size` | `bs`, `batch_size` | Batch size |
| `bond-dim` | `bond_dim` | MPS bond dimension |
| `in-dim` | `in_dim` | MPS input dimension |
| `seed` | - | Random seed |
| `data-seed` | `data_seed` | Dataset generation seed |
| `epsilon` | `strength`, `eps` | Adversarial perturbation (adv only) |
| `trades-beta` | `trades_beta`, `beta` | TRADES trade-off (adv only) |
| `clean-weight` | `clean_weight` | Clean example weight (adv only) |
| `critic-lr` | `critic_lr` | Critic learning rate (gan only) |
| `critic-wd` | `critic_wd` | Critic weight decay (gan only) |
| `r-real` | `r_real` | Real/synthetic ratio (gan only) |
| `num-spc` | `num_spc` | Samples per class (gan only) |
| `num-bins` | `num_bins` | Sampling bins (gan only) |

## Resolver Output

When you run the analysis, the resolver prints a summary:

```
============================================================
Resolved Configuration
============================================================

Regime: pre (Classification pretraining)

Parameters (3 varied, 2 excluded):
  + lr               -> trainer.classification.optimizer.kwargs.lr
  + weight-decay     -> trainer.classification.optimizer.kwargs.weight_decay
  + batch-size       -> trainer.classification.batch_size
  - bond-dim         (excluded: single value)
  - in-dim           (excluded: single value)

Metrics:
  Validation: acc, loss
  Robustness: rob/0.1, rob/0.3 (auto-detected)
  Test:       acc, loss

Pretrained Model: None detected
============================================================
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

### 6. Metric-Metric Correlations
Shows how different metrics correlate with each other:
- Heatmap of all metric pairs
- Scatter matrix with correlation coefficients
- Useful for understanding trade-offs (e.g., accuracy vs robustness)

## API Reference

### Resolver Functions

```python
from analysis.utils import (
    resolve_params,
    resolve_metrics,
    filter_varied_params,
    format_resolved_config,
    config_path_to_column,
    get_available_params,
    get_available_regimes,
)

# Resolve parameter shorthand to full config paths
params = resolve_params("pre", ["lr", "weight-decay"])
# Returns: {"lr": "trainer.classification.optimizer.kwargs.lr", ...}

# Get metrics for regime (auto-detects robustness strengths from df)
metrics = resolve_metrics("pre", df)
# Returns: {"validation": [...], "robustness": [...], "test": [...]}

# Convert full path to DataFrame column name
col = config_path_to_column("trainer.classification.optimizer.kwargs.lr")
# Returns: "config/lr"

# Filter to only params that varied in the sweep
varied, excluded = filter_varied_params(df, ["config/lr", "config/weight_decay"])

# Get available params/regimes
params = get_available_params("adv")  # ["lr", "weight-decay", "epsilon", ...]
regimes = get_available_regimes()  # ["pre", "gen", "adv", "gan"]

# Format resolved config for display
print(format_resolved_config(regime, params, varied, excluded, metrics))
```

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
    experiment_pattern="lrwd_hpo",
    dataset_name="spirals_4k",  # optional
)

# From local directory
df = load_local_hpo_runs("outputs/lrwd_hpo_cls_d30D18fourier_spirals_4k_2201")

# Find all sweep directories
sweeps = find_local_sweep_dirs(outputs_dir="outputs", pattern="*hpo*")
```

### Common Wandb Filters

```python
# By experiment group (exact)
filters = {"group": "lrwd_hpo_cls_d30D18fourier_spirals_4k_2201"}

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

## Uncertainty Quantification (UQ) Analysis

The analysis module includes **UQ evaluation** for assessing adversarial defense using the Born Machine's marginal likelihood p(x).

### What is UQ for Born Machines?

Born Machines learn the joint distribution p(x,c), enabling computation of the marginal input likelihood p(x) = sum_c p(x,c). This provides two defense mechanisms:

1. **Detection**: Reject inputs whose log p(x) falls below a threshold (calibrated from clean data percentiles)
2. **Purification**: For rejected inputs, gradient-descend on NLL within a perturbation ball to find a nearby high-likelihood point, then classify that instead

### Running UQ Analysis

**In VS Code** (recommended):
```
1. Open analysis/uq_analysis.py
2. Edit RUN_DIR to point to your trained model's output directory
3. Use "Run Cell" (Ctrl+Enter) to execute each #%% cell interactively
```

**As a script**:
```bash
cd /path/to/project
python -m analysis.uq_analysis
```

### Configuration

Edit the configuration section in `uq_analysis.py`:

```python
# Data source: "local" or "wandb"
DATA_SOURCE = "local"
RUN_DIR = "outputs/classification_example"

# Purification parameters
PURIFICATION_NORM = "inf"
PURIFICATION_NUM_STEPS = 20
PURIFICATION_RADII = [0.05, 0.1, 0.15, 0.2, 0.3]

# Detection thresholds (percentiles of clean log p(x))
THRESHOLD_PERCENTILES = [1, 5, 10, 20]

# Attack parameters
ATTACK_METHOD = "PGD"
ATTACK_STRENGTHS = [0.1, 0.2, 0.3]
ATTACK_NUM_STEPS = 20
```

### Output Files

After running UQ analysis, check `analysis/outputs/uq/`:

| File | Description |
|------|-------------|
| `log_likelihood_histogram.png` | Clean vs adversarial log p(x) distributions with threshold lines |
| `detection_rate_vs_threshold.png` | Detection rate curves for each attack epsilon |
| `purification_accuracy_heatmap.png` | Accuracy heatmap: rows=epsilon, cols=radius |
| `purification_before_after.png` | Mean log p(x) before/after purification |
| `uq_summary.txt` | Text summary with all metrics |

### Sweep Integration

UQ evaluation can be included in sweep analysis by setting `COMPUTE_UQ = True` in `sweep_analysis.py`:

```python
COMPUTE_UQ = True
UQ_CONFIG = {
    "norm": "inf",
    "num_steps": 20,
    "radii": [0.1, 0.2, 0.3],
    "percentiles": [1, 5, 10, 20],
    "attack_method": "PGD",
    "attack_strengths": [0.1, 0.2, 0.3],
}
```

### API Reference

```python
from analysis.utils import UQEvaluation, UQConfig

uq_config = UQConfig(
    norm="inf",
    num_steps=20,
    radii=[0.1, 0.2, 0.3],
    percentiles=[1, 5, 10, 20],
    attack_method="PGD",
    attack_strengths=[0.1, 0.2, 0.3],
)

uq_eval = UQEvaluation(config=uq_config)
results = uq_eval.evaluate(bornmachine, test_loader, device)

# Access results
print(f"Clean accuracy: {results.clean_accuracy}")
print(f"Detection rates: {results.detection_rates}")
print(results.summary())
```

### Debugging: UQ Pipeline

```python
# 1. Verify log p(x) computation
bm = BornMachine.load("path/to/checkpoint.pt")
bm.to(device)
bm.cache_log_Z()
print(f"log Z = {bm._log_Z}")

data = datahandler.data["test"].to(device)
log_px = bm.marginal_log_probability(data)
print(f"log p(x) range: [{log_px.min():.2f}, {log_px.max():.2f}]")
assert torch.isfinite(log_px).all()
# Note: log p(x) CAN be positive — p(x) is a density, not a probability mass

# 2. Verify purification improves likelihood
from src.utils.purification.minimal import LikelihoodPurification
purifier = LikelihoodPurification(norm="inf", num_steps=20)
noisy = data[:10] + 0.1 * torch.randn_like(data[:10])
noisy = noisy.clamp(*bm.input_range)
log_px_before = bm.marginal_log_probability(noisy).detach()
purified, log_px_after = purifier.purify(bm, noisy, radius=0.15, device=device)
print(f"Mean log p(x) before: {log_px_before.mean():.2f}")
print(f"Mean log p(x) after:  {log_px_after.mean():.2f}")

# 3. Verify detection thresholds
from analysis.utils import compute_thresholds
thresholds, clean_log_px = compute_thresholds(bm, test_loader, [1, 5, 10], device)
print(f"Thresholds: {thresholds}")
```

---

## Distribution Visualization

The analysis module includes **distribution visualization** for inspecting the learned probability distributions p(c|x) and p(x,c) of a trained BornMachine over the 2D input space.

### What it Shows

- **Row 1:** p(c|x) conditional class probability heatmaps per class + decision boundary (argmax)
- **Row 2:** p(x,c) joint probability heatmaps per class + marginal p(x) = sum_c p(x,c)
- **Data overlay:** Training data points overlaid on all subplots for verification

### Running Distribution Visualization

**In VS Code** (recommended):
```
1. Open analysis/visualize_distributions.py
2. Edit RUN_DIR to point to your trained model's output directory
3. Use "Run Cell" (Ctrl+Enter) to execute each #%% cell interactively
```

**As a script**:
```bash
cd /path/to/project
python -m analysis.visualize_distributions
```

### Configuration

Edit the configuration section in `visualize_distributions.py`:

```python
RUN_DIR = "outputs/classification_example"  # Path to trained model
RESOLUTION = 150       # Grid resolution (150x150 = 22,500 points)
NORMALIZE_JOINT = True # Normalize p(x,c) by partition function
SHOW_DATA = True       # Overlay training data points
DEVICE = "cuda"        # or "cpu"
SAVE_DIR = "analysis/outputs/distributions/"
```

### Programmatic API

```python
from analysis.visualize_distributions import (
    visualize_from_run_dir,
    make_grid,
    compute_conditional_probs,
    compute_joint_probs,
    plot_distributions,
)

# High-level: one call does everything
fig = visualize_from_run_dir(
    run_dir="outputs/my_run",
    resolution=150,
    normalize_joint=True,
    show_data=True,
    device="cuda",
    save_dir="analysis/outputs/distributions/",
)

# Or use individual functions for more control
grid_x1, grid_x2, grid_points = make_grid(input_range=(0, 1), resolution=150)
conditional = compute_conditional_probs(bm, grid_points, device)
joint = compute_joint_probs(bm, grid_points, device, normalize=True)
fig = plot_distributions(conditional, joint, grid_x1, grid_x2,
                         train_data=data, train_labels=labels)
```

### Output Files

After running, check `analysis/outputs/distributions/`:

| File | Description |
|------|-------------|
| `distributions.png` | Combined heatmap figure with conditional, joint, and decision boundary |

---

## Post-Hoc Sweep Analysis

The analysis module includes **sweep_analysis.py** — a notebook that **loads each saved model** from a sweep directory and **recomputes metrics post-hoc**. This supersedes the old `run_statistics.py`.

### Why Post-Hoc?

- **Compute metrics that weren't tracked during training** (e.g., FID, generative loss)
- **Ensure data splits match** each run's config (correct seeds via `dataset.overwrite = True`)
- **Consistent evaluation** across all runs (e.g., same attack strengths via evasion override)
- **Sanity check** post-hoc metrics against W&B summary metrics from training

### What it Shows

1. **Per-model evaluation** — loads each checkpoint and computes acc, rob, MIA, cls loss, gen loss, FID
2. **Accuracy histogram** — distribution of clean, robust, and MIA accuracy
3. **Mean + std bar plot** — error bars show standard deviation (not stderr)
4. **Metric-metric correlation heatmap**
5. **Accuracy vs stopping criterion scatter**
6. **Pareto frontiers** — clean acc vs rob acc, MIA acc vs rob acc
7. **Summary statistics table** — best (from best run by stop criterion), mean, std, stderr
8. **Sanity check** — compare post-hoc metrics with W&B summary metrics
9. **Distribution visualization** — for best model (heatmaps + generated samples)
10. **Summary export** — text file with best model, stats table, Pareto-optimal runs

### Running Sweep Analysis

**In VS Code** (recommended):
```
1. Open analysis/sweep_analysis.py
2. Configure SWEEP_DIR, REGIME, metric toggles, overrides
3. Use "Run Cell" (Ctrl+Enter) to execute each #%% cell interactively
```

**As a script**:
```bash
cd /path/to/project
python -m analysis.sweep_analysis
```

### Configuration

```python
SWEEP_DIR = "outputs/seed_sweep_adv_d30D18fourier_moons_4k_1202"
REGIME = "adv"

# Metric toggles
COMPUTE_ACC = True
COMPUTE_ROB = True
COMPUTE_MIA = True
COMPUTE_CLS_LOSS = False
COMPUTE_GEN_LOSS = False
COMPUTE_FID = False

# Override evasion config for consistency across all runs
EVASION_OVERRIDE = {"method": "PGD", "strengths": [0.05, 0.1], "num_steps": 20}
# Or None to use each run's own evasion config

EVAL_SPLITS = ["test"]
DEVICE = "cuda"
```

### Shared Utility Modules

Functions used by both `sweep_analysis.py` and `hpo_analysis.py` are in `analysis/utils/`:

- **`statistics.py`**: `compute_statistics`, `get_best_run`, `create_summary_table`, `compute_pareto_frontier`, `plot_pareto_frontier`, `get_pareto_runs`, `compute_metric_correlations`, `plot_correlation_heatmap`, `plot_accuracy_histogram`, `plot_mean_with_std`, `plot_scatter_vs_metric`, `plot_accuracy_vs_strength`, `plot_accuracy_vs_strength_band`, `clean_column_name`
- **`evaluate.py`**: `EvalConfig`, `evaluate_run`, `evaluate_sweep`, `resolve_stop_criterion`

### Programmatic API

```python
from analysis.utils import EvalConfig, evaluate_sweep

eval_cfg = EvalConfig(
    compute_acc=True, compute_rob=True, compute_mia=True,
    splits=["test"], device="cuda",
    evasion_override={"method": "PGD", "strengths": [0.1]},
)

df = evaluate_sweep("outputs/my_sweep", eval_cfg, config_keys=["tracking.seed"])
```

---

## Best Classification Config Fetching

### `get_best_classification_config()`

Fetches the best classification run from W&B and returns a dict with everything needed to reproduce the pretraining or set up downstream training.

```python
from analysis.utils import get_best_classification_config

config = get_best_classification_config(
    entity="my-entity",
    project="gan_train",
    group="sanity_check_cls_d30D18fourier_moons_4k_2701"
)
```

**Return structure:**

| Key | Type | Description |
|-----|------|-------------|
| `run_id` | `str` | W&B run ID |
| `run_name` | `str` | W&B run name |
| `group` | `str` | W&B group name |
| `cls_config` | `dict` | Classification trainer config (optimizer, criterion, etc.) |
| `born_config` | `dict` | Born machine config (init_kwargs, embedding) |
| `dataset_config` | `dict` | Dataset config (name, split_seed, split, gen_dow_kwargs) |
| `tracking_seed` | `int` | Tracking seed used in the run |
| `metrics` | `dict` | Summary metrics (valid_acc, valid_loss, test_acc) |

### `extract_dataset_config()`

Extracts dataset configuration from a W&B run object.

```python
from analysis.utils import extract_dataset_config

run = fetch_best_classification_run(...)
ds_config = extract_dataset_config(run)
# Returns: {"name": "moons_4k", "split_seed": 11, "split": [0.5, 0.25, 0.25],
#           "gen_dow_kwargs": {"seed": 25, "name": "moons_4k", "size": 4000, "noise": 0.05}}
```

### `print_classification_config_yaml()`

Prints the best classification config in copy-pasteable YAML format, including `tracking`, `dataset`, `trainer.classification`, and `born` sections.

```python
from analysis.utils import print_classification_config_yaml

print_classification_config_yaml(
    entity="my-entity",
    project="gan_train",
    group="sanity_check_cls_d30D18fourier_moons_4k_2701"
)
```

## Adversarial HPO with Pretrained Models

### Workflow

To avoid re-running classification pretraining for every adversarial HPO trial:

1. **Pretrain once** using the standard classification experiment, save the model.
2. **Fetch the best config** using `print_classification_config_yaml()` to verify the split seed and born config.
3. **Run adversarial HPO** using `adversarial/fourier_d30D18/hpo/moons.yaml`, which loads the pretrained model:

```bash
python -m experiments.adversarial --multirun \
    +experiments=adversarial/fourier_d30D18/hpo/moons \
    model_path=/path/to/pretrained/best_cls_loss_moons_4k.pt
```

### Split Reproducibility

The data split is governed by `dataset.split_seed` (passed as `random_state` to sklearn's `train_test_split`), which is independent of `tracking.seed`. As long as both pretraining and adversarial HPO use the same dataset config (same `split_seed`, `split` ratios, and data file), the train/valid/test split is identical.

The `adversarial/fourier_d30D18/hpo/moons.yaml` config explicitly sets `dataset.split_seed: 11` to match the default in `moons_4k.yaml`.

### Key Differences from `hpo.yaml`

| Setting | `hpo.yaml` (old) | `adversarial/fourier_d30D18/hpo/moons.yaml` (new) |
|---------|-------------------|------------------------|
| Classification trainer | `adam500_loss` | `null` (skipped) |
| Model source | Created from scratch | `model_path` (hardcoded or override) |
| `tracking.seed` | Swept (`range(1, 1000)`) | Fixed (`71`) |
| `dataset.split_seed` | Inherited from dataset config | Explicitly set to `11` |

---

## Adding New Analyses

1. Create a new `.py` file with `# %%` cells
2. Import utilities:
   ```python
   from analysis.utils import WandbFetcher, load_local_hpo_runs
   ```
3. Follow the pattern in `hpo_analysis.py`
4. Add documentation to this GUIDE.md
