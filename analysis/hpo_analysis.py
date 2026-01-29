# %% [markdown]
# # HPO Experiment Analysis
#
# This notebook analyzes hyperparameter optimization results.
#
# **Data Sources:**
# - **wandb**: Fetches run data from Weights & Biases API
# - **local**: Loads data from local outputs/ directory (Hydra multirun)
#
# **Visualizations:**
# 1. Scatter plots: Each parameter vs each metric
# 2. Parameter distribution histograms colored by metric
# 3. Contour plots: Parameter pairs vs metrics (interpolated from point evaluations)
# 4. Marginal importance: Which parameters matter most
# 5. Parameter interaction effects: How parameter pairs jointly affect metrics
# 6. Pareto frontier: Trade-off between accuracy and robustness
# 7. Metric-metric correlations: How metrics relate to each other
# 8. Parameter-metric correlations: How parameters affect metrics

# %% [markdown]
# ## Setup and Configuration

# %%
import sys
from pathlib import Path

# Handle both script and interactive execution
if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    # Interactive/notebook mode - assume we're in analysis/
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

# Check for command line argument to override LOCAL_SWEEP_DIR
_CLI_SWEEP_DIR = None
if len(sys.argv) > 1:
    _CLI_SWEEP_DIR = sys.argv[1]
    print(f"Using sweep dir from command line: {_CLI_SWEEP_DIR}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import warnings

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION FOR YOUR EXPERIMENT
# =============================================================================

# Data source: "wandb" or "local"
DATA_SOURCE = "wandb"  # Change to "local" to load from outputs/ folder

# --- WANDB SETTINGS (used if DATA_SOURCE == "wandb") ---
WANDB_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
WANDB_PROJECT = "gan_train"
EXPERIMENT_PATTERN = "generative_hpo"  # Regex pattern to match run groups
DATASET_NAME = "circles_4k"  # e.g., "spirals_4k", "moons_4k", or None for all

# --- LOCAL SETTINGS (used if DATA_SOURCE == "local") ---
# Path to sweep directory relative to project root
# Can be overridden by command line argument: python -m analysis.hpo_analysis outputs/sweep_name
LOCAL_SWEEP_DIR = _CLI_SWEEP_DIR if _CLI_SWEEP_DIR else "outputs/lrwdbs_hpo_spirals_4k_22Jan26"

# =============================================================================
# HPO CONFIGURATION (simplified interface)
# =============================================================================

# Training regime: "pre", "gen", "adv", "gan"
#   - "pre": Classification pretraining (trainer.classification.*)
#   - "gen": Generative NLL training (trainer.generative.*)
#   - "adv": Adversarial training (trainer.adversarial.*)
#   - "gan": GAN-style training (trainer.ganstyle.*)
REGIME = "gen"

# Parameter shorthand names to analyze
# Common: "lr", "weight-decay" (or "wd"), "batch-size" (or "bs"), "bond-dim", "in-dim", "seed"
# For "adv": also "epsilon", "trades-beta", "clean-weight"
# For "gan": also "critic-lr", "r-real", "num-spc"
# Set to None to use regime defaults
PARAM_SHORTHANDS = ["lr", "weight-decay", "batch-size"]

# Auto-detect varied parameters: only analyze params that actually varied in the sweep
# Non-varied params are automatically excluded from analysis
AUTO_DETECT_VARIED = True

# =============================================================================
# ADVANCED SETTINGS (usually don't need to change)
# =============================================================================

# Primary metric for importance analysis (None = auto from regime: validation accuracy)
PRIMARY_METRIC = None

# Whether to minimize primary metric (False for acc, True for loss)
PRIMARY_METRIC_MINIMIZE = False

# Parameter pairs for interaction analysis
# None = auto from first two varied params
# Or specify as list of tuples: [("lr", "weight-decay")]
INTERACTION_PAIRS = None

# Pareto frontier metrics
# None = auto (validation accuracy vs first robustness metric)
# Or specify as: [(metric_col, maximize), (metric_col, maximize)]
PARETO_METRICS = None

# Plot settings
FIGSIZE = (10, 6)
DPI = 100

# =============================================================================
# RESOLVED SETTINGS (auto-populated from resolver)
# =============================================================================
# These are set by the resolver below. Only override if you need manual control.
HPO_PARAMS = None  # Will be set by resolver
VALIDATION_METRICS = None  # Will be set by resolver
ROBUSTNESS_METRICS = None  # Will be set by resolver
TEST_METRICS = None  # Will be set by resolver

# %% [markdown]
# ## Load Data
#
# Data loading is regime-aware: it uses the REGIME setting to determine which
# config paths to extract from wandb/local runs.

# %%
def load_data(source: str = DATA_SOURCE, regime: str = REGIME) -> pd.DataFrame:
    """
    Load HPO experiment data from configured source.

    Args:
        source: "wandb" or "local"
        regime: Training regime to determine which config keys to extract

    Returns:
        DataFrame with config parameters and summary metrics
    """
    # Get regime-specific config keys
    from analysis.utils import resolve_params, REGIME_PARAM_MAP

    # Get all config keys for this regime
    if regime in REGIME_PARAM_MAP:
        config_keys = list(REGIME_PARAM_MAP[regime].values())
        # Add common keys that aren't regime-specific
        config_keys.extend([
            "dataset.name",
            "experiment",
        ])
    else:
        config_keys = None  # Fall back to defaults

    if source == "wandb":
        from analysis.utils import WandbFetcher

        print(f"Fetching runs from wandb: {WANDB_ENTITY}/{WANDB_PROJECT}")
        print(f"Experiment pattern: {EXPERIMENT_PATTERN}")
        print(f"Regime: {regime} (using regime-specific config keys)")
        if DATASET_NAME:
            print(f"Dataset filter: {DATASET_NAME}")

        fetcher = WandbFetcher(entity=WANDB_ENTITY, project=WANDB_PROJECT)

        # Build filter
        filters = {"state": "finished"}
        if "*" in EXPERIMENT_PATTERN or "." in EXPERIMENT_PATTERN:
            filters["group"] = {"$regex": EXPERIMENT_PATTERN}
        else:
            filters["group"] = {"$regex": f".*{EXPERIMENT_PATTERN}.*"}
        if DATASET_NAME:
            filters["config.dataset.name"] = DATASET_NAME

        runs = fetcher.fetch_runs(filters=filters)

        if not runs:
            print(f"No runs found matching pattern: {EXPERIMENT_PATTERN}")
            return pd.DataFrame()

        print(f"Found {len(runs)} runs")
        df = fetcher.runs_to_dataframe(runs, config_keys=config_keys)

    elif source == "local":
        from analysis.utils import load_local_hpo_runs

        sweep_path = project_root / LOCAL_SWEEP_DIR
        print(f"Loading runs from local directory: {sweep_path}")
        print(f"Regime: {regime} (using regime-specific config keys)")

        df = load_local_hpo_runs(sweep_path, config_keys=config_keys)

    else:
        raise ValueError(f"Unknown data source: {source}. Use 'wandb' or 'local'.")

    if df.empty:
        print("\nNo data loaded. Check your configuration.")
        print(f"  - DATA_SOURCE: {source}")
        if source == "wandb":
            print(f"  - WANDB_ENTITY: {WANDB_ENTITY}")
            print(f"  - WANDB_PROJECT: {WANDB_PROJECT}")
            print(f"  - EXPERIMENT_PATTERN: {EXPERIMENT_PATTERN}")
        else:
            print(f"  - LOCAL_SWEEP_DIR: {LOCAL_SWEEP_DIR}")

    return df


# %%
# Load the data
print("=" * 60)
print(f"Loading data from: {DATA_SOURCE}")
print("=" * 60)
df = load_data()

# %%
# Show data summary and filter to finished runs
if not df.empty:
    print("\n=== Data Summary ===")
    print(f"Total runs: {len(df)}")

    if "state" in df.columns:
        print(f"Finished runs: {len(df[df['state'] == 'finished'])}")
        df = df[df["state"] == "finished"].copy()

    print(f"\nConfig columns:")
    config_cols = [c for c in df.columns if c.startswith("config/")]
    for col in config_cols:
        unique_vals = df[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        if unique_vals <= 5:
            print(f"    Values: {sorted(df[col].dropna().unique())}")

    print(f"\nSummary metric columns:")
    summary_cols = [c for c in df.columns if c.startswith("summary/")]
    for col in sorted(summary_cols)[:20]:
        valid = df[col].notna().sum()
        if valid > 0:
            print(f"  {col}: {valid} values, range [{df[col].min():.4f}, {df[col].max():.4f}]")
    if len(summary_cols) > 20:
        print(f"  ... and {len(summary_cols) - 20} more")

# %% [markdown]
# ## Resolve Configuration
#
# Convert shorthand regime/param names to full config paths and metric names.
# Auto-detect which parameters actually varied in the sweep.

# %%
from analysis.utils import (
    resolve_params,
    resolve_metrics,
    filter_varied_params,
    format_resolved_config,
    config_path_to_column,
    detect_pretrained_info,
)

# Resolve params and metrics based on regime
_resolved_params = resolve_params(REGIME, PARAM_SHORTHANDS)
_resolved_metrics = resolve_metrics(REGIME, df if not df.empty else None)

# Get DataFrame column names for params
_param_columns = [config_path_to_column(p) for p in _resolved_params.values()]

# Create reverse mapping: column name -> shorthand
_col_to_short = {config_path_to_column(v): k for k, v in _resolved_params.items()}

# Filter to varied params if enabled
if AUTO_DETECT_VARIED and not df.empty:
    _varied_cols, _excluded_cols = filter_varied_params(df, _param_columns)
    _varied_shorthands = [_col_to_short.get(c, c) for c in _varied_cols]
    _excluded_shorthands = [_col_to_short.get(c, c) for c in _excluded_cols]
else:
    _varied_cols = _param_columns
    _excluded_cols = []
    _varied_shorthands = list(_resolved_params.keys())
    _excluded_shorthands = []

# Check for pretrained model info
_pretrained_info = detect_pretrained_info(df) if not df.empty else None

# Display resolved configuration
print(format_resolved_config(
    regime=REGIME,
    params=_resolved_params,
    varied_params=_varied_shorthands,
    excluded_params=_excluded_shorthands,
    metrics=_resolved_metrics,
    pretrained_info=_pretrained_info,
))

# %%
# Set global variables for rest of analysis (if not manually overridden)
if HPO_PARAMS is None:
    HPO_PARAMS = _varied_cols

if VALIDATION_METRICS is None:
    VALIDATION_METRICS = _resolved_metrics["validation"]

if ROBUSTNESS_METRICS is None:
    ROBUSTNESS_METRICS = _resolved_metrics["robustness"]

if TEST_METRICS is None:
    TEST_METRICS = _resolved_metrics["test"]

# Auto-set primary metric if not specified
if PRIMARY_METRIC is None:
    PRIMARY_METRIC = VALIDATION_METRICS[0] if VALIDATION_METRICS else None

# Auto-set interaction pairs if not specified (use column names, not shorthands)
if INTERACTION_PAIRS is None and len(_varied_cols) >= 2:
    INTERACTION_PAIRS = [(_varied_cols[0], _varied_cols[1])]
elif INTERACTION_PAIRS is not None:
    # Convert shorthand pairs to column names if needed
    _new_pairs = []
    for p1, p2 in INTERACTION_PAIRS:
        # Check if it's a shorthand or already a column name
        c1 = config_path_to_column(_resolved_params.get(p1, "")) if p1 in _resolved_params else p1
        c2 = config_path_to_column(_resolved_params.get(p2, "")) if p2 in _resolved_params else p2
        _new_pairs.append((c1, c2))
    INTERACTION_PAIRS = _new_pairs

# Auto-set Pareto metrics if not specified
if PARETO_METRICS is None:
    _acc_col = next((m for m in VALIDATION_METRICS if "acc" in m), None)
    _rob_col = ROBUSTNESS_METRICS[0] if ROBUSTNESS_METRICS else None
    if _acc_col and _rob_col:
        PARETO_METRICS = [(_acc_col, True), (_rob_col, True)]
    elif _acc_col:
        PARETO_METRICS = []  # Can't do Pareto with only one metric

print(f"\nAnalysis will use:")
print(f"  HPO_PARAMS: {HPO_PARAMS}")
print(f"  PRIMARY_METRIC: {PRIMARY_METRIC}")
print(f"  INTERACTION_PAIRS: {INTERACTION_PAIRS}")
print(f"  PARETO_METRICS: {PARETO_METRICS}")

# %% [markdown]
# ## Utility Functions

# %%
def get_available_columns(df: pd.DataFrame, candidates: list, verbose: bool = True) -> list:
    """
    Return subset of candidate columns that exist in dataframe.
    Tries to find similar columns if exact match not found.
    """
    if df.empty:
        return []

    available = []
    seen_warnings = set()

    for col in candidates:
        if col in df.columns:
            available.append(col)
        else:
            # Try finding similar column by last part of name
            target = col.split("/")[-1]
            similar = [c for c in df.columns if target in c and c not in available]
            if similar:
                best_match = similar[0]
                if verbose and col not in seen_warnings:
                    print(f"Note: '{col}' not found, using '{best_match}'")
                    seen_warnings.add(col)
                available.append(best_match)

    return available


def clean_column_name(col: str) -> str:
    """Convert column name to readable label."""
    name = col.replace("config/", "").replace("summary/", "")
    name = name.replace("_", " ").replace("/", " / ")
    return name.title()


def is_log_scale_param(param: str) -> bool:
    """Check if parameter should be plotted on log scale."""
    log_params = ["lr", "weight_decay", "learning_rate"]
    return any(lp in param.lower() for lp in log_params)


# %% [markdown]
# ## Setup Output Directory
#
# All outputs (images, best results txt) go to outputs/sweep_name/.

# %%
# Determine sweep name from LOCAL_SWEEP_DIR or DATASET_NAME
if DATA_SOURCE == "local":
    sweep_name = Path(LOCAL_SWEEP_DIR).name
else:
    sweep_name = DATASET_NAME if DATASET_NAME else EXPERIMENT_PATTERN
sweep_name = sweep_name.replace("/", "_")  # sanitize for filenames

# Create output directory inside outputs/sweep_name/
output_dir = project_root / "outputs" / sweep_name
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {output_dir}")

# %% [markdown]
# ## 1. Parameter vs Metric Scatter Plots
#
# For each optimized parameter, show scatter plots against each validation metric.

# %%
def plot_param_vs_metric_scatter(
    df: pd.DataFrame,
    param_col: str,
    metric_col: str,
    ax: plt.Axes = None,
    log_x: bool = False,
    color: str = "steelblue",
    alpha: float = 0.6,
) -> plt.Axes:
    """Create scatter plot of parameter vs metric with trend line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Get valid data
    valid_mask = df[param_col].notna() & df[metric_col].notna()
    x = df.loc[valid_mask, param_col].astype(float)
    y = df.loc[valid_mask, metric_col].astype(float)

    if len(x) == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return ax

    # Handle zero/negative values for log scale
    if log_x and (x <= 0).any():
        x = x[x > 0]
        y = y[df.loc[valid_mask, param_col].astype(float) > 0]

    if len(x) == 0:
        ax.text(0.5, 0.5, "No valid data for log scale", ha="center", va="center", transform=ax.transAxes)
        return ax

    ax.scatter(x, y, alpha=alpha, c=color, edgecolors="white", linewidth=0.5, s=50)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(clean_column_name(param_col))
    ax.set_ylabel(clean_column_name(metric_col))

    # Add trend line
    try:
        if log_x and len(x) > 2:
            z = np.polyfit(np.log10(x), y, 1)
            x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            y_line = z[0] * np.log10(x_line) + z[1]
        elif len(x) > 2:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = z[0] * x_line + z[1]
        else:
            x_line, y_line = None, None

        if x_line is not None:
            ax.plot(x_line, y_line, "r--", alpha=0.7, label="Trend")
            ax.legend(loc="best")
    except Exception:
        pass

    ax.grid(True, alpha=0.3)
    return ax


def plot_all_param_metric_combinations(
    df: pd.DataFrame,
    params: list,
    metrics: list,
) -> plt.Figure:
    """Create grid of scatter plots for all parameter-metric combinations."""
    params = get_available_columns(df, params, verbose=False)
    metrics = get_available_columns(df, metrics, verbose=False)

    if not params or not metrics:
        print("No valid parameter or metric columns found")
        return None

    n_params = len(params)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_metrics, n_params,
        figsize=(5 * n_params, 4 * n_metrics),
        dpi=DPI,
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, param in enumerate(params):
            ax = axes[i, j]
            log_x = is_log_scale_param(param)
            plot_param_vs_metric_scatter(df, param, metric, ax=ax, log_x=log_x)

            if i == 0:
                ax.set_title(clean_column_name(param))

    fig.tight_layout()
    return fig


# %%
# Create scatter plots
if not df.empty:
    all_metrics = VALIDATION_METRICS + ROBUSTNESS_METRICS
    print("\n=== Creating Parameter vs Metric Scatter Plots ===")

    fig = plot_all_param_metric_combinations(df, params=HPO_PARAMS, metrics=all_metrics)
    if fig:
        output_path = output_dir / "param_metric_scatter.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.show()

# %% [markdown]
# ## 2. Parameter Distribution Histograms

# %%
def plot_param_histogram_by_metric(
    df: pd.DataFrame,
    param_col: str,
    metric_col: str,
    n_bins: int = 15,
    ax: plt.Axes = None,
    log_x: bool = False,
) -> plt.Axes:
    """Create histogram of parameter colored by mean metric value in each bin."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    valid_mask = df[param_col].notna() & df[metric_col].notna()
    x = df.loc[valid_mask, param_col].astype(float).values
    y = df.loc[valid_mask, metric_col].astype(float).values

    if len(x) == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return ax

    # Handle log scale
    if log_x:
        x_positive = x > 0
        x = x[x_positive]
        y = y[x_positive]
        if len(x) == 0:
            ax.text(0.5, 0.5, "No positive values", ha="center", va="center", transform=ax.transAxes)
            return ax
        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    else:
        bins = np.linspace(x.min(), x.max(), n_bins + 1)

    # Compute mean metric per bin
    bin_indices = np.digitize(x, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means[i] = y[mask].mean()
            bin_counts[i] = mask.sum()

    # Color by metric value
    valid_means = bin_means[~np.isnan(bin_means)]
    if len(valid_means) > 0:
        norm = plt.Normalize(valid_means.min(), valid_means.max())
        colors = [cm.viridis(norm(m)) if not np.isnan(m) else "lightgray" for m in bin_means]
    else:
        colors = "steelblue"

    # Plot
    bar_positions = bins[:-1]
    bar_widths = np.diff(bins)
    ax.bar(bar_positions, bin_counts, width=bar_widths, align="edge",
           color=colors, edgecolor="white", linewidth=0.5)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(clean_column_name(param_col))
    ax.set_ylabel("Count")

    # Colorbar
    if len(valid_means) > 0:
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(clean_column_name(metric_col))

    ax.grid(True, alpha=0.3, axis="y")
    return ax


# %%
# Create histograms
if not df.empty:
    print("\n=== Creating Parameter Histograms ===")
    available_metrics = get_available_columns(df, VALIDATION_METRICS + ROBUSTNESS_METRICS, verbose=False)

    for metric in available_metrics[:2]:  # Just first 2 metrics to avoid too many plots
        params = get_available_columns(df, HPO_PARAMS, verbose=False)
        if not params:
            continue

        fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4), dpi=DPI, squeeze=False)

        for j, param in enumerate(params):
            ax = axes[0, j]
            log_x = is_log_scale_param(param)
            plot_param_histogram_by_metric(df, param, metric, ax=ax, log_x=log_x)
            ax.set_title(clean_column_name(param))

        fig.suptitle(f"Parameter Distributions (colored by {clean_column_name(metric)})", y=1.02)
        fig.tight_layout()

        metric_name = metric.split("/")[-1].replace(" ", "_")
        output_path = output_dir / f"param_hist_{metric_name}.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.show()

# %% [markdown]
# ## 3. Contour Plots: LR × Weight Decay vs Metrics
#
# **How the contour plot is created from point evaluations:**
#
# HPO experiments produce discrete point evaluations: each run samples a specific
# (lr, weight_decay) pair and measures the resulting metric. To visualize the
# underlying response surface, we use **scipy.interpolate.griddata** with linear
# interpolation:
#
# 1. **Log transformation**: LR and weight_decay are transformed to log10 space
#    since they typically span several orders of magnitude.
#
# 2. **Grid creation**: A regular 50×50 grid is created spanning the min/max of
#    the observed parameter values in log space.
#
# 3. **Linear interpolation**: For each grid point, the metric value is estimated
#    by triangulating the scattered data points (Delaunay triangulation) and
#    performing barycentric interpolation within the enclosing triangle.
#
# 4. **Extrapolation**: Grid points outside the convex hull of the data are
#    marked as NaN (shown as white regions in the plot).
#
# The red dots show the actual HPO evaluations; the colored contours show the
# interpolated surface. Regions with sparse sampling may be less accurate.

# %%
def create_contour_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    log_x: bool = True,
    log_y: bool = True,
    grid_resolution: int = 50,
    method: str = "linear",
    title: str = None,
    levels: int = 20,
) -> tuple:
    """Create 2D contour plot from scattered data."""
    valid_mask = df[x_col].notna() & df[y_col].notna() & df[z_col].notna()
    x = df.loc[valid_mask, x_col].astype(float).values
    y = df.loc[valid_mask, y_col].astype(float).values
    z = df.loc[valid_mask, z_col].astype(float).values

    if len(x) < 4:
        print(f"Not enough data points for contour plot ({len(x)} points)")
        return None, None

    if log_x or log_y:
        pos_mask = (x > 0) & (y > 0) if (log_x and log_y) else (x > 0 if log_x else y > 0)
        x, y, z = x[pos_mask], y[pos_mask], z[pos_mask]

    if len(x) < 4:
        return None, None

    x_t = np.log10(x) if log_x else x
    y_t = np.log10(y) if log_y else y

    xi = np.linspace(x_t.min(), x_t.max(), grid_resolution)
    yi = np.linspace(y_t.min(), y_t.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zi_grid = griddata((x_t, y_t), z, (xi_grid, yi_grid), method=method, fill_value=np.nan)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    contour = ax.contourf(xi_grid, yi_grid, zi_grid, levels=levels, cmap="viridis")
    ax.scatter(x_t, y_t, c="red", s=20, alpha=0.6, edgecolors="white", linewidth=0.5)

    ax.set_xlabel(clean_column_name(x_col) + (" (log10)" if log_x else ""))
    ax.set_ylabel(clean_column_name(y_col) + (" (log10)" if log_y else ""))

    if title:
        ax.set_title(title)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(clean_column_name(z_col))
    ax.grid(True, alpha=0.3)

    return fig, ax


# %%
# Create contour plots for LR × Weight Decay
if not df.empty:
    print("\n=== Creating Contour Plots ===")

    lr_cols = get_available_columns(df, ["config/lr"], verbose=False)
    wd_cols = get_available_columns(df, ["config/weight_decay"], verbose=False)

    if lr_cols and wd_cols:
        lr_col, wd_col = lr_cols[0], wd_cols[0]
        metrics_to_plot = get_available_columns(df, VALIDATION_METRICS + ROBUSTNESS_METRICS, verbose=False)

        for metric_col in metrics_to_plot:
            metric_name = metric_col.split("/")[-1]
            title = f"{clean_column_name(metric_col)} vs LR and Weight Decay"

            fig, ax = create_contour_plot(df, lr_col, wd_col, metric_col, title=title)
            if fig:
                output_path = output_dir / f"contour_lr_wd_{metric_name}.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Saved contour plot to: {output_path}")
                plt.show()
    else:
        print("Could not find LR and/or weight_decay columns")
        print(f"Available config columns: {[c for c in df.columns if c.startswith('config/')]}")

# %% [markdown]
# ## 4. Marginal Importance Analysis
#
# Marginal importance quantifies how much each parameter affects the primary metric,
# averaged over all other parameter values. This is computed by:
#
# 1. Binning each parameter into groups (using quantiles for continuous params)
# 2. Computing the mean metric value within each bin
# 3. Measuring the variance of these bin means (higher variance = more important)
#
# The importance score is normalized so that all parameters sum to 1.

# %%
def compute_marginal_importance(
    df: pd.DataFrame,
    params: list,
    metric_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute marginal importance of each parameter for the given metric.

    Returns DataFrame with columns: parameter, importance, mean_range, bin_means
    """
    params = get_available_columns(df, params, verbose=False)
    metric_cols = get_available_columns(df, [metric_col], verbose=False)

    if not params or not metric_cols:
        return pd.DataFrame()

    metric_col = metric_cols[0]
    results = []

    for param in params:
        valid_mask = df[param].notna() & df[metric_col].notna()
        if valid_mask.sum() < n_bins:
            continue

        x = df.loc[valid_mask, param].astype(float).values
        y = df.loc[valid_mask, metric_col].astype(float).values

        # Use log scale for lr/weight_decay
        if is_log_scale_param(param) and (x > 0).all():
            x = np.log10(x)

        # Bin by quantiles to handle non-uniform sampling
        try:
            bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
            # Make edges unique
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                continue
            bin_indices = np.digitize(x, bin_edges[1:-1])
        except Exception:
            continue

        # Compute mean metric per bin
        bin_means = []
        for i in range(len(bin_edges) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(y[mask].mean())

        if len(bin_means) < 2:
            continue

        bin_means = np.array(bin_means)
        importance = np.var(bin_means)  # Variance of bin means
        mean_range = bin_means.max() - bin_means.min()

        results.append({
            "parameter": param,
            "importance_raw": importance,
            "mean_range": mean_range,
            "bin_means": bin_means,
            "n_bins_actual": len(bin_means),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Normalize importance to sum to 1
    total = result_df["importance_raw"].sum()
    if total > 0:
        result_df["importance"] = result_df["importance_raw"] / total
    else:
        result_df["importance"] = 0

    return result_df.sort_values("importance", ascending=False)


def plot_marginal_importance(
    importance_df: pd.DataFrame,
    metric_name: str,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot marginal importance as horizontal bar chart."""
    if importance_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=DPI)

    # Left: Importance bar chart
    ax = axes[0]
    params = [clean_column_name(p) for p in importance_df["parameter"]]
    importance = importance_df["importance"].values

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(params)))
    bars = ax.barh(params, importance, color=colors, edgecolor="white")
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"Marginal Importance for {clean_column_name(metric_name)}")
    ax.invert_yaxis()

    # Add percentage labels
    for bar, imp in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{imp * 100:.1f}%", va="center", fontsize=9)

    ax.set_xlim(0, max(importance) * 1.2)
    ax.grid(True, alpha=0.3, axis="x")

    # Right: Mean range (practical effect size)
    ax = axes[1]
    mean_range = importance_df["mean_range"].values

    bars = ax.barh(params, mean_range, color=colors, edgecolor="white")
    ax.set_xlabel(f"Range of Mean {clean_column_name(metric_name)}")
    ax.set_title("Effect Size (Max - Min Bin Mean)")
    ax.invert_yaxis()

    for bar, mr in zip(bars, mean_range):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{mr:.4f}", va="center", fontsize=9)

    ax.set_xlim(0, max(mean_range) * 1.2)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


# %%
# Compute and plot marginal importance
if not df.empty:
    print("\n=== Marginal Importance Analysis ===")

    metric_cols = get_available_columns(df, [PRIMARY_METRIC], verbose=True)
    if metric_cols:
        metric_col = metric_cols[0]
        importance_df = compute_marginal_importance(df, HPO_PARAMS, metric_col)

        if not importance_df.empty:
            print(f"\nParameter importance for {clean_column_name(metric_col)}:")
            print(importance_df[["parameter", "importance", "mean_range", "n_bins_actual"]].to_string(index=False))

            fig = plot_marginal_importance(importance_df, metric_col)
            if fig:
                output_path = output_dir / "marginal_importance.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"\nSaved to: {output_path}")
                plt.show()
        else:
            print("Could not compute marginal importance (insufficient data)")
    else:
        print(f"Primary metric '{PRIMARY_METRIC}' not found in data")

# %% [markdown]
# ## 5. Parameter Interaction Effects
#
# Interaction effects show how pairs of parameters jointly affect the metric.
# A strong interaction means the effect of one parameter depends on the value
# of another (the surface is "twisted" rather than additive).
#
# We visualize this with contour plots for each configured parameter pair.
# The red dots show actual evaluations; regions without dots are interpolated.

# %%
def plot_interaction_effect(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric_col: str,
    grid_resolution: int = 30,
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Plot interaction effect between two parameters on a metric.

    Returns (fig, interaction_strength) where interaction_strength is a measure
    of how non-additive the effects are.
    """
    cols = get_available_columns(df, [param1, param2, metric_col], verbose=False)
    if len(cols) < 3:
        return None, None

    param1, param2, metric_col = cols[0], cols[1], cols[2]

    valid_mask = df[param1].notna() & df[param2].notna() & df[metric_col].notna()
    x = df.loc[valid_mask, param1].astype(float).values
    y = df.loc[valid_mask, param2].astype(float).values
    z = df.loc[valid_mask, metric_col].astype(float).values

    if len(x) < 4:
        return None, None

    # Log transform if needed
    log_x = is_log_scale_param(param1)
    log_y = is_log_scale_param(param2)

    if log_x:
        pos_x = x > 0
        x, y, z = x[pos_x], y[pos_x], z[pos_x]
    if log_y:
        pos_y = y > 0
        x, y, z = x[pos_y], y[pos_y], z[pos_y]

    if len(x) < 4:
        return None, None

    x_t = np.log10(x) if log_x else x
    y_t = np.log10(y) if log_y else y

    # Create grid and interpolate
    xi = np.linspace(x_t.min(), x_t.max(), grid_resolution)
    yi = np.linspace(y_t.min(), y_t.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zi_grid = griddata((x_t, y_t), z, (xi_grid, yi_grid), method="linear", fill_value=np.nan)

    # Compute interaction strength (deviation from additive model)
    # Fit additive model: z = a*x + b*y + c
    try:
        from numpy.linalg import lstsq
        A = np.column_stack([x_t, y_t, np.ones_like(x_t)])
        coeffs, residuals, _, _ = lstsq(A, z, rcond=None)
        z_additive = coeffs[0] * x_t + coeffs[1] * y_t + coeffs[2]
        ss_res = np.sum((z - z_additive) ** 2)
        ss_tot = np.sum((z - z.mean()) ** 2)
        interaction_strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        # This is R² of additive model; lower means more interaction
        interaction_strength = 1 - interaction_strength  # Flip so higher = more interaction
    except Exception:
        interaction_strength = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    contour = ax.contourf(xi_grid, yi_grid, zi_grid, levels=15, cmap="viridis")
    ax.scatter(x_t, y_t, c="red", s=25, alpha=0.6, edgecolors="white", linewidth=0.5,
               label=f"Evaluations (n={len(x)})")

    ax.set_xlabel(clean_column_name(param1) + (" (log10)" if log_x else ""))
    ax.set_ylabel(clean_column_name(param2) + (" (log10)" if log_y else ""))

    title = f"Interaction: {clean_column_name(param1)} × {clean_column_name(param2)}"
    if interaction_strength is not None:
        title += f"\n(non-additivity: {interaction_strength:.3f})"
    ax.set_title(title)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(clean_column_name(metric_col))

    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, interaction_strength


# %%
# Plot interaction effects for configured pairs
if not df.empty:
    print("\n=== Parameter Interaction Effects ===")

    metric_cols = get_available_columns(df, [PRIMARY_METRIC], verbose=False)
    if not metric_cols:
        print(f"Primary metric '{PRIMARY_METRIC}' not found")
    elif not INTERACTION_PAIRS:
        print("No interaction pairs configured (INTERACTION_PAIRS is empty)")
    else:
        metric_col = metric_cols[0]
        print(f"Analyzing interactions for: {clean_column_name(metric_col)}")

        interaction_results = []

        for param1, param2 in INTERACTION_PAIRS:
            fig, strength = plot_interaction_effect(df, param1, param2, metric_col)

            if fig:
                p1_name = param1.split("/")[-1]
                p2_name = param2.split("/")[-1]
                output_path = output_dir / f"interaction_{p1_name}_{p2_name}.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Saved interaction plot to: {output_path}")
                plt.show()

                if strength is not None:
                    interaction_results.append({
                        "param1": clean_column_name(param1),
                        "param2": clean_column_name(param2),
                        "non_additivity": strength,
                    })
            else:
                print(f"Could not plot interaction for {param1} × {param2}")

        if interaction_results:
            print("\nInteraction summary (higher = stronger interaction):")
            int_df = pd.DataFrame(interaction_results).sort_values("non_additivity", ascending=False)
            print(int_df.to_string(index=False))

# %% [markdown]
# ## 6. Pareto Frontier: Accuracy vs Robustness Trade-off
#
# The Pareto frontier identifies **non-dominated** runs: configurations where you
# cannot improve one metric without worsening the other. These represent the best
# achievable trade-offs between competing objectives.
#
# A point is Pareto-optimal if no other point is better in ALL objectives.
# Points on the frontier define the "efficient boundary" of what's achievable.

# %%
def compute_pareto_frontier(
    x: np.ndarray,
    y: np.ndarray,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """
    Compute indices of Pareto-optimal points.

    Args:
        x, y: Arrays of metric values
        maximize_x, maximize_y: Whether higher is better for each metric

    Returns:
        Boolean array indicating Pareto-optimal points
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    # Flip signs if minimizing
    x_comp = x if maximize_x else -x
    y_comp = y if maximize_y else -y

    for i in range(n):
        if not is_pareto[i]:
            continue
        # Check if any other point dominates point i
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is >= in both and > in at least one
            if (x_comp[j] >= x_comp[i] and y_comp[j] >= y_comp[i] and
                (x_comp[j] > x_comp[i] or y_comp[j] > y_comp[i])):
                is_pareto[i] = False
                break

    return is_pareto


def plot_pareto_frontier(
    df: pd.DataFrame,
    metric1: str,
    metric2: str,
    maximize1: bool = True,
    maximize2: bool = True,
    figsize: tuple = (10, 8),
    show_labels: bool = True,
) -> plt.Figure:
    """
    Plot scatter of two metrics with Pareto frontier highlighted.

    Args:
        df: DataFrame with HPO results
        metric1, metric2: Column names for the two metrics (x and y axis)
        maximize1, maximize2: Whether higher is better for each metric
        figsize: Figure size
        show_labels: Whether to label Pareto-optimal points with run names

    Returns:
        matplotlib Figure
    """
    cols = get_available_columns(df, [metric1, metric2], verbose=True)
    if len(cols) < 2:
        print(f"Could not find both metrics: {metric1}, {metric2}")
        return None

    metric1, metric2 = cols[0], cols[1]

    valid_mask = df[metric1].notna() & df[metric2].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) < 2:
        print("Not enough valid data points for Pareto analysis")
        return None

    x = valid_df[metric1].astype(float).values
    y = valid_df[metric2].astype(float).values

    # Compute Pareto frontier
    is_pareto = compute_pareto_frontier(x, y, maximize1, maximize2)
    n_pareto = is_pareto.sum()

    # Sort Pareto points for line plotting
    pareto_x = x[is_pareto]
    pareto_y = y[is_pareto]
    sort_idx = np.argsort(pareto_x)
    pareto_x_sorted = pareto_x[sort_idx]
    pareto_y_sorted = pareto_y[sort_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    # Plot all points
    ax.scatter(x[~is_pareto], y[~is_pareto], c="lightgray", s=40, alpha=0.6,
               edgecolors="white", linewidth=0.5, label=f"Dominated (n={len(x) - n_pareto})")

    # Plot Pareto-optimal points
    ax.scatter(pareto_x, pareto_y, c="red", s=80, alpha=0.9,
               edgecolors="darkred", linewidth=1, label=f"Pareto-optimal (n={n_pareto})", zorder=5)

    # Draw Pareto frontier line
    ax.plot(pareto_x_sorted, pareto_y_sorted, "r--", alpha=0.7, linewidth=2, zorder=4)

    # Optionally label Pareto points with run names
    if show_labels and "run_name" in valid_df.columns and n_pareto <= 15:
        pareto_names = valid_df.loc[valid_df.index[is_pareto], "run_name"].values
        for i, (px, py) in enumerate(zip(pareto_x, pareto_y)):
            name = pareto_names[i] if i < len(pareto_names) else ""
            # Shorten name for display
            short_name = name.split("-")[-1][:8] if "-" in name else name[:8]
            ax.annotate(short_name, (px, py), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel(clean_column_name(metric1) + (" (↑ better)" if maximize1 else " (↓ better)"))
    ax.set_ylabel(clean_column_name(metric2) + (" (↑ better)" if maximize2 else " (↓ better)"))
    ax.set_title(f"Pareto Frontier: {clean_column_name(metric1)} vs {clean_column_name(metric2)}")

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Add summary stats
    stats_text = f"Total runs: {len(x)}\nPareto-optimal: {n_pareto} ({100*n_pareto/len(x):.1f}%)"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    return fig


def get_pareto_runs(
    df: pd.DataFrame,
    metric1: str,
    metric2: str,
    maximize1: bool = True,
    maximize2: bool = True,
) -> pd.DataFrame:
    """Return DataFrame of Pareto-optimal runs sorted by metric1."""
    cols = get_available_columns(df, [metric1, metric2], verbose=False)
    if len(cols) < 2:
        return pd.DataFrame()

    metric1, metric2 = cols[0], cols[1]

    valid_mask = df[metric1].notna() & df[metric2].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) < 2:
        return pd.DataFrame()

    x = valid_df[metric1].astype(float).values
    y = valid_df[metric2].astype(float).values

    is_pareto = compute_pareto_frontier(x, y, maximize1, maximize2)

    pareto_df = valid_df[is_pareto].copy()

    # Select columns to display
    config_cols = [c for c in pareto_df.columns if c.startswith("config/")]
    display_cols = ["run_name"] + config_cols + [metric1, metric2]
    display_cols = [c for c in display_cols if c in pareto_df.columns]

    return pareto_df[display_cols].sort_values(metric1, ascending=not maximize1)


# %%
# Plot Pareto frontier
if not df.empty and PARETO_METRICS and len(PARETO_METRICS) >= 2:
    print("\n=== Pareto Frontier Analysis ===")

    metric1, maximize1 = PARETO_METRICS[0]
    metric2, maximize2 = PARETO_METRICS[1]

    fig = plot_pareto_frontier(df, metric1, metric2, maximize1, maximize2)

    if fig:
        m1_name = metric1.split("/")[-1]
        m2_name = metric2.split("/")[-1]
        output_path = output_dir / f"pareto_{m1_name}_vs_{m2_name}.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved Pareto frontier to: {output_path}")
        plt.show()

        # Print Pareto-optimal runs
        print("\n--- Pareto-Optimal Runs ---")
        pareto_df = get_pareto_runs(df, metric1, metric2, maximize1, maximize2)
        if not pareto_df.empty:
            print(pareto_df.to_string(index=False))
        else:
            print("No Pareto-optimal runs found")

# %% [markdown]
# ## 7. Best Runs Summary

# %%
def get_best_runs(df: pd.DataFrame, metric_col: str, n_best: int = 5, minimize: bool = True) -> pd.DataFrame:
    """Get the best N runs for a given metric."""
    valid_df = df[df[metric_col].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame()

    sorted_df = valid_df.sort_values(metric_col, ascending=minimize)
    config_cols = [c for c in sorted_df.columns if c.startswith("config/")]
    display_cols = ["run_name"] + config_cols + [metric_col]
    display_cols = [c for c in display_cols if c in sorted_df.columns]

    return sorted_df.head(n_best)[display_cols]


# %%
if not df.empty:
    print("\n" + "=" * 60)
    print("BEST RUNS SUMMARY")
    print("=" * 60)

    # Validation accuracy (maximize)
    acc_cols = get_available_columns(df, ["summary/pre/valid/acc"], verbose=False)
    if acc_cols:
        print(f"\n--- Best by Validation Accuracy (higher is better) ---")
        best = get_best_runs(df, acc_cols[0], n_best=5, minimize=False)
        if not best.empty:
            print(best.to_string(index=False))

    # Validation loss (minimize)
    loss_cols = get_available_columns(df, ["summary/pre/valid/loss"], verbose=False)
    if loss_cols:
        print(f"\n--- Best by Validation Loss (lower is better) ---")
        best = get_best_runs(df, loss_cols[0], n_best=5, minimize=True)
        if not best.empty:
            print(best.to_string(index=False))

    # Robustness (maximize)
    rob_cols = get_available_columns(df, ROBUSTNESS_METRICS, verbose=False)
    for rob_col in rob_cols:
        print(f"\n--- Best by {clean_column_name(rob_col)} (higher is better) ---")
        best = get_best_runs(df, rob_col, n_best=5, minimize=False)
        if not best.empty:
            print(best.to_string(index=False))

# %% [markdown]
# ## 8. Parameter-Metric Correlations

# %%
def compute_param_metric_correlations(df: pd.DataFrame, params: list, metrics: list) -> pd.DataFrame:
    """Compute Pearson correlations between parameters and metrics."""
    params = get_available_columns(df, params, verbose=False)
    metrics = get_available_columns(df, metrics, verbose=False)

    if not params or not metrics:
        return pd.DataFrame()

    corr_data = {}
    for param in params:
        param_name = clean_column_name(param)
        corr_data[param_name] = {}

        for metric in metrics:
            metric_name = clean_column_name(metric)
            valid_mask = df[param].notna() & df[metric].notna()

            if valid_mask.sum() > 2:
                x = df.loc[valid_mask, param].astype(float)
                y = df.loc[valid_mask, metric].astype(float)

                # Log transform for lr/weight_decay
                if is_log_scale_param(param) and (x > 0).all():
                    x = np.log10(x)

                corr = np.corrcoef(x, y)[0, 1]
                corr_data[param_name][metric_name] = corr
            else:
                corr_data[param_name][metric_name] = np.nan

    return pd.DataFrame(corr_data).T


# %% [markdown]
# ## 9. Metric-Metric Correlations
#
# Shows how different metrics correlate with each other across all runs.
# This helps understand trade-offs (e.g., accuracy vs robustness).

# %%
def compute_metric_metric_correlations(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Compute Pearson correlations between all pairs of metrics.

    Args:
        df: DataFrame with HPO results
        metrics: List of metric column names

    Returns:
        DataFrame with correlation matrix (metrics x metrics)
    """
    metrics = get_available_columns(df, metrics, verbose=False)

    if len(metrics) < 2:
        print("Need at least 2 metrics for correlation analysis")
        return pd.DataFrame()

    # Build correlation matrix
    n = len(metrics)
    corr_matrix = np.full((n, n), np.nan)
    metric_names = [clean_column_name(m) for m in metrics]

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            valid_mask = df[m1].notna() & df[m2].notna()
            if valid_mask.sum() > 2:
                x = df.loc[valid_mask, m1].astype(float)
                y = df.loc[valid_mask, m2].astype(float)
                corr_matrix[i, j] = np.corrcoef(x, y)[0, 1]

    return pd.DataFrame(corr_matrix, index=metric_names, columns=metric_names)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlations",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot correlation matrix as heatmap with values."""
    if corr_df.empty:
        print("No correlation data to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=9)

    # Add correlation values
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
    fig.tight_layout()

    return fig


def plot_metric_scatter_matrix(df: pd.DataFrame, metrics: list, figsize: tuple = (12, 12)) -> plt.Figure:
    """
    Create scatter plot matrix showing relationships between all metric pairs.
    Diagonal shows histograms of each metric.
    """
    metrics = get_available_columns(df, metrics, verbose=False)

    if len(metrics) < 2:
        print("Need at least 2 metrics for scatter matrix")
        return None

    n = len(metrics)
    fig, axes = plt.subplots(n, n, figsize=figsize, dpi=DPI)

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                valid = df[m1].dropna().astype(float)
                if len(valid) > 0:
                    ax.hist(valid, bins=20, color="steelblue", edgecolor="white", alpha=0.7)
                ax.set_ylabel("Count" if j == 0 else "")
            else:
                # Off-diagonal: scatter
                valid_mask = df[m1].notna() & df[m2].notna()
                if valid_mask.sum() > 0:
                    x = df.loc[valid_mask, m2].astype(float)
                    y = df.loc[valid_mask, m1].astype(float)
                    ax.scatter(x, y, alpha=0.5, s=20, c="steelblue")

                    # Add correlation text
                    if valid_mask.sum() > 2:
                        corr = np.corrcoef(x, y)[0, 1]
                        ax.text(0.05, 0.95, f"r={corr:.2f}", transform=ax.transAxes,
                                fontsize=8, va="top", ha="left",
                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            # Labels
            if i == n - 1:
                ax.set_xlabel(clean_column_name(m2), fontsize=8)
            if j == 0:
                ax.set_ylabel(clean_column_name(m1), fontsize=8)

            ax.tick_params(labelsize=7)

    fig.suptitle("Metric-Metric Relationships", y=1.02)
    fig.tight_layout()
    return fig


# %%
# Compute and plot correlations
if not df.empty:
    all_metrics = VALIDATION_METRICS + ROBUSTNESS_METRICS + TEST_METRICS

    # Parameter-Metric correlations
    print("\n=== Parameter-Metric Correlations ===")
    param_metric_corr = compute_param_metric_correlations(df, HPO_PARAMS, all_metrics)

    if not param_metric_corr.empty:
        print("(LR and weight_decay are log-transformed)")
        print(param_metric_corr.round(3).to_string())

        fig = plot_correlation_heatmap(param_metric_corr, title="Parameter-Metric Correlations")
        if fig:
            output_path = output_dir / "param_metric_correlations.png"
            plt.savefig(output_path, bbox_inches="tight")
            print(f"\nSaved to: {output_path}")
            plt.show()

    # Metric-Metric correlations
    print("\n=== Metric-Metric Correlations ===")
    metric_metric_corr = compute_metric_metric_correlations(df, all_metrics)

    if not metric_metric_corr.empty:
        print(metric_metric_corr.round(3).to_string())

        fig = plot_correlation_heatmap(metric_metric_corr, title="Metric-Metric Correlations")
        if fig:
            output_path = output_dir / "metric_metric_correlations.png"
            plt.savefig(output_path, bbox_inches="tight")
            print(f"\nSaved to: {output_path}")
            plt.show()

        # Scatter matrix for key metrics
        key_metrics = get_available_columns(df, VALIDATION_METRICS + ROBUSTNESS_METRICS[:2], verbose=False)
        if len(key_metrics) >= 2:
            fig = plot_metric_scatter_matrix(df, key_metrics)
            if fig:
                output_path = output_dir / "metric_scatter_matrix.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Saved scatter matrix to: {output_path}")
                plt.show()

# %% [markdown]
# ## 10. Export Results

# %%
if not df.empty:
    # Save best runs summary to output_dir (already set up earlier)
    summary_path = output_dir / "best_runs_summary.txt"
    with open(summary_path, "w") as f:
        f.write("HPO Best Runs Summary\n")
        f.write(f"Sweep: {sweep_name}\n")
        f.write(f"Data source: {DATA_SOURCE}\n")
        f.write("=" * 60 + "\n\n")

        for metric_col, minimize, label in [
            (get_available_columns(df, ["summary/pre/valid/acc"], verbose=False), False, "Validation Accuracy"),
            (get_available_columns(df, ["summary/pre/valid/loss"], verbose=False), True, "Validation Loss"),
        ]:
            if metric_col:
                f.write(f"Best by {label}:\n")
                best = get_best_runs(df, metric_col[0], n_best=10, minimize=minimize)
                if not best.empty:
                    f.write(best.to_string(index=False) + "\n\n")

    print(f"Saved best runs summary to: {summary_path}")

# %%
print("\n" + "=" * 60)
print("Analysis complete!")
print(f"Data source: {DATA_SOURCE}")
print(f"Total runs analyzed: {len(df) if not df.empty else 0}")
print("=" * 60)
