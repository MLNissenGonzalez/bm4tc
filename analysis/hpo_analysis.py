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
# 1. Scatter/histogram plots: Each parameter vs each metric
# 2. Surface plots: lr × weight_decay vs validation accuracy
# 3. Surface plots: lr × weight_decay vs HPO goal metric
# 4. Surface plots: lr × weight_decay vs robust accuracy (if available)
# 5. Metric-metric correlations: How metrics relate to each other
# 6. Parameter-metric correlations: How parameters affect metrics

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
EXPERIMENT_PATTERN = "lrwdbs_hpo"  # Regex pattern to match run groups
DATASET_NAME = None  # e.g., "spirals_4k", "moons_4k", or None for all

# --- LOCAL SETTINGS (used if DATA_SOURCE == "local") ---
# Path to sweep directory relative to project root
LOCAL_SWEEP_DIR = "outputs/lrwdbs_hpo_spirals_4k_22Jan26"

# --- ANALYSIS SETTINGS ---
# HPO parameters that were optimized (column names after fetching)
HPO_PARAMS = [
    "config/lr",
    "config/weight_decay",
    "config/batch_size",
]

# Metrics to analyze (will auto-detect available columns)
VALIDATION_METRICS = [
    "summary/pre/valid/acc",
    "summary/pre/valid/loss",
]

# Robustness metrics (check your wandb/local data for exact names)
ROBUSTNESS_METRICS = [
    "summary/pre/valid/rob/0.1",
    "summary/pre/valid/rob/0.3",
]

# Test metrics (for final evaluation)
TEST_METRICS = [
    "summary/pre/test/acc",
    "summary/pre/test/loss",
]

# Plot settings
FIGSIZE = (10, 6)
SURFACE_FIGSIZE = (12, 8)
DPI = 100

# %% [markdown]
# ## Load Data

# %%
def load_data(source: str = DATA_SOURCE) -> pd.DataFrame:
    """
    Load HPO experiment data from configured source.

    Args:
        source: "wandb" or "local"

    Returns:
        DataFrame with config parameters and summary metrics
    """
    if source == "wandb":
        from analysis.utils import fetch_hpo_runs

        print(f"Fetching runs from wandb: {WANDB_ENTITY}/{WANDB_PROJECT}")
        print(f"Experiment pattern: {EXPERIMENT_PATTERN}")
        if DATASET_NAME:
            print(f"Dataset filter: {DATASET_NAME}")

        df = fetch_hpo_runs(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            experiment_pattern=EXPERIMENT_PATTERN,
            dataset_name=DATASET_NAME,
        )

    elif source == "local":
        from analysis.utils import load_local_hpo_runs

        sweep_path = project_root / LOCAL_SWEEP_DIR
        print(f"Loading runs from local directory: {sweep_path}")

        df = load_local_hpo_runs(sweep_path)

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
        output_path = project_root / "analysis" / "param_metric_scatter.png"
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
        output_path = project_root / "analysis" / f"param_hist_{metric_name}.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.show()

# %% [markdown]
# ## 3. Surface Plots: LR × Weight Decay vs Metrics

# %%
def create_surface_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    log_x: bool = True,
    log_y: bool = True,
    grid_resolution: int = 50,
    method: str = "linear",
    title: str = None,
) -> tuple:
    """Create 3D surface plot from scattered data."""
    valid_mask = df[x_col].notna() & df[y_col].notna() & df[z_col].notna()
    x = df.loc[valid_mask, x_col].astype(float).values
    y = df.loc[valid_mask, y_col].astype(float).values
    z = df.loc[valid_mask, z_col].astype(float).values

    if len(x) < 4:
        print(f"Not enough data points for surface plot ({len(x)} points, need >= 4)")
        return None, None

    # Filter positive values for log scale
    if log_x or log_y:
        pos_mask = (x > 0) & (y > 0) if (log_x and log_y) else (x > 0 if log_x else y > 0)
        x, y, z = x[pos_mask], y[pos_mask], z[pos_mask]

    if len(x) < 4:
        print("Not enough positive values for log scale")
        return None, None

    # Transform to log space
    x_t = np.log10(x) if log_x else x
    y_t = np.log10(y) if log_y else y

    # Create interpolation grid
    xi = np.linspace(x_t.min(), x_t.max(), grid_resolution)
    yi = np.linspace(y_t.min(), y_t.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zi_grid = griddata((x_t, y_t), z, (xi_grid, yi_grid), method=method, fill_value=np.nan)

    # Create plot
    fig = plt.figure(figsize=SURFACE_FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(xi_grid, yi_grid, zi_grid, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.scatter(x_t, y_t, z, c="red", s=20, alpha=0.6, label="Data")

    ax.set_xlabel(clean_column_name(x_col) + (" (log10)" if log_x else ""))
    ax.set_ylabel(clean_column_name(y_col) + (" (log10)" if log_y else ""))
    ax.set_zlabel(clean_column_name(z_col))

    if title:
        ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    return fig, ax


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
# Create surface plots for LR × Weight Decay
if not df.empty:
    print("\n=== Creating Surface Plots ===")

    lr_cols = get_available_columns(df, ["config/lr"], verbose=False)
    wd_cols = get_available_columns(df, ["config/weight_decay"], verbose=False)

    if lr_cols and wd_cols:
        lr_col, wd_col = lr_cols[0], wd_cols[0]
        metrics_to_plot = get_available_columns(df, VALIDATION_METRICS + ROBUSTNESS_METRICS, verbose=False)

        for metric_col in metrics_to_plot:
            metric_name = metric_col.split("/")[-1]
            title = f"{clean_column_name(metric_col)} vs LR and Weight Decay"

            # Contour plot (more readable)
            fig, ax = create_contour_plot(df, lr_col, wd_col, metric_col, title=title)
            if fig:
                output_path = project_root / "analysis" / f"contour_lr_wd_{metric_name}.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Saved contour plot to: {output_path}")
                plt.show()

            # Surface plot
            fig, ax = create_surface_plot(df, lr_col, wd_col, metric_col, title=title)
            if fig:
                output_path = project_root / "analysis" / f"surface_lr_wd_{metric_name}.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Saved surface plot to: {output_path}")
                plt.show()
    else:
        print("Could not find LR and/or weight_decay columns")
        print(f"Available config columns: {[c for c in df.columns if c.startswith('config/')]}")

# %% [markdown]
# ## 4. Best Runs Summary

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
# ## 5. Parameter-Metric Correlations

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
# ## 6. Metric-Metric Correlations
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
            output_path = project_root / "analysis" / "param_metric_correlations.png"
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
            output_path = project_root / "analysis" / "metric_metric_correlations.png"
            plt.savefig(output_path, bbox_inches="tight")
            print(f"\nSaved to: {output_path}")
            plt.show()

        # Scatter matrix for key metrics
        key_metrics = get_available_columns(df, VALIDATION_METRICS + ROBUSTNESS_METRICS[:2], verbose=False)
        if len(key_metrics) >= 2:
            fig = plot_metric_scatter_matrix(df, key_metrics)
            if fig:
                output_path = project_root / "analysis" / "metric_scatter_matrix.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Saved scatter matrix to: {output_path}")
                plt.show()

# %% [markdown]
# ## 7. Export Results

# %%
if not df.empty:
    output_dir = project_root / "analysis" / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save processed dataframe
    csv_path = output_dir / "hpo_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved processed data to: {csv_path}")

    # Save correlation matrices
    if "param_metric_corr" in dir() and not param_metric_corr.empty:
        param_metric_corr.to_csv(output_dir / "param_metric_correlations.csv")

    if "metric_metric_corr" in dir() and not metric_metric_corr.empty:
        metric_metric_corr.to_csv(output_dir / "metric_metric_correlations.csv")

    # Save best runs summary
    summary_path = output_dir / "best_runs_summary.txt"
    with open(summary_path, "w") as f:
        f.write("HPO Best Runs Summary\n")
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
