# %% [markdown]
# # Sweep Statistics Analysis
#
# This notebook provides basic statistics and visualizations for any sweep
# (wandb or local Hydra multirun).
#
# **Data Sources:**
# - **wandb**: Fetches run data from Weights & Biases API
# - **local**: Loads data from local outputs/ directory (Hydra multirun)
#
# **Visualizations:**
# 1. Histogram of accuracy (clean, robust, MIA if available)
# 2. Mean accuracies with std dev bars
# 3. Scatter plot of accuracy vs validation loss
# 4. Scatter plot of accuracy vs stopping criterion
# 5. Best run summary by validation loss
# 6. Final summary table (best, mean, std of accuracies)
# 7. Distribution visualization for best run
#
# **Features:**
# - Regime-aware metric resolution
# - Optional MIA computation (loads each model)
# - Manual EFFECTIVE_N override for std error correction

# %% [markdown]
# ## Setup and Imports

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
import torch
import warnings

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION FOR YOUR EXPERIMENT
# =============================================================================

# Data source: "wandb" or "local"
DATA_SOURCE = "wandb"  # Change to "wandb" to fetch from W&B

# --- WANDB SETTINGS (used if DATA_SOURCE == "wandb") ---
WANDB_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
WANDB_PROJECT = "gan_train"
EXPERIMENT_PATTERN = "adv_hpo"  # Regex pattern to match run groups
DATASET_NAME = "spirals_4k"  # e.g., "spirals_4k", "moons_4k", or None for all

# --- LOCAL SETTINGS (used if DATA_SOURCE == "local") ---
# Path to sweep directory relative to project root
# Can be overridden by command line argument: python -m analysis.run_statistics outputs/sweep_name
LOCAL_SWEEP_DIR = _CLI_SWEEP_DIR if _CLI_SWEEP_DIR else "outputs/cls_seed_sweep_circles_4k_10Feb26"

# --- REGIME SETTINGS ---
# Training regime: "pre", "gen", "adv", "gan"
#   - "pre": Classification pretraining (trainer.classification.*)
#   - "gen": Generative NLL training (trainer.generative.*)
#   - "adv": Adversarial training (trainer.adversarial.*)
#   - "gan": GAN-style training (trainer.ganstyle.*)
REGIME = "adv"

# --- MIA SETTINGS ---
# MIA computation is expensive - it loads each model and runs inference
COMPUTE_MIA = False  # Set to True to compute MIA for all runs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- STATISTICS SETTINGS ---
# Manual override for effective N (number of independent runs)
# Set to None to use actual run count, or set to a number if some runs
# are not truly independent (e.g., old sweeps included the dead tracking.random_state param)
EFFECTIVE_N = None

# --- PLOT SETTINGS ---
FIGSIZE = (10, 6)
DPI = 100

REGIME_TRAINER_PREFIX = {
    "pre": "trainer.classification",
    "gen": "trainer.generative",
    "adv": "trainer.adversarial",
    "gan": "trainer.ganstyle",
}

# %% [markdown]
# ## Data Loading

# %%
def load_data(source: str = DATA_SOURCE, regime: str = REGIME) -> pd.DataFrame:
    """
    Load sweep data from configured source.

    Args:
        source: "wandb" or "local"
        regime: Training regime to determine which config keys to extract

    Returns:
        DataFrame with config parameters and summary metrics
    """
    from analysis.utils import REGIME_PARAM_MAP

    # Get regime-specific config keys
    if regime in REGIME_PARAM_MAP:
        config_keys = list(REGIME_PARAM_MAP[regime].values())
        config_keys.extend([
            "dataset.name",
            "experiment",
        ])
        if regime in REGIME_TRAINER_PREFIX:
            config_keys.append(f"{REGIME_TRAINER_PREFIX[regime]}.stop_crit")
    else:
        config_keys = None

    if source == "wandb":
        from analysis.utils import WandbFetcher

        print(f"Fetching runs from wandb: {WANDB_ENTITY}/{WANDB_PROJECT}")
        print(f"Experiment pattern: {EXPERIMENT_PATTERN}")
        print(f"Regime: {regime}")
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
        print(f"Regime: {regime}")

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

# %% [markdown]
# ## Resolve Metrics and Load Data

# %%
from analysis.utils import resolve_metrics, REGIME_PARAM_MAP

print("=" * 60)
print(f"Loading data from: {DATA_SOURCE}")
print("=" * 60)

df = load_data()

# %%
# Resolve metrics based on regime
if not df.empty:
    _resolved_metrics = resolve_metrics(REGIME, df)

    # Extract key metric columns
    ACC_COL = _resolved_metrics["validation"][0]  # e.g., summary/pre/valid/acc
    LOSS_COL = _resolved_metrics["validation"][1]  # e.g., summary/pre/valid/loss
    ROB_COLS = _resolved_metrics["robustness"]     # e.g., [summary/pre/valid/rob/0.1, ...]

    print(f"\nResolved metrics for regime '{REGIME}':")
    print(f"  Accuracy: {ACC_COL}")
    print(f"  Loss: {LOSS_COL}")
    print(f"  Robustness: {ROB_COLS}")

    # Filter to finished runs
    if "state" in df.columns:
        n_before = len(df)
        df = df[df["state"] == "finished"].copy()
        print(f"\nFiltered to finished runs: {n_before} -> {len(df)}")

    print(f"\nTotal runs: {len(df)}")

# %%
# Resolve stopping criterion metric
from analysis.utils import REGIME_METRIC_PREFIX

STOP_CRIT_COL = None
STOP_CRIT_LABEL = None

if not df.empty and "config/stop_crit" in df.columns:
    unique = df["config/stop_crit"].dropna().unique()
    if len(unique) == 1:
        stop_name = unique[0]
        prefix = REGIME_METRIC_PREFIX[REGIME]

        if stop_name == "rob":
            # Average robustness columns (how the trainer uses it)
            valid_rob = [c for c in ROB_COLS if c in df.columns]
            if valid_rob:
                df["_avg_rob_acc"] = df[valid_rob].mean(axis=1)
                STOP_CRIT_COL = "_avg_rob_acc"
                STOP_CRIT_LABEL = "Avg Robust Accuracy (stop crit)"
        else:
            candidate = f"summary/{prefix}/valid/{stop_name}"
            if candidate in df.columns:
                STOP_CRIT_COL = candidate
                STOP_CRIT_LABEL = f"{stop_name} (stop crit)"

        if STOP_CRIT_COL:
            print(f"  Stop criterion: {stop_name} -> {STOP_CRIT_COL}")
        else:
            print(f"  Stop criterion '{stop_name}' column not found in data — skipping")
    else:
        print(f"  Multiple stopping criteria found: {list(unique)} — skipping stop crit plot")

# %% [markdown]
# ## Setup Output Directory

# %%
# Determine sweep name
if DATA_SOURCE == "local":
    sweep_name = Path(LOCAL_SWEEP_DIR).name
else:
    sweep_name = DATASET_NAME if DATASET_NAME else EXPERIMENT_PATTERN
sweep_name = sweep_name.replace("/", "_")

# Create output directory inside analysis/outputs/
output_dir = project_root / "analysis" / "outputs" / sweep_name
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {output_dir}")

# %% [markdown]
# ## MIA Computation (Optional)

# %%
def compute_mia_for_run(run_path: str, device: str = DEVICE) -> float:
    """
    Compute MIA attack accuracy for a single run.

    Args:
        run_path: Path to the run directory
        device: Torch device string

    Returns:
        MIA attack accuracy score
    """
    from analysis.utils import load_run_config, find_model_checkpoint
    from analysis.utils import MIAEvaluation
    from src.models import BornMachine
    from src.data import DataHandler
    from torch.utils.data import DataLoader, TensorDataset

    try:
        run_dir = Path(run_path)
        device_obj = torch.device(device)

        # Load config and model
        cfg = load_run_config(run_dir)
        checkpoint_path = find_model_checkpoint(run_dir)
        bm = BornMachine.load(str(checkpoint_path))
        bm.to(device_obj)

        # Load data
        datahandler = DataHandler(cfg.dataset)
        datahandler.load()
        datahandler.split_and_rescale(bm)

        # Create data loaders
        train_data = datahandler.data["train"]
        train_labels = datahandler.labels["train"]
        test_data = datahandler.data["test"]
        test_labels = datahandler.labels["test"]

        train_loader = DataLoader(
            TensorDataset(train_data, train_labels),
            batch_size=256,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(test_data, test_labels),
            batch_size=256,
            shuffle=False
        )

        # Run MIA evaluation
        mia_eval = MIAEvaluation()
        results = mia_eval.evaluate(bm, train_loader, test_loader, device_obj)

        return results.attack_accuracy

    except Exception as e:
        print(f"  Warning: MIA computation failed for {run_path}: {e}")
        return np.nan


def compute_mia_for_all_runs(df: pd.DataFrame, device: str = DEVICE) -> pd.Series:
    """
    Compute MIA attack accuracy for all runs in the DataFrame.

    Args:
        df: DataFrame with run_path column
        device: Torch device string

    Returns:
        Series of MIA attack accuracy scores indexed by DataFrame index
    """
    mia_scores = []

    print("\nComputing MIA for all runs...")
    for i, (idx, row) in enumerate(df.iterrows()):
        run_path = row.get("run_path")
        if run_path is None:
            print(f"  Warning: No run_path for run {idx}")
            mia_scores.append(np.nan)
            continue

        print(f"  [{i+1}/{len(df)}] Processing {Path(run_path).name}...")
        score = compute_mia_for_run(run_path, device)
        mia_scores.append(score)
        print(f"    MIA Accuracy: {score:.4f}" if not np.isnan(score) else "    MIA: failed")

    return pd.Series(mia_scores, index=df.index)


# %%
# Compute MIA if enabled
MIA_COL = None
if COMPUTE_MIA and not df.empty:
    print("\n" + "=" * 60)
    print("Computing MIA for all runs (this may take a while)...")
    print("=" * 60)

    df["mia_accuracy"] = compute_mia_for_all_runs(df, DEVICE)
    MIA_COL = "mia_accuracy"

    valid_mia = df["mia_accuracy"].notna().sum()
    print(f"\nMIA computed for {valid_mia}/{len(df)} runs")
    print(f"Mean MIA Accuracy: {df['mia_accuracy'].mean():.4f}")

# %% [markdown]
# ## Statistical Utility Functions

# %%
def compute_statistics(
    df: pd.DataFrame,
    metric_col: str,
    effective_n: int = None,
) -> dict:
    """
    Compute best, mean, std, and stderr for a metric.

    Args:
        df: DataFrame with metric column
        metric_col: Column name for the metric
        effective_n: Override for sample size in stderr calculation

    Returns:
        Dict with best, mean, std, stderr, n values
    """
    if metric_col not in df.columns:
        return {"best": np.nan, "mean": np.nan, "std": np.nan, "stderr": np.nan, "n": 0}

    values = df[metric_col].dropna()
    n = len(values)

    if n == 0:
        return {"best": np.nan, "mean": np.nan, "std": np.nan, "stderr": np.nan, "n": 0}

    # Use effective_n for stderr if provided
    n_for_stderr = effective_n if effective_n is not None else n

    return {
        "best": values.max(),
        "mean": values.mean(),
        "std": values.std(),
        "stderr": values.std() / np.sqrt(n_for_stderr) if n_for_stderr > 0 else np.nan,
        "n": n,
    }


def get_best_run_by_loss(df: pd.DataFrame, loss_col: str) -> pd.Series:
    """
    Get the best run by lowest validation loss.

    Args:
        df: DataFrame with loss column
        loss_col: Column name for the loss metric

    Returns:
        Series representing the best run
    """
    if loss_col not in df.columns:
        return pd.Series()

    valid_df = df[df[loss_col].notna()]
    if valid_df.empty:
        return pd.Series()

    best_idx = valid_df[loss_col].idxmin()
    return df.loc[best_idx]


def clean_column_name(col: str) -> str:
    """Convert column name to readable label."""
    name = col.replace("config/", "").replace("summary/", "")
    name = name.replace("_", " ").replace("/", " / ")
    return name.title()

# %% [markdown]
# ## Visualization 1: Accuracy Histogram

# %%
def plot_accuracy_histogram(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: list = None,
    mia_col: str = None,
    title: str = "Accuracy Distribution",
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """
    Plot histograms of accuracy metrics.

    Args:
        df: DataFrame with accuracy columns
        acc_col: Column name for clean accuracy
        rob_cols: List of column names for robustness metrics
        mia_col: Column name for MIA accuracy (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    # Determine number of subplots
    n_plots = 1
    if rob_cols:
        n_plots += len([c for c in rob_cols if c in df.columns])
    if mia_col and mia_col in df.columns:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), dpi=DPI, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0

    # Clean accuracy
    if acc_col in df.columns:
        ax = axes[plot_idx]
        values = df[acc_col].dropna()
        ax.hist(values, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.4f}")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Count")
        ax.set_title("Clean Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Robustness metrics
    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                ax = axes[plot_idx]
                values = df[rob_col].dropna()
                if len(values) > 0:
                    ax.hist(values, bins=15, color="orange", edgecolor="white", alpha=0.8)
                    ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.4f}")
                    strength = rob_col.split("/")[-1]
                    ax.set_xlabel("Robust Accuracy")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Robust Acc (ε={strength})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                plot_idx += 1

    # MIA
    if mia_col and mia_col in df.columns:
        ax = axes[plot_idx]
        values = df[mia_col].dropna()
        if len(values) > 0:
            ax.hist(values, bins=15, color="purple", edgecolor="white", alpha=0.8)
            ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.4f}")
            ax.set_xlabel("MIA Accuracy")
            ax.set_ylabel("Count")
            ax.set_title("MIA Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# %%
if not df.empty:
    print("\n=== Creating Accuracy Histogram ===")

    fig = plot_accuracy_histogram(
        df,
        acc_col=ACC_COL,
        rob_cols=ROB_COLS,
        mia_col=MIA_COL,
        title=f"Accuracy Distribution ({sweep_name})",
    )

    output_path = output_dir / "accuracy_histogram.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.show()

# %% [markdown]
# ## Visualization 2: Mean Accuracies with Error Bars

# %%
def plot_mean_accuracies_with_errorbars(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: list = None,
    mia_col: str = None,
    title: str = "Mean Accuracies",
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """
    Plot mean accuracies with standard deviation bars.

    Args:
        df: DataFrame with accuracy columns
        acc_col: Column name for clean accuracy
        rob_cols: List of column names for robustness metrics
        mia_col: Column name for MIA accuracy (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    metrics = []
    means = []
    stds = []
    colors = []

    # Clean accuracy
    if acc_col in df.columns:
        stats = compute_statistics(df, acc_col)
        metrics.append("Clean\nAccuracy")
        means.append(stats["mean"])
        stds.append(stats["std"])
        colors.append("steelblue")

    # Robustness metrics
    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                stats = compute_statistics(df, rob_col)
                strength = rob_col.split("/")[-1]
                metrics.append(f"Robust\n(ε={strength})")
                means.append(stats["mean"])
                stds.append(stats["std"])
                colors.append("orange")

    # MIA
    if mia_col and mia_col in df.columns:
        stats = compute_statistics(df, mia_col)
        metrics.append("MIA\nAccuracy")
        means.append(stats["mean"])
        stds.append(stats["std"])
        colors.append("purple")

    if not metrics:
        print("No valid metrics to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", alpha=0.8)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


# %%
if not df.empty:
    print("\n=== Creating Mean Accuracies Plot ===")

    fig = plot_mean_accuracies_with_errorbars(
        df,
        acc_col=ACC_COL,
        rob_cols=ROB_COLS,
        mia_col=MIA_COL,
        title=f"Mean Accuracies \u00b1 Std Dev ({sweep_name})",
    )

    if fig:
        output_path = output_dir / "mean_accuracies_errorbars.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.show()

# %% [markdown]
# ## Visualization 3: Scatter Plot vs Validation Loss

# %%
def plot_accuracy_vs_loss_scatter(
    df: pd.DataFrame,
    loss_col: str,
    acc_col: str,
    rob_cols: list = None,
    mia_col: str = None,
    title: str = "Accuracy vs Validation Loss",
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """
    Plot accuracy metrics against validation loss.

    Args:
        df: DataFrame with metric columns
        loss_col: Column name for loss
        acc_col: Column name for clean accuracy
        rob_cols: List of column names for robustness metrics
        mia_col: Column name for MIA accuracy (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if loss_col not in df.columns:
        print(f"Loss column '{loss_col}' not found")
        return None

    # Count number of metrics to plot
    n_plots = 1
    if rob_cols:
        n_plots += len([c for c in rob_cols if c in df.columns])
    if mia_col and mia_col in df.columns:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), dpi=DPI, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    loss = df[loss_col]

    # Clean accuracy
    if acc_col in df.columns:
        ax = axes[plot_idx]
        valid_mask = loss.notna() & df[acc_col].notna()
        ax.scatter(loss[valid_mask], df.loc[valid_mask, acc_col],
                   alpha=0.6, c="steelblue", edgecolors="white", s=50)
        ax.set_xlabel("Validation Loss")
        ax.set_ylabel("Accuracy")
        ax.set_title("Clean Accuracy")
        ax.grid(True, alpha=0.3)

        # Add correlation
        if valid_mask.sum() > 2:
            corr = np.corrcoef(loss[valid_mask], df.loc[valid_mask, acc_col])[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment="top")
        plot_idx += 1

    # Robustness metrics
    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                ax = axes[plot_idx]
                valid_mask = loss.notna() & df[rob_col].notna()
                if valid_mask.sum() > 0:
                    ax.scatter(loss[valid_mask], df.loc[valid_mask, rob_col],
                               alpha=0.6, c="orange", edgecolors="white", s=50)
                    strength = rob_col.split("/")[-1]
                    ax.set_xlabel("Validation Loss")
                    ax.set_ylabel("Robust Accuracy")
                    ax.set_title(f"Robust Acc (ε={strength})")
                    ax.grid(True, alpha=0.3)

                    if valid_mask.sum() > 2:
                        corr = np.corrcoef(loss[valid_mask], df.loc[valid_mask, rob_col])[0, 1]
                        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                                fontsize=10, verticalalignment="top")
                plot_idx += 1

    # MIA
    if mia_col and mia_col in df.columns:
        ax = axes[plot_idx]
        valid_mask = loss.notna() & df[mia_col].notna()
        if valid_mask.sum() > 0:
            ax.scatter(loss[valid_mask], df.loc[valid_mask, mia_col],
                       alpha=0.6, c="purple", edgecolors="white", s=50)
            ax.set_xlabel("Validation Loss")
            ax.set_ylabel("MIA Accuracy")
            ax.set_title("MIA Accuracy")
            ax.grid(True, alpha=0.3)

            if valid_mask.sum() > 2:
                corr = np.corrcoef(loss[valid_mask], df.loc[valid_mask, mia_col])[0, 1]
                ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# %%
if not df.empty:
    print("\n=== Creating Accuracy vs Loss Scatter ===")

    fig = plot_accuracy_vs_loss_scatter(
        df,
        loss_col=LOSS_COL,
        acc_col=ACC_COL,
        rob_cols=ROB_COLS,
        mia_col=MIA_COL,
        title=f"Accuracy vs Validation Loss ({sweep_name})",
    )

    if fig:
        output_path = output_dir / "accuracy_vs_loss_scatter.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.show()

# %% [markdown]
# ## Visualization 4: Accuracy vs Stopping Criterion

# %%
def plot_accuracy_vs_stop_crit_scatter(
    df: pd.DataFrame,
    stop_crit_col: str,
    acc_col: str,
    rob_cols: list = None,
    mia_col: str = None,
    stop_crit_label: str = "Stop Criterion",
    title: str = "Accuracy vs Stopping Criterion",
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """
    Plot accuracy metrics against the stopping criterion metric.

    Args:
        df: DataFrame with metric columns
        stop_crit_col: Column name for stopping criterion metric
        acc_col: Column name for clean accuracy
        rob_cols: List of column names for robustness metrics
        mia_col: Column name for MIA accuracy (optional)
        stop_crit_label: Label for the x-axis
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if stop_crit_col not in df.columns:
        print(f"Stop criterion column '{stop_crit_col}' not found")
        return None

    # Count number of metrics to plot
    n_plots = 1
    if rob_cols:
        n_plots += len([c for c in rob_cols if c in df.columns])
    if mia_col and mia_col in df.columns:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), dpi=DPI, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    stop_vals = df[stop_crit_col]

    # Clean accuracy
    if acc_col in df.columns:
        ax = axes[plot_idx]
        valid_mask = stop_vals.notna() & df[acc_col].notna()
        ax.scatter(stop_vals[valid_mask], df.loc[valid_mask, acc_col],
                   alpha=0.6, c="steelblue", edgecolors="white", s=50)
        ax.set_xlabel(stop_crit_label)
        ax.set_ylabel("Accuracy")
        ax.set_title("Clean Accuracy")
        ax.grid(True, alpha=0.3)

        if valid_mask.sum() > 2:
            corr = np.corrcoef(stop_vals[valid_mask], df.loc[valid_mask, acc_col])[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment="top")
        plot_idx += 1

    # Robustness metrics
    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                ax = axes[plot_idx]
                valid_mask = stop_vals.notna() & df[rob_col].notna()
                if valid_mask.sum() > 0:
                    ax.scatter(stop_vals[valid_mask], df.loc[valid_mask, rob_col],
                               alpha=0.6, c="orange", edgecolors="white", s=50)
                    strength = rob_col.split("/")[-1]
                    ax.set_xlabel(stop_crit_label)
                    ax.set_ylabel("Robust Accuracy")
                    ax.set_title(f"Robust Acc (ε={strength})")
                    ax.grid(True, alpha=0.3)

                    if valid_mask.sum() > 2:
                        corr = np.corrcoef(stop_vals[valid_mask], df.loc[valid_mask, rob_col])[0, 1]
                        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                                fontsize=10, verticalalignment="top")
                plot_idx += 1

    # MIA
    if mia_col and mia_col in df.columns:
        ax = axes[plot_idx]
        valid_mask = stop_vals.notna() & df[mia_col].notna()
        if valid_mask.sum() > 0:
            ax.scatter(stop_vals[valid_mask], df.loc[valid_mask, mia_col],
                       alpha=0.6, c="purple", edgecolors="white", s=50)
            ax.set_xlabel(stop_crit_label)
            ax.set_ylabel("MIA Accuracy")
            ax.set_title("MIA Accuracy")
            ax.grid(True, alpha=0.3)

            if valid_mask.sum() > 2:
                corr = np.corrcoef(stop_vals[valid_mask], df.loc[valid_mask, mia_col])[0, 1]
                ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# %%
if not df.empty and STOP_CRIT_COL is not None:
    print("\n=== Creating Accuracy vs Stop Criterion Scatter ===")

    fig = plot_accuracy_vs_stop_crit_scatter(
        df,
        stop_crit_col=STOP_CRIT_COL,
        acc_col=ACC_COL,
        rob_cols=ROB_COLS,
        mia_col=MIA_COL,
        stop_crit_label=STOP_CRIT_LABEL,
        title=f"Accuracy vs Stopping Criterion ({sweep_name})",
    )

    if fig:
        output_path = output_dir / "accuracy_vs_stop_crit_scatter.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.show()

# %% [markdown]
# ## Best Run Selection + Distribution Visualization

# %%
if not df.empty:
    print("\n=== Best Run by Validation Loss ===")

    best_run = get_best_run_by_loss(df, LOSS_COL)

    if not best_run.empty:
        print(f"\nBest run: {best_run.get('run_name', best_run.get('run_id', 'unknown'))}")
        print(f"  Validation Loss: {best_run.get(LOSS_COL, np.nan):.6f}")
        print(f"  Clean Accuracy: {best_run.get(ACC_COL, np.nan):.4f}")

        for rob_col in ROB_COLS:
            if rob_col in best_run.index:
                strength = rob_col.split("/")[-1]
                print(f"  Robust Accuracy (ε={strength}): {best_run.get(rob_col, np.nan):.4f}")

        if MIA_COL and MIA_COL in best_run.index:
            print(f"  MIA Accuracy: {best_run.get(MIA_COL, np.nan):.4f}")

        # Distribution visualization for best run
        best_run_path = best_run.get("run_path")
        if best_run_path:
            print(f"\n--- Generating distribution visualization for best run ---")
            try:
                from analysis.visualize_distributions import visualize_from_run_dir

                fig = visualize_from_run_dir(
                    run_dir=best_run_path,
                    resolution=150,
                    normalize_joint=True,
                    show_data=True,
                    device=DEVICE,
                    save_dir=str(output_dir),
                )

                # Rename the output file
                default_path = output_dir / "distributions.png"
                final_path = output_dir / "best_run_distributions.png"
                if default_path.exists():
                    default_path.rename(final_path)
                    print(f"Saved distribution plot to: {final_path}")

                plt.show()

            except Exception as e:
                print(f"Warning: Could not generate distribution visualization: {e}")
        else:
            print("Warning: No run_path found for best run (wandb source?)")
    else:
        print("No valid runs found")

# %% [markdown]
# ## Final Summary Table

# %%
def create_summary_table(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: list = None,
    mia_col: str = None,
    effective_n: int = None,
) -> pd.DataFrame:
    """
    Create a summary table with best, mean, std, and stderr for all metrics.

    Args:
        df: DataFrame with metric columns
        acc_col: Column name for clean accuracy
        rob_cols: List of column names for robustness metrics
        mia_col: Column name for MIA accuracy (optional)
        effective_n: Override for sample size in stderr calculation

    Returns:
        Summary DataFrame
    """
    rows = []

    # Clean accuracy
    if acc_col in df.columns:
        stats = compute_statistics(df, acc_col, effective_n)
        rows.append({
            "Metric": "Clean Accuracy",
            "Best": stats["best"],
            "Mean": stats["mean"],
            "Std": stats["std"],
            "Std Error": stats["stderr"],
            "N": stats["n"],
        })

    # Robustness metrics
    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                stats = compute_statistics(df, rob_col, effective_n)
                strength = rob_col.split("/")[-1]
                rows.append({
                    "Metric": f"Robust Accuracy (ε={strength})",
                    "Best": stats["best"],
                    "Mean": stats["mean"],
                    "Std": stats["std"],
                    "Std Error": stats["stderr"],
                    "N": stats["n"],
                })

    # MIA
    if mia_col and mia_col in df.columns:
        stats = compute_statistics(df, mia_col, effective_n)
        rows.append({
            "Metric": "MIA Accuracy",
            "Best": stats["best"],
            "Mean": stats["mean"],
            "Std": stats["std"],
            "Std Error": stats["stderr"],
            "N": stats["n"],
        })

    return pd.DataFrame(rows)


# %%
if not df.empty:
    print("\n=== Summary Statistics ===")

    summary_df = create_summary_table(
        df,
        acc_col=ACC_COL,
        rob_cols=ROB_COLS,
        mia_col=MIA_COL,
        effective_n=EFFECTIVE_N,
    )

    if not summary_df.empty:
        print(f"\nEffective N: {EFFECTIVE_N if EFFECTIVE_N else len(df)} (actual runs: {len(df)})")
        print()
        print(summary_df.to_string(index=False))

        # Save to CSV
        csv_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSaved summary to: {csv_path}")

# %% [markdown]
# ## Export Full Summary

# %%
if not df.empty:
    summary_path = output_dir / "run_statistics_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Run Statistics Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Sweep: {sweep_name}\n")
        f.write(f"Data source: {DATA_SOURCE}\n")
        f.write(f"Regime: {REGIME}\n")
        f.write(f"Total runs: {len(df)}\n")
        if EFFECTIVE_N:
            f.write(f"Effective N: {EFFECTIVE_N}\n")
        f.write("\n")

        # Metrics used
        f.write("Metrics:\n")
        f.write(f"  Accuracy: {ACC_COL}\n")
        f.write(f"  Loss: {LOSS_COL}\n")
        if ROB_COLS:
            f.write(f"  Robustness: {ROB_COLS}\n")
        if MIA_COL:
            f.write(f"  MIA: {MIA_COL}\n")
        f.write("\n")

        # Summary statistics
        f.write("-" * 60 + "\n")
        f.write("Summary Statistics\n")
        f.write("-" * 60 + "\n\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        # Best run details
        f.write("-" * 60 + "\n")
        f.write("Best Run (by Validation Loss)\n")
        f.write("-" * 60 + "\n\n")

        best_run = get_best_run_by_loss(df, LOSS_COL)
        if not best_run.empty:
            f.write(f"Run: {best_run.get('run_name', best_run.get('run_id', 'unknown'))}\n")
            if "run_path" in best_run.index:
                f.write(f"Path: {best_run.get('run_path')}\n")
            f.write(f"Validation Loss: {best_run.get(LOSS_COL, np.nan):.6f}\n")
            f.write(f"Clean Accuracy: {best_run.get(ACC_COL, np.nan):.4f}\n")

            for rob_col in ROB_COLS:
                if rob_col in best_run.index:
                    strength = rob_col.split("/")[-1]
                    f.write(f"Robust Accuracy (ε={strength}): {best_run.get(rob_col, np.nan):.4f}\n")

            if MIA_COL and MIA_COL in best_run.index:
                f.write(f"MIA Accuracy: {best_run.get(MIA_COL, np.nan):.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Generated outputs:\n")
        f.write("  - accuracy_histogram.png\n")
        f.write("  - mean_accuracies_errorbars.png\n")
        f.write("  - accuracy_vs_loss_scatter.png\n")
        f.write("  - accuracy_vs_stop_crit_scatter.png\n")
        f.write("  - best_run_distributions.png\n")
        f.write("  - summary_statistics.csv\n")
        f.write("  - run_statistics_summary.txt\n")
        f.write("=" * 60 + "\n")

    print(f"\nExported full summary to: {summary_path}")

# %% [markdown]
# ## Completion

# %%
print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print(f"\nSweep: {sweep_name}")
print(f"Data source: {DATA_SOURCE}")
print(f"Regime: {REGIME}")
print(f"Total runs analyzed: {len(df) if not df.empty else 0}")
print(f"\nOutputs saved to: {output_dir}")
print("=" * 60)

# %%
