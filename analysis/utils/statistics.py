"""
Shared statistics and visualization functions for sweep analysis.

Extracted from run_statistics.py and hpo_analysis.py to enable reuse
across analysis notebooks.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_column_name(col: str) -> str:
    """Convert column name to readable label.

    Args:
        col: Raw column name (e.g. "config/lr" or "summary/pre/valid/acc").

    Returns:
        Human-readable label.
    """
    name = col.replace("config/", "").replace("summary/", "").replace("eval/", "")
    name = name.replace("_", " ").replace("/", " / ")
    return name.title()


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    df: pd.DataFrame,
    metric_col: str,
    effective_n: Optional[int] = None,
) -> dict:
    """Compute best, mean, std, and stderr for a metric.

    Args:
        df: DataFrame containing *metric_col*.
        metric_col: Column name for the metric.
        effective_n: Override sample size for stderr calculation.

    Returns:
        Dict with keys best, mean, std, stderr, n.
    """
    if metric_col not in df.columns:
        return {"best": np.nan, "mean": np.nan, "std": np.nan, "stderr": np.nan, "n": 0}

    values = df[metric_col].dropna()
    n = len(values)

    if n == 0:
        return {"best": np.nan, "mean": np.nan, "std": np.nan, "stderr": np.nan, "n": 0}

    n_for_stderr = effective_n if effective_n is not None else n

    return {
        "best": values.max(),
        "mean": values.mean(),
        "std": values.std(),
        "stderr": values.std() / np.sqrt(n_for_stderr) if n_for_stderr > 0 else np.nan,
        "n": n,
    }


def get_best_run(
    df: pd.DataFrame,
    metric_col: str,
    minimize: bool = True,
) -> Optional[pd.Series]:
    """Get the run with the best value of a given metric.

    Args:
        df: DataFrame with metric column.
        metric_col: Column to optimise.
        minimize: If True select lowest value, else highest.

    Returns:
        Series representing the best run, or None.
    """
    if metric_col not in df.columns:
        return None
    valid_df = df[df[metric_col].notna()]
    if valid_df.empty:
        return None
    best_idx = valid_df[metric_col].idxmin() if minimize else valid_df[metric_col].idxmax()
    return df.loc[best_idx]


def create_summary_table(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: Optional[List[str]] = None,
    mia_col: Optional[str] = None,
    effective_n: Optional[int] = None,
    stop_crit_col: Optional[str] = None,
    stop_crit_minimize: bool = True,
) -> pd.DataFrame:
    """Create a summary table with best, mean, std, and stderr for all metrics.

    The "Best" column shows values from the single run that achieved the best
    stopping criterion value, falling back to per-column best when
    *stop_crit_col* is not provided.

    Args:
        df: DataFrame with metric columns.
        acc_col: Column for clean accuracy.
        rob_cols: Columns for robustness metrics.
        mia_col: Column for MIA accuracy (optional).
        effective_n: Override for sample size in stderr.
        stop_crit_col: Column used as stopping criterion.
        stop_crit_minimize: Whether to minimise the stop criterion.

    Returns:
        Summary DataFrame.
    """
    best_run = None
    if stop_crit_col and stop_crit_col in df.columns:
        best_run = get_best_run(df, stop_crit_col, minimize=stop_crit_minimize)

    def _best_val(metric_col, stats):
        if best_run is not None and metric_col in best_run.index and pd.notna(best_run[metric_col]):
            return best_run[metric_col]
        return stats["best"]

    rows = []

    if acc_col in df.columns:
        stats = compute_statistics(df, acc_col, effective_n)
        rows.append({
            "Metric": "Clean Accuracy",
            "Best": _best_val(acc_col, stats),
            "Mean": stats["mean"],
            "Std": stats["std"],
            "Std Error": stats["stderr"],
            "N": stats["n"],
        })

    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                stats = compute_statistics(df, rob_col, effective_n)
                strength = rob_col.split("/")[-1]
                rows.append({
                    "Metric": f"Robust Accuracy (eps={strength})",
                    "Best": _best_val(rob_col, stats),
                    "Mean": stats["mean"],
                    "Std": stats["std"],
                    "Std Error": stats["stderr"],
                    "N": stats["n"],
                })

    if mia_col and mia_col in df.columns:
        stats = compute_statistics(df, mia_col, effective_n)
        rows.append({
            "Metric": "MIA Accuracy",
            "Best": _best_val(mia_col, stats),
            "Mean": stats["mean"],
            "Std": stats["std"],
            "Std Error": stats["stderr"],
            "N": stats["n"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    x: np.ndarray,
    y: np.ndarray,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """Compute boolean mask of Pareto-optimal points.

    Args:
        x, y: Arrays of metric values.
        maximize_x, maximize_y: Whether higher is better for each metric.

    Returns:
        Boolean array indicating Pareto-optimal points.
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    x_comp = x if maximize_x else -x
    y_comp = y if maximize_y else -y

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (x_comp[j] >= x_comp[i] and y_comp[j] >= y_comp[i] and
                    (x_comp[j] > x_comp[i] or y_comp[j] > y_comp[i])):
                is_pareto[i] = False
                break

    return is_pareto


def get_pareto_runs(
    df: pd.DataFrame,
    metric1: str,
    metric2: str,
    maximize1: bool = True,
    maximize2: bool = True,
) -> pd.DataFrame:
    """Return DataFrame of Pareto-optimal runs sorted by *metric1*.

    Args:
        df: DataFrame with metric columns.
        metric1, metric2: Column names for the two objectives.
        maximize1, maximize2: Whether higher is better.

    Returns:
        Filtered DataFrame of Pareto-optimal runs.
    """
    if metric1 not in df.columns or metric2 not in df.columns:
        return pd.DataFrame()

    valid_mask = df[metric1].notna() & df[metric2].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) < 2:
        return pd.DataFrame()

    x = valid_df[metric1].astype(float).values
    y = valid_df[metric2].astype(float).values
    is_pareto = compute_pareto_frontier(x, y, maximize1, maximize2)

    return valid_df[is_pareto].sort_values(metric1, ascending=not maximize1)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def compute_metric_correlations(
    df: pd.DataFrame,
    metrics: List[str],
) -> pd.DataFrame:
    """Compute Pearson correlations between all pairs of metrics.

    Args:
        df: DataFrame with metric columns.
        metrics: List of metric column names.

    Returns:
        Correlation matrix DataFrame (metrics x metrics).
    """
    metrics = [m for m in metrics if m in df.columns]

    if len(metrics) < 2:
        return pd.DataFrame()

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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_accuracy_histogram(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: Optional[List[str]] = None,
    mia_col: Optional[str] = None,
    title: str = "Accuracy Distribution",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
) -> plt.Figure:
    """Plot histograms of accuracy metrics.

    Args:
        df: DataFrame with accuracy columns.
        acc_col: Column for clean accuracy.
        rob_cols: Columns for robustness metrics.
        mia_col: Column for MIA accuracy.
        title: Plot title.
        figsize: Base figure size (width per subplot is 5).
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure.
    """
    n_plots = 1
    if rob_cols:
        n_plots += len([c for c in rob_cols if c in df.columns])
    if mia_col and mia_col in df.columns:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    plot_idx = 0

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
                    ax.set_title(f"Robust Acc (eps={strength})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                plot_idx += 1

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


def plot_mean_with_std(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: Optional[List[str]] = None,
    mia_col: Optional[str] = None,
    title: str = "Mean Accuracies",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
) -> Optional[plt.Figure]:
    """Plot mean accuracies with standard deviation error bars.

    Args:
        df: DataFrame with accuracy columns.
        acc_col: Column for clean accuracy.
        rob_cols: Columns for robustness metrics.
        mia_col: Column for MIA accuracy.
        title: Plot title.
        figsize: Figure size.
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure, or None if no valid metrics.
    """
    metrics = []
    means = []
    stds = []
    colors = []

    if acc_col in df.columns:
        stats = compute_statistics(df, acc_col)
        metrics.append("Clean\nAccuracy")
        means.append(stats["mean"])
        stds.append(stats["std"])
        colors.append("steelblue")

    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns:
                stats = compute_statistics(df, rob_col)
                strength = rob_col.split("/")[-1]
                metrics.append(f"Robust\n(eps={strength})")
                means.append(stats["mean"])
                stds.append(stats["std"])
                colors.append("orange")

    if mia_col and mia_col in df.columns:
        stats = compute_statistics(df, mia_col)
        metrics.append("MIA\nAccuracy")
        means.append(stats["mean"])
        stds.append(stats["std"])
        colors.append("purple")

    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", alpha=0.8)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_scatter_vs_metric(
    df: pd.DataFrame,
    x_col: str,
    acc_col: str,
    rob_cols: Optional[List[str]] = None,
    mia_col: Optional[str] = None,
    x_label: str = "X",
    title: str = "Accuracy vs Metric",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
) -> Optional[plt.Figure]:
    """Scatter plot of accuracy metrics against an arbitrary x-axis metric.

    Unifies the old ``plot_accuracy_vs_loss_scatter`` and
    ``plot_accuracy_vs_stop_crit_scatter``.

    Args:
        df: DataFrame with metric columns.
        x_col: Column for the x-axis metric.
        acc_col: Column for clean accuracy.
        rob_cols: Columns for robustness metrics.
        mia_col: Column for MIA accuracy.
        x_label: Label for the x-axis.
        title: Plot title.
        figsize: Figure size (width per subplot is 5).
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure, or None if x_col missing.
    """
    if x_col not in df.columns:
        return None

    n_plots = 1
    if rob_cols:
        n_plots += len([c for c in rob_cols if c in df.columns])
    if mia_col and mia_col in df.columns:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    plot_idx = 0
    x_vals = df[x_col]

    def _scatter(ax, y_col, color, y_label, subplot_title):
        nonlocal plot_idx
        if y_col not in df.columns:
            return
        valid_mask = x_vals.notna() & df[y_col].notna()
        if valid_mask.sum() == 0:
            return
        ax.scatter(
            x_vals[valid_mask], df.loc[valid_mask, y_col],
            alpha=0.6, c=color, edgecolors="white", s=50,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(subplot_title)
        ax.grid(True, alpha=0.3)
        if valid_mask.sum() > 2:
            corr = np.corrcoef(x_vals[valid_mask], df.loc[valid_mask, y_col])[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment="top")
        plot_idx += 1

    _scatter(axes[plot_idx], acc_col, "steelblue", "Accuracy", "Clean Accuracy")

    if rob_cols:
        for rob_col in rob_cols:
            if rob_col in df.columns and plot_idx < len(axes):
                strength = rob_col.split("/")[-1]
                _scatter(axes[plot_idx], rob_col, "orange", "Robust Accuracy", f"Robust Acc (eps={strength})")

    if mia_col and mia_col in df.columns and plot_idx < len(axes):
        _scatter(axes[plot_idx], mia_col, "purple", "MIA Accuracy", "MIA Accuracy")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_accuracy_vs_strength(
    runs: pd.DataFrame,
    acc_col: str,
    rob_cols: List[str],
    run_labels: Optional[List[str]] = None,
    title: str = "Accuracy vs Perturbation Strength",
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 100,
) -> Optional[plt.Figure]:
    """Plot (robust) accuracy as a function of attack strength for one or more runs.

    Always includes zero strength (clean accuracy) as the first point.

    Args:
        runs: DataFrame where each row is a run to plot.
        acc_col: Column for clean accuracy (used as the eps=0 point).
        rob_cols: Columns for robustness at different strengths.
            Strengths are parsed from column names like ``eval/test/rob/0.15``.
        run_labels: Optional labels for each run (defaults to run_name or index).
        title: Plot title.
        figsize: Figure size.
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure, or None if insufficient data.
    """
    if acc_col not in runs.columns or not rob_cols:
        return None

    # Parse strengths from column names and sort
    strength_col_pairs = []
    for col in rob_cols:
        try:
            strength = float(col.split("/")[-1])
            if col in runs.columns:
                strength_col_pairs.append((strength, col))
        except ValueError:
            continue

    if not strength_col_pairs:
        return None

    strength_col_pairs.sort(key=lambda sc: sc[0])
    strengths = [0.0] + [s for s, _ in strength_col_pairs]
    cols = [acc_col] + [c for _, c in strength_col_pairs]

    # Build labels
    if run_labels is None:
        if "run_name" in runs.columns:
            run_labels = [str(n) for n in runs["run_name"].values]
        else:
            run_labels = [str(i) for i in range(len(runs))]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = plt.cm.tab10
    for idx, (_, run) in enumerate(runs.iterrows()):
        accs = [float(run[c]) if pd.notna(run[c]) else np.nan for c in cols]
        color = cmap(idx % 10)
        label = run_labels[idx] if idx < len(run_labels) else str(idx)
        ax.plot(strengths, accs, "o-", color=color, label=f"Run {label}",
                markersize=6, linewidth=1.5)

    ax.set_xlabel("Perturbation Strength (epsilon)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_accuracy_vs_strength_band(
    df: pd.DataFrame,
    acc_col: str,
    rob_cols: List[str],
    n_sigma: float = 2.0,
    title: str = "Mean Accuracy vs Perturbation Strength",
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 100,
) -> Optional[plt.Figure]:
    """Plot mean accuracy +/- n*std as a shaded band across perturbation strengths.

    Args:
        df: DataFrame where each row is a run.
        acc_col: Column for clean accuracy (used as eps=0 point).
        rob_cols: Columns for robustness at different strengths.
        n_sigma: Number of standard deviations for the band.
        title: Plot title.
        figsize: Figure size.
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure, or None if insufficient data.
    """
    if acc_col not in df.columns or not rob_cols:
        return None

    # Parse strengths from column names and sort
    strength_col_pairs = []
    for col in rob_cols:
        try:
            strength = float(col.split("/")[-1])
            if col in df.columns:
                strength_col_pairs.append((strength, col))
        except ValueError:
            continue

    if not strength_col_pairs:
        return None

    strength_col_pairs.sort(key=lambda sc: sc[0])
    strengths = [0.0] + [s for s, _ in strength_col_pairs]
    cols = [acc_col] + [c for _, c in strength_col_pairs]

    # Compute mean and std at each strength
    means = []
    stds = []
    for col in cols:
        vals = df[col].dropna()
        means.append(vals.mean() if len(vals) > 0 else np.nan)
        stds.append(vals.std() if len(vals) > 1 else 0.0)

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(strengths, means, "o-", color="steelblue", linewidth=2, markersize=6, label="Mean")
    ax.fill_between(
        strengths, means - n_sigma * stds, means + n_sigma * stds,
        alpha=0.25, color="steelblue", label=f"\u00b1{n_sigma:.0f}\u03c3",
    )

    ax.set_xlabel("Perturbation Strength (epsilon)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pareto_frontier(
    df: pd.DataFrame,
    metric1: str,
    metric2: str,
    maximize1: bool = True,
    maximize2: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    show_labels: bool = True,
    dpi: int = 100,
) -> Optional[plt.Figure]:
    """Scatter plot of two metrics with Pareto frontier highlighted.

    Args:
        df: DataFrame with metric columns.
        metric1, metric2: Column names (x and y axis).
        maximize1, maximize2: Whether higher is better.
        figsize: Figure size.
        show_labels: Label Pareto-optimal points with run names.
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure, or None if insufficient data.
    """
    if metric1 not in df.columns or metric2 not in df.columns:
        return None

    valid_mask = df[metric1].notna() & df[metric2].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) < 2:
        return None

    x = valid_df[metric1].astype(float).values
    y = valid_df[metric2].astype(float).values

    is_pareto = compute_pareto_frontier(x, y, maximize1, maximize2)
    n_pareto = is_pareto.sum()

    pareto_x = x[is_pareto]
    pareto_y = y[is_pareto]
    sort_idx = np.argsort(pareto_x)
    pareto_x_sorted = pareto_x[sort_idx]
    pareto_y_sorted = pareto_y[sort_idx]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(x[~is_pareto], y[~is_pareto], c="lightgray", s=40, alpha=0.6,
               edgecolors="white", linewidth=0.5, label=f"Dominated (n={len(x) - n_pareto})")
    ax.scatter(pareto_x, pareto_y, c="red", s=80, alpha=0.9,
               edgecolors="darkred", linewidth=1, label=f"Pareto-optimal (n={n_pareto})", zorder=5)
    ax.plot(pareto_x_sorted, pareto_y_sorted, "r--", alpha=0.7, linewidth=2, zorder=4)

    if show_labels and "run_name" in valid_df.columns and n_pareto <= 15:
        pareto_names = valid_df.loc[valid_df.index[is_pareto], "run_name"].values
        for i, (px, py) in enumerate(zip(pareto_x, pareto_y)):
            name = pareto_names[i] if i < len(pareto_names) else ""
            short_name = name.split("-")[-1][:8] if "-" in name else name[:8]
            ax.annotate(short_name, (px, py), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.8)

    dir1 = "\u2191 better" if maximize1 else "\u2193 better"
    dir2 = "\u2191 better" if maximize2 else "\u2193 better"
    ax.set_xlabel(f"{clean_column_name(metric1)} ({dir1})")
    ax.set_ylabel(f"{clean_column_name(metric2)} ({dir2})")
    ax.set_title(f"Pareto Frontier: {clean_column_name(metric1)} vs {clean_column_name(metric2)}")

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    stats_text = f"Total runs: {len(x)}\nPareto-optimal: {n_pareto} ({100 * n_pareto / len(x):.1f}%)"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlations",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
) -> Optional[plt.Figure]:
    """Plot correlation matrix as heatmap with annotated values.

    Args:
        corr_df: Correlation DataFrame (square).
        title: Plot title.
        figsize: Figure size.
        dpi: Figure DPI.

    Returns:
        Matplotlib Figure, or None if empty.
    """
    if corr_df.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=9)

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
