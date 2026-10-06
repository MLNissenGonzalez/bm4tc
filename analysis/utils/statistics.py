"""
Shared statistics and visualization functions for sweep analysis.

Extracted from run_statistics.py and hpo_analysis.py to enable reuse
across analysis notebooks.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


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
