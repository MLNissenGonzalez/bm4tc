"""Analysis utilities for post-experiment analysis."""

from .runs import (
    load_run_config,
    find_model_checkpoint,
)

from .statistics import (
    clean_column_name,
    compute_statistics,
    get_best_run,
    create_summary_table,
    compute_pareto_frontier,
    get_pareto_runs,
    compute_metric_correlations,
)

from .resolve import (
    resolve_regime_from_path,
    resolve_embedding_from_path,
    embedding_range_size,
)

__all__ = [
    # Run loading
    "load_run_config",
    "find_model_checkpoint",
    # Statistics
    "clean_column_name",
    "compute_statistics",
    "get_best_run",
    "create_summary_table",
    "compute_pareto_frontier",
    "get_pareto_runs",
    "compute_metric_correlations",
    # Path resolvers
    "resolve_regime_from_path",
    "resolve_embedding_from_path",
    "embedding_range_size",
]
