"""Analysis utilities for post-experiment analysis."""

from .wandb_fetcher import (
    WandbFetcher,
    fetch_hpo_runs,
    get_hpo_metrics_summary,
    load_local_hpo_runs,
    load_local_run,
    find_local_sweep_dirs,
    # Best run fetching for downstream training
    fetch_best_classification_run,
    extract_classification_config,
    extract_born_config,
    extract_dataset_config,
    get_best_classification_config,
    print_classification_config_yaml,
)

from .mia import (
    MIAFeatureConfig,
    MIAFeatureExtractor,
    MIAResults,
    MIAEvaluation,
)

from .uq import (
    UQConfig,
    UQResults,
    UQEvaluation,
    PurificationMetrics,
    compute_thresholds,
    compute_log_px,
)

from .mia_utils import (
    load_run_config,
    load_run_config_from_wandb,
    find_model_checkpoint,
    download_wandb_checkpoint,
)

from .statistics import (
    clean_column_name,
    compute_statistics,
    get_best_run,
    create_summary_table,
    compute_pareto_frontier,
    get_pareto_runs,
    compute_metric_correlations,
    plot_accuracy_histogram,
    plot_mean_with_std,
    plot_scatter_vs_metric,
    plot_accuracy_vs_strength,
    plot_accuracy_vs_strength_band,
    plot_pareto_frontier,
    plot_correlation_heatmap,
)

from .evaluate import (
    EvalConfig,
    evaluate_run,
    evaluate_sweep,
    resolve_stop_criterion,
)

from .resolve import (
    resolve_params,
    resolve_metrics,
    resolve_primary_metric,
    filter_varied_params,
    format_resolved_config,
    config_path_to_column,
    detect_robustness_strengths,
    detect_pretrained_info,
    normalize_param,
    get_available_params,
    get_available_regimes,
    REGIME_PARAM_MAP,
    REGIME_DESCRIPTIONS,
    REGIME_METRIC_PREFIX,
    REGIME_DEFAULT_PARAMS,
    PARAM_ALIASES,
)

__all__ = [
    # Wandb/HPO utilities
    "WandbFetcher",
    "fetch_hpo_runs",
    "get_hpo_metrics_summary",
    "load_local_hpo_runs",
    "load_local_run",
    "find_local_sweep_dirs",
    # Best run fetching for downstream training
    "fetch_best_classification_run",
    "extract_classification_config",
    "extract_born_config",
    "extract_dataset_config",
    "get_best_classification_config",
    "print_classification_config_yaml",
    # MIA evaluation
    "MIAFeatureConfig",
    "MIAFeatureExtractor",
    "MIAResults",
    "MIAEvaluation",
    # UQ evaluation
    "UQConfig",
    "UQResults",
    "UQEvaluation",
    "PurificationMetrics",
    "compute_thresholds",
    "compute_log_px",
    # Config loading utilities
    "load_run_config",
    "load_run_config_from_wandb",
    "find_model_checkpoint",
    "download_wandb_checkpoint",
    # HPO resolver utilities
    "resolve_params",
    "resolve_metrics",
    "resolve_primary_metric",
    "filter_varied_params",
    "format_resolved_config",
    "config_path_to_column",
    "detect_robustness_strengths",
    "detect_pretrained_info",
    "normalize_param",
    "get_available_params",
    "get_available_regimes",
    "REGIME_PARAM_MAP",
    "REGIME_DESCRIPTIONS",
    "REGIME_METRIC_PREFIX",
    "REGIME_DEFAULT_PARAMS",
    "PARAM_ALIASES",
    # Statistics & visualization
    "clean_column_name",
    "compute_statistics",
    "get_best_run",
    "create_summary_table",
    "compute_pareto_frontier",
    "get_pareto_runs",
    "compute_metric_correlations",
    "plot_accuracy_histogram",
    "plot_mean_with_std",
    "plot_scatter_vs_metric",
    "plot_accuracy_vs_strength",
    "plot_accuracy_vs_strength_band",
    "plot_pareto_frontier",
    "plot_correlation_heatmap",
    # Post-hoc evaluation
    "EvalConfig",
    "evaluate_run",
    "evaluate_sweep",
    "resolve_stop_criterion",
]
