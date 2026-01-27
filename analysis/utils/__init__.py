"""Analysis utilities for post-experiment analysis."""

from .wandb_fetcher import (
    WandbFetcher,
    fetch_hpo_runs,
    get_hpo_metrics_summary,
    load_local_hpo_runs,
    load_local_run,
    find_local_sweep_dirs,
)

from .mia import (
    MIAFeatureConfig,
    MIAFeatureExtractor,
    MIAResults,
    MIAEvaluation,
)

from .mia_utils import (
    load_run_config,
    load_run_config_from_wandb,
    find_model_checkpoint,
    download_wandb_checkpoint,
)

from .resolve import (
    resolve_params,
    resolve_metrics,
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
    # MIA evaluation
    "MIAFeatureConfig",
    "MIAFeatureExtractor",
    "MIAResults",
    "MIAEvaluation",
    # Config loading utilities
    "load_run_config",
    "load_run_config_from_wandb",
    "find_model_checkpoint",
    "download_wandb_checkpoint",
    # HPO resolver utilities
    "resolve_params",
    "resolve_metrics",
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
]
