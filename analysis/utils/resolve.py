"""
HPO Analysis Resolver — convert shorthand regime/param names to full config paths.

Example usage:
    from analysis.utils.resolve import resolve_params, resolve_metrics, filter_varied_params

    params = resolve_params("dis", ["lr", "weight-decay"])
    # Returns: {"lr": "trainer.discriminative.optimizer.kwargs.lr", ...}

    metrics = resolve_metrics("dis", df)
    # Returns: {"validation": [...], "robustness": [...], "test": [...]}

    varied, excluded = filter_varied_params(df, param_columns)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import re

from src.utils.embeddings import _EMBEDDING_RANGE_SIZE, embedding_range_size


# =============================================================================
# REGIME PATH RESOLVER
# =============================================================================

# Matches regime tokens in output directory paths.  Old paths use "cls"; new
# paths use "dis".  Both are mapped to "dis".
_REGIME_PATH_PATTERN = re.compile(r"^(cls|dis)?(gen)?(adv)?$")


def resolve_regime_from_path(sweep_dir: str) -> Optional[str]:
    """Infer the training regime from a sweep directory path.

    The standard sweep output directory has the format::

        outputs/{experiment}/{training_regime}/{embedding}/d{in}r{bond}/{dataset}_{ddmm}

    where ``training_regime`` encodes the active trainers:

    * ``dis`` (or legacy ``cls``) — discriminative training
    * ``gen``  — generative NLL training
    * ``adv``  — adversarial training

    Priority when multiple codes are present: ``adv`` > ``gen`` > ``dis``.

    Args:
        sweep_dir: Path to the sweep directory (relative or absolute).

    Returns:
        One of ``"dis"``, ``"gen"``, ``"adv"``, or ``None``.

    Examples:
        >>> resolve_regime_from_path("outputs/seed_sweep/adv/fourier/d4r3/circles_0102")
        'adv'
        >>> resolve_regime_from_path("outputs/seed_sweep/dis/fourier/d4r3/circles_0102")
        'dis'
        >>> resolve_regime_from_path("outputs/seed_sweep/cls/fourier/d4r3/circles_0102")
        'dis'
    """
    path_str = str(sweep_dir).replace("\\", "/")
    tokens = re.split(r"[/_]", path_str)

    for token in tokens:
        m = _REGIME_PATH_PATTERN.match(token.lower())
        if m and m.group(0):
            if m.group(3):   # adv present
                return "adv"
            if m.group(2):   # gen present
                return "gen"
            if m.group(1):   # dis/cls present
                return "dis"

    return None


# =============================================================================
# EMBEDDING RESOLVER
# =============================================================================

_KNOWN_EMBEDDINGS = set(_EMBEDDING_RANGE_SIZE.keys())


def resolve_embedding_from_path(sweep_dir: str) -> Optional[str]:
    """Infer the embedding type from a sweep directory path.

    Returns one of the known embedding names or None.
    """
    path_str = str(sweep_dir).replace("\\", "/")
    tokens = re.split(r"[/_]", path_str)
    for token in tokens:
        if token.lower() in _KNOWN_EMBEDDINGS:
            return token.lower()
    return None


# Re-export for callers that import directly from this module.
__all_range = ["_EMBEDDING_RANGE_SIZE", "embedding_range_size"]


# =============================================================================
# PARAMETER ALIASES
# =============================================================================

PARAM_ALIASES: Dict[str, str] = {
    "learning-rate":  "lr",
    "learning_rate":  "lr",
    "learningrate":   "lr",
    "wd":             "weight-decay",
    "weight_decay":   "weight-decay",
    "weightdecay":    "weight-decay",
    "batch_size":     "batch-size",
    "batchsize":      "batch-size",
    "bs":             "batch-size",
    "bond_dim":       "bond-dim",
    "bonddim":        "bond-dim",
    "in_dim":         "in-dim",
    "indim":          "in-dim",
    "clean_weight":   "clean-weight",
    "cleanweight":    "clean-weight",
    "strength":       "epsilon",
    "eps":            "epsilon",
    "data_seed":      "data-seed",
    "dataseed":       "data-seed",
    "split_seed":     "split-seed",
    "splitseed":      "split-seed",
}


# =============================================================================
# REGIME PARAMETER MAPPINGS
# =============================================================================

REGIME_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "dis": {
        "lr":           "trainer.discriminative.optimizer.kwargs.lr",
        "weight-decay": "trainer.discriminative.optimizer.kwargs.weight_decay",
        "batch-size":   "trainer.discriminative.batch_size",
        "bond-dim":     "born.init_kwargs.bond_dim",
        "in-dim":       "born.init_kwargs.in_dim",
        "seed":         "tracking.seed",
        "data-seed":    "dataset.gen_dow_kwargs.seed",
        "split-seed":   "dataset.split_seed",
    },
    "gen": {
        "lr":           "trainer.generative.optimizer.kwargs.lr",
        "weight-decay": "trainer.generative.optimizer.kwargs.weight_decay",
        "batch-size":   "trainer.generative.batch_size",
        "bond-dim":     "born.init_kwargs.bond_dim",
        "in-dim":       "born.init_kwargs.in_dim",
        "seed":         "tracking.seed",
        "data-seed":    "dataset.gen_dow_kwargs.seed",
        "split-seed":   "dataset.split_seed",
    },
    "adv": {
        "lr":           "trainer.adversarial.optimizer.kwargs.lr",
        "weight-decay": "trainer.adversarial.optimizer.kwargs.weight_decay",
        "batch-size":   "trainer.adversarial.batch_size",
        "epsilon":      "trainer.adversarial.evasion.strengths",
        "clean-weight": "trainer.adversarial.clean_weight",
        "bond-dim":     "born.init_kwargs.bond_dim",
        "in-dim":       "born.init_kwargs.in_dim",
        "seed":         "tracking.seed",
        "data-seed":    "dataset.gen_dow_kwargs.seed",
        "split-seed":   "dataset.split_seed",
    },
}


# =============================================================================
# REGIME METADATA
# =============================================================================

REGIME_DESCRIPTIONS: Dict[str, str] = {
    "dis": "Discriminative training",
    "gen": "Generative NLL training",
    "adv": "Adversarial training",
}

REGIME_METRIC_PREFIX: Dict[str, str] = {
    "dis": "dis",
    "gen": "gen",
    "adv": "adv",
}

REGIME_DEFAULT_PARAMS: Dict[str, List[str]] = {
    "dis": ["lr", "weight-decay", "batch-size"],
    "gen": ["lr", "weight-decay", "batch-size"],
    "adv": ["lr", "weight-decay", "epsilon", "clean-weight"],
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_param(name: str) -> str:
    """Normalize parameter name using aliases."""
    name = name.lower().strip().replace(" ", "-")
    return PARAM_ALIASES.get(name, name)


def resolve_params(
    regime: str,
    shorthands: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Resolve shorthand param names to full config paths.

    Args:
        regime: Training regime ("dis", "gen", "adv")
        shorthands: List of param shorthand names, or None for regime defaults

    Returns:
        Dict mapping shorthand → full config path

    Raises:
        ValueError: If regime unknown or shorthand not found for regime
    """
    regime = regime.lower().strip()

    if regime not in REGIME_PARAM_MAP:
        available = ", ".join(REGIME_PARAM_MAP.keys())
        raise ValueError(f"Unknown regime '{regime}'. Available: {available}")

    regime_params = REGIME_PARAM_MAP[regime]

    if shorthands is None:
        shorthands = REGIME_DEFAULT_PARAMS.get(regime, list(regime_params.keys()))

    result = {}
    unknown = []

    for name in shorthands:
        canonical = normalize_param(name)
        if canonical in regime_params:
            result[canonical] = regime_params[canonical]
        else:
            unknown.append(name)

    if unknown:
        available = ", ".join(regime_params.keys())
        raise ValueError(
            f"Unknown params for regime '{regime}': {unknown}. "
            f"Available: {available}"
        )

    return result


def config_path_to_column(full_path: str) -> str:
    """Convert full config path to DataFrame column name.

    Example:
        >>> config_path_to_column("trainer.discriminative.optimizer.kwargs.lr")
        "config/lr"
    """
    short_key = full_path.split(".")[-1]
    if short_key == "kwargs":
        parts = full_path.split(".")
        if len(parts) >= 2:
            short_key = parts[-2]
    return f"config/{short_key}"


def resolve_metrics(
    regime: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, List[str]]:
    """Resolve metric column names for a regime (W&B summary key format)."""
    regime = regime.lower().strip()

    if regime not in REGIME_METRIC_PREFIX:
        available = ", ".join(REGIME_METRIC_PREFIX.keys())
        raise ValueError(f"Unknown regime '{regime}'. Available: {available}")

    prefix = REGIME_METRIC_PREFIX[regime]

    validation = [
        f"summary/{prefix}/valid/acc",
        f"summary/{prefix}/valid/loss",
    ]
    test = [
        f"summary/{prefix}/test/acc",
        f"summary/{prefix}/test/loss",
    ]

    robustness = []
    if df is not None:
        strengths = detect_robustness_strengths(df, prefix)
        robustness = [f"summary/{prefix}/valid/rob/{s}" for s in strengths]
    else:
        robustness = [
            f"summary/{prefix}/valid/rob/0.1",
            f"summary/{prefix}/valid/rob/0.3",
        ]

    return {"validation": validation, "robustness": robustness, "test": test}


def resolve_primary_metric(
    shorthand: str,
    validation_metrics: List[str],
    robustness_metrics: List[str],
    test_metrics: List[str],
) -> Tuple[Optional[str], bool]:
    """Resolve a PRIMARY_METRIC shorthand to a full column name."""
    all_metrics = validation_metrics + robustness_metrics + test_metrics
    key = shorthand.strip().lower()

    if key in ("rob", "robustness"):
        return (robustness_metrics[0], False) if robustness_metrics else (None, False)

    if key in ("acc", "accuracy"):
        match = next((m for m in validation_metrics if "acc" in m), None)
        return (match, False) if match else (None, False)

    if key == "loss":
        match = next((m for m in validation_metrics if "loss" in m), None)
        return (match, True) if match else (None, False)

    if key in ("test-acc", "test_acc"):
        match = next((m for m in test_metrics if "acc" in m), None)
        return (match, False) if match else (None, False)

    if key in ("test-loss", "test_loss"):
        match = next((m for m in test_metrics if "loss" in m), None)
        return (match, True) if match else (None, False)

    if shorthand in all_metrics:
        return shorthand, "loss" in shorthand

    matches = [m for m in all_metrics if shorthand in m]
    if matches:
        resolved = matches[0]
        return resolved, "loss" in resolved

    return None, False


def detect_robustness_strengths(df: pd.DataFrame, prefix: str) -> List[float]:
    """Auto-detect robustness metric strengths from DataFrame columns."""
    pattern = re.compile(rf"summary/{prefix}/valid/rob/([0-9.]+)")
    strengths = set()
    for col in df.columns:
        match = pattern.match(col)
        if match:
            try:
                strengths.add(float(match.group(1)))
            except ValueError:
                pass
    return sorted(strengths)


def filter_varied_params(
    df: pd.DataFrame,
    param_columns: List[str],
    min_unique: int = 2,
) -> Tuple[List[str], List[str]]:
    """Filter to only parameters that varied in the sweep."""
    varied = []
    excluded = []
    for col in param_columns:
        if col not in df.columns:
            excluded.append(col)
            continue
        col_hashable = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        n_unique = col_hashable.nunique(dropna=True)
        if n_unique >= min_unique:
            varied.append(col)
        else:
            excluded.append(col)
    return varied, excluded


def detect_pretrained_info(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Check if a pretrained model was loaded and extract info."""
    model_path_col = "config/model_path"
    if model_path_col not in df.columns:
        return None
    paths = df[model_path_col].dropna().unique()
    if len(paths) == 0:
        return None
    return {
        "model_paths": list(paths),
        "n_runs_with_pretrained": df[model_path_col].notna().sum(),
    }


def format_resolved_config(
    regime: str,
    params: Dict[str, str],
    varied_params: List[str],
    excluded_params: List[str],
    metrics: Dict[str, List[str]],
    pretrained_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Format resolved configuration for display."""
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append("Resolved Configuration")
    lines.append(sep)
    lines.append("")

    desc = REGIME_DESCRIPTIONS.get(regime, regime)
    lines.append(f"Regime: {regime} ({desc})")
    lines.append("")

    n_varied = len(varied_params)
    n_excluded = len(excluded_params)
    lines.append(f"Parameters ({n_varied} varied, {n_excluded} excluded):")

    for shorthand in varied_params:
        full_path = params.get(shorthand, "?")
        lines.append(f"  + {shorthand:<16} -> {full_path}")
    for shorthand in excluded_params:
        lines.append(f"  - {shorthand:<16} (excluded: single value)")

    lines.append("")
    lines.append("Metrics:")

    def format_metric_list(metric_list: List[str], max_show: int = 4) -> str:
        if not metric_list:
            return "(none)"
        short_names = [m.split("/")[-1] for m in metric_list[:max_show]]
        result = ", ".join(short_names)
        if len(metric_list) > max_show:
            result += f", ... (+{len(metric_list) - max_show} more)"
        return result

    lines.append(f"  Validation: {format_metric_list(metrics.get('validation', []))}")
    if metrics.get("robustness"):
        rob_names = [m.split("/")[-1] for m in metrics["robustness"]]
        lines.append(f"  Robustness: {', '.join(rob_names)} (auto-detected)")
    else:
        lines.append("  Robustness: (none detected)")
    lines.append(f"  Test:       {format_metric_list(metrics.get('test', []))}")
    lines.append("")

    if pretrained_info:
        n_runs = pretrained_info.get("n_runs_with_pretrained", 0)
        lines.append(f"Pretrained Model: detected in {n_runs} runs")
    else:
        lines.append("Pretrained Model: None detected")

    lines.append(sep)
    return "\n".join(lines)


def get_available_params(regime: str) -> List[str]:
    """Get list of available parameter shorthand names for a regime."""
    regime = regime.lower().strip()
    if regime not in REGIME_PARAM_MAP:
        return []
    return list(REGIME_PARAM_MAP[regime].keys())


def get_available_regimes() -> List[str]:
    """Get list of available training regimes."""
    return list(REGIME_PARAM_MAP.keys())


def resolve_stop_criterion(
    stop_crit: str,
    df: pd.DataFrame,
) -> Tuple[Optional[str], bool, str]:
    """Map a stop-criterion name to the corresponding eval CSV column.

    With flat (test-only) column names the mapping is direct.

    Returns:
        (column_name, minimize, label) or (None, True, "") if unresolvable.
    """
    if stop_crit == "acc":
        col = "acc"
        return (col, False, "Accuracy") if col in df.columns else (None, True, "")
    if stop_crit in ("loss", "dis_loss"):
        col = "dis_loss"
        return (col, True, "Dis Loss") if col in df.columns else (None, True, "")
    if stop_crit == "gen_loss":
        col = "gen_loss"
        return (col, True, "Gen Loss") if col in df.columns else (None, True, "")
    if stop_crit == "rob":
        rob_cols = [c for c in df.columns if c.startswith("rob/")]
        if rob_cols:
            df["_avg_rob_acc"] = df[rob_cols].mean(axis=1)
            return ("_avg_rob_acc", False, "Avg Robust Accuracy")
        return (None, True, "")
    return (None, True, "")
