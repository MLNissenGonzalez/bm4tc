"""
HPO Analysis Resolver - Convert shorthand regime/param names to full config paths.

This module provides a user-friendly interface for configuring HPO analysis by
allowing users to specify training regimes and parameter shorthand names instead
of full config paths.

Example usage:
    from analysis.utils.resolve import resolve_params, resolve_metrics, filter_varied_params

    # Resolve params for classification training
    params = resolve_params("pre", ["lr", "weight-decay"])
    # Returns: {"lr": "trainer.classification.optimizer.kwargs.lr", ...}

    # Get metrics for the regime
    metrics = resolve_metrics("pre", df)
    # Returns: {"validation": [...], "robustness": [...], "test": [...]}

    # Filter to only params that varied
    varied, excluded = filter_varied_params(df, param_columns)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import re


# =============================================================================
# REGIME PATH RESOLVER
# =============================================================================

# Matches exactly the strings produced by the ${training_regime:} OmegaConf
# resolver: any concatenation of the four code tokens cls / gen / adv / gan.
_REGIME_PATH_PATTERN = re.compile(r"^(cls)?(gen)?(adv)?(gan)?$")


def resolve_regime_from_path(sweep_dir: str) -> Optional[str]:
    """Infer the training regime from a sweep directory path.

    The standard sweep output directory has the format::

        outputs/{experiment}/{training_regime}/{embedding}/d{in}D{bond}/{dataset}_{ddmm}

    where ``training_regime`` is the string produced by the ``${training_regime:}``
    OmegaConf resolver — a concatenation of the active trainer codes:

    * ``cls``  — classification pre-training
    * ``gen``  — generative NLL training
    * ``adv``  — adversarial training
    * ``gan``  — GAN-style training

    Possible values include ``cls``, ``gen``, ``adv``, ``gan``, ``clsadv``,
    ``clsgen``, ``clsgan``, ``clsgenadv``, etc.

    Old-style flat paths (e.g. ``seed_sweep_adv_d30D18fourier_moons_4k_1202``)
    are also supported: the path is tokenised on both ``/`` and ``_`` and each
    token is matched against the same pattern.

    Priority when multiple codes are present: ``adv`` > ``gen`` > ``gan`` > ``cls``.

    Args:
        sweep_dir: Path to the sweep directory (relative or absolute).

    Returns:
        One of ``"pre"``, ``"gen"``, ``"adv"``, ``"gan"``, or ``None`` if
        no known regime code was found in the path.

    Examples:
        >>> resolve_regime_from_path("outputs/seed_sweep/adv/fourier/d30D18/moons_4k_0102")
        'adv'
        >>> resolve_regime_from_path("outputs/seed_sweep/clsadv/fourier/d30D18/moons_4k_0102")
        'adv'
        >>> resolve_regime_from_path("outputs/seed_sweep/cls/fourier/d30D18/moons_4k_0102")
        'pre'
        >>> resolve_regime_from_path("outputs/seed_sweep/clsgen/legendre/d30D18/moons_4k_0102")
        'gen'
        >>> resolve_regime_from_path("outputs/seed_sweep_adv_d30D18fourier_moons_4k_1202")
        'adv'
    """
    path_str = str(sweep_dir).replace("\\", "/")
    # Tokenise on both "/" and "_" to handle new-style (/adv/) and old-style (_adv_) paths.
    tokens = re.split(r"[/_]", path_str)

    for token in tokens:
        m = _REGIME_PATH_PATTERN.match(token.lower())
        if m and m.group(0):  # Non-empty exact match → valid regime string
            # Priority: adv > gen > gan > cls
            if m.group(3):  # adv present
                return "adv"
            if m.group(2):  # gen present
                return "gen"
            if m.group(4):  # gan present
                return "gan"
            if m.group(1):  # cls present (no secondary trainer)
                return "pre"

    return None


# =============================================================================
# EMBEDDING RESOLVER
# =============================================================================

# Known embedding names; used to detect the embedding from a sweep path.
_KNOWN_EMBEDDINGS = {"fourier", "legendre", "hermite"}

_EMBEDDING_RANGE_SIZE: Dict[str, float] = {
    "fourier":  1.0,   # range (0, 1)
    "legendre": 2.0,   # range (-1, 1)
    "hermite":  1.0,   # range (0, 1)
}


def resolve_embedding_from_path(sweep_dir: str) -> Optional[str]:
    """Infer the embedding type from a sweep directory path.

    Returns 'fourier', 'legendre', 'hermite', or None.
    The embedding appears as a whole path token (e.g. .../hermite/... or ..._hermite_...).
    """
    path_str = str(sweep_dir).replace("\\", "/")
    tokens = re.split(r"[/_]", path_str)
    for token in tokens:
        if token.lower() in _KNOWN_EMBEDDINGS:
            return token.lower()
    return None


def embedding_range_size(embedding: Optional[str]) -> float:
    """Return the total size of the input range for an embedding.
    Falls back to 1.0 (Fourier) if unknown."""
    return _EMBEDDING_RANGE_SIZE.get((embedding or "").lower(), 1.0)


# =============================================================================
# PARAMETER ALIASES
# =============================================================================

PARAM_ALIASES: Dict[str, str] = {
    # Learning rate aliases
    "learning-rate": "lr",
    "learning_rate": "lr",
    "learningrate": "lr",
    # Weight decay aliases
    "wd": "weight-decay",
    "weight_decay": "weight-decay",
    "weightdecay": "weight-decay",
    # Batch size aliases
    "batch_size": "batch-size",
    "batchsize": "batch-size",
    "bs": "batch-size",
    # Model param aliases
    "bond_dim": "bond-dim",
    "bonddim": "bond-dim",
    "in_dim": "in-dim",
    "indim": "in-dim",
    # Adversarial param aliases
    "trades_beta": "trades-beta",
    "tradesbeta": "trades-beta",
    "beta": "trades-beta",
    "clean_weight": "clean-weight",
    "cleanweight": "clean-weight",
    "strength": "epsilon",
    "eps": "epsilon",
    # GAN param aliases
    "critic_lr": "critic-lr",
    "criticlr": "critic-lr",
    "critic_wd": "critic-wd",
    "criticwd": "critic-wd",
    "r_real": "r-real",
    "rreal": "r-real",
    "num_spc": "num-spc",
    "numspc": "num-spc",
    "num_bins": "num-bins",
    "numbins": "num-bins",
    # Seed aliases
    "data_seed": "data-seed",
    "dataseed": "data-seed",
    "split_seed": "split-seed",
    "splitseed": "split-seed",
}


# =============================================================================
# REGIME PARAMETER MAPPINGS
# =============================================================================

# Maps regime → shorthand → full config path
REGIME_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "pre": {
        "lr": "trainer.classification.optimizer.kwargs.lr",
        "weight-decay": "trainer.classification.optimizer.kwargs.weight_decay",
        "batch-size": "trainer.classification.batch_size",
        "bond-dim": "born.init_kwargs.bond_dim",
        "in-dim": "born.init_kwargs.in_dim",
        "seed": "tracking.seed",
        "data-seed": "dataset.gen_dow_kwargs.seed",
        "split-seed": "dataset.split_seed",
    },
    "gen": {
        "lr": "trainer.generative.optimizer.kwargs.lr",
        "weight-decay": "trainer.generative.optimizer.kwargs.weight_decay",
        "batch-size": "trainer.generative.batch_size",
        "bond-dim": "born.init_kwargs.bond_dim",
        "in-dim": "born.init_kwargs.in_dim",
        "seed": "tracking.seed",
        "data-seed": "dataset.gen_dow_kwargs.seed",
        "split-seed": "dataset.split_seed",
    },
    "adv": {
        "lr": "trainer.adversarial.optimizer.kwargs.lr",
        "weight-decay": "trainer.adversarial.optimizer.kwargs.weight_decay",
        "batch-size": "trainer.adversarial.batch_size",
        "epsilon": "trainer.adversarial.evasion.strengths",
        "trades-beta": "trainer.adversarial.trades_beta",
        "clean-weight": "trainer.adversarial.clean_weight",
        "bond-dim": "born.init_kwargs.bond_dim",
        "in-dim": "born.init_kwargs.in_dim",
        "seed": "tracking.seed",
        "data-seed": "dataset.gen_dow_kwargs.seed",
        "split-seed": "dataset.split_seed",
    },
    "gan": {
        # Generator optimizer
        "lr": "trainer.ganstyle.optimizer.kwargs.lr",
        "weight-decay": "trainer.ganstyle.optimizer.kwargs.weight_decay",
        # Critic optimizer
        "critic-lr": "trainer.ganstyle.critic.discrimination.optimizer.kwargs.lr",
        "critic-wd": "trainer.ganstyle.critic.discrimination.optimizer.kwargs.weight_decay",
        # GAN-specific
        "r-real": "trainer.ganstyle.r_real",
        "num-spc": "trainer.ganstyle.sampling.num_spc",
        "num-bins": "trainer.ganstyle.sampling.num_bins",
        "tolerance": "trainer.ganstyle.tolerance",
        # Model params
        "bond-dim": "born.init_kwargs.bond_dim",
        "in-dim": "born.init_kwargs.in_dim",
        "seed": "tracking.seed",
        "data-seed": "dataset.gen_dow_kwargs.seed",
    },
}


# =============================================================================
# REGIME METADATA
# =============================================================================

REGIME_DESCRIPTIONS: Dict[str, str] = {
    "pre": "Classification pretraining",
    "gen": "Generative NLL training",
    "adv": "Adversarial training",
    "gan": "GAN-style training",
}

REGIME_METRIC_PREFIX: Dict[str, str] = {
    "pre": "pre",
    "gen": "gen",
    "adv": "adv",
    "gan": "gan",
}

# Default params to analyze per regime (if user doesn't specify)
REGIME_DEFAULT_PARAMS: Dict[str, List[str]] = {
    "pre": ["lr", "weight-decay", "batch-size"],
    "gen": ["lr", "weight-decay", "batch-size"],
    "adv": ["lr", "weight-decay", "epsilon", "trades-beta"],
    "gan": ["lr", "weight-decay", "critic-lr", "r-real"],
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_param(name: str) -> str:
    """
    Normalize parameter name using aliases.

    Args:
        name: Parameter name (any case/format)

    Returns:
        Canonical shorthand name (e.g., "lr", "weight-decay")
    """
    name = name.lower().strip().replace(" ", "-")
    return PARAM_ALIASES.get(name, name)


def resolve_params(
    regime: str,
    shorthands: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Resolve shorthand param names to full config paths.

    Args:
        regime: Training regime ("pre", "gen", "adv", "gan")
        shorthands: List of param shorthand names, or None for regime defaults

    Returns:
        Dict mapping shorthand → full config path

    Raises:
        ValueError: If regime unknown or shorthand not found for regime

    Example:
        >>> resolve_params("pre", ["lr", "wd"])
        {
            "lr": "trainer.classification.optimizer.kwargs.lr",
            "weight-decay": "trainer.classification.optimizer.kwargs.weight_decay"
        }
    """
    regime = regime.lower().strip()

    if regime not in REGIME_PARAM_MAP:
        available = ", ".join(REGIME_PARAM_MAP.keys())
        raise ValueError(f"Unknown regime '{regime}'. Available: {available}")

    regime_params = REGIME_PARAM_MAP[regime]

    # Use defaults if not specified
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
    """
    Convert full config path to DataFrame column name.

    Args:
        full_path: Full config path (e.g., "trainer.classification.optimizer.kwargs.lr")

    Returns:
        Column name (e.g., "config/lr")

    Example:
        >>> config_path_to_column("trainer.classification.optimizer.kwargs.lr")
        "config/lr"
        >>> config_path_to_column("born.init_kwargs.bond_dim")
        "config/bond_dim"
    """
    # Get the last part of the path
    short_key = full_path.split(".")[-1]

    # Handle the "kwargs" case - go back one more level
    if short_key == "kwargs":
        parts = full_path.split(".")
        if len(parts) >= 2:
            short_key = parts[-2]

    return f"config/{short_key}"


def resolve_metrics(
    regime: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, List[str]]:
    """
    Resolve metric column names for a regime.

    Args:
        regime: Training regime
        df: Optional DataFrame to auto-detect robustness strengths from

    Returns:
        Dict with keys:
          - "validation": validation metric column names
          - "robustness": robustness metric column names (auto-detected)
          - "test": test metric column names
    """
    regime = regime.lower().strip()

    if regime not in REGIME_METRIC_PREFIX:
        available = ", ".join(REGIME_METRIC_PREFIX.keys())
        raise ValueError(f"Unknown regime '{regime}'. Available: {available}")

    prefix = REGIME_METRIC_PREFIX[regime]

    # Standard metrics
    validation = [
        f"summary/{prefix}/valid/acc",
        f"summary/{prefix}/valid/loss",
    ]
    test = [
        f"summary/{prefix}/test/acc",
        f"summary/{prefix}/test/loss",
    ]

    # Auto-detect robustness metrics from DataFrame
    robustness = []
    if df is not None:
        strengths = detect_robustness_strengths(df, prefix)
        robustness = [f"summary/{prefix}/valid/rob/{s}" for s in strengths]
    else:
        # Default strengths if no DataFrame provided
        robustness = [
            f"summary/{prefix}/valid/rob/0.1",
            f"summary/{prefix}/valid/rob/0.3",
        ]

    return {
        "validation": validation,
        "robustness": robustness,
        "test": test,
    }


def resolve_primary_metric(
    shorthand: str,
    validation_metrics: List[str],
    robustness_metrics: List[str],
    test_metrics: List[str],
) -> Tuple[Optional[str], bool]:
    """
    Resolve a PRIMARY_METRIC shorthand to a full column name.

    Shorthand mapping:
      - "rob" / "robustness"   -> first robustness metric (maximize)
      - "acc" / "accuracy"     -> first validation metric containing "acc" (maximize)
      - "loss"                 -> first validation metric containing "loss" (minimize)
      - "test-acc" / "test_acc"-> first test metric containing "acc" (maximize)
      - "test-loss"/ "test_loss"-> first test metric containing "loss" (minimize)
      - substring match (e.g. "rob/0.15") -> matching metric from all pools
      - full column name       -> pass through

    Args:
        shorthand: User-provided metric shorthand or column name
        validation_metrics: List of validation metric column names
        robustness_metrics: List of robustness metric column names
        test_metrics: List of test metric column names

    Returns:
        (resolved_column_name, minimize_flag) or (None, False) on failure
    """
    all_metrics = validation_metrics + robustness_metrics + test_metrics
    key = shorthand.strip().lower()

    # Direct shorthands
    if key in ("rob", "robustness"):
        if robustness_metrics:
            return robustness_metrics[0], False  # maximize
        return None, False

    if key in ("acc", "accuracy"):
        match = next((m for m in validation_metrics if "acc" in m), None)
        if match:
            return match, False  # maximize
        return None, False

    if key == "loss":
        match = next((m for m in validation_metrics if "loss" in m), None)
        if match:
            return match, True  # minimize
        return None, False

    if key in ("test-acc", "test_acc"):
        match = next((m for m in test_metrics if "acc" in m), None)
        if match:
            return match, False  # maximize
        return None, False

    if key in ("test-loss", "test_loss"):
        match = next((m for m in test_metrics if "loss" in m), None)
        if match:
            return match, True  # minimize
        return None, False

    # Exact match (full column name already)
    if shorthand in all_metrics:
        minimize = "loss" in shorthand
        return shorthand, minimize

    # Substring match across all pools
    matches = [m for m in all_metrics if shorthand in m]
    if matches:
        resolved = matches[0]
        minimize = "loss" in resolved
        return resolved, minimize

    return None, False


def detect_robustness_strengths(df: pd.DataFrame, prefix: str) -> List[float]:
    """
    Auto-detect robustness metric strengths from DataFrame columns.

    Args:
        df: DataFrame with metric columns
        prefix: Metric prefix (e.g., "pre", "adv")

    Returns:
        List of strength values found, sorted ascending
    """
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
    """
    Filter to only parameters that varied in the sweep.

    Args:
        df: DataFrame with config columns
        param_columns: List of column names to check
        min_unique: Minimum unique values to be considered "varied"

    Returns:
        Tuple of (varied_columns, excluded_columns)
    """
    varied = []
    excluded = []

    for col in param_columns:
        if col not in df.columns:
            excluded.append(col)
            continue

        # Convert lists to tuples so pandas can hash them for nunique
        col_hashable = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        n_unique = col_hashable.nunique(dropna=True)
        if n_unique >= min_unique:
            varied.append(col)
        else:
            excluded.append(col)

    return varied, excluded


def detect_pretrained_info(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Check if a pretrained model was loaded and extract info.

    Looks for:
      - config/model_path: path to pretrained model
      - Pretrained model performance metrics

    Args:
        df: DataFrame with config/metric columns

    Returns:
        Dict with pretrained info or None if no pretrained model detected
    """
    # Check for model path in config
    model_path_col = "config/model_path"

    if model_path_col not in df.columns:
        return None

    # Get unique model paths (excluding None/NaN)
    paths = df[model_path_col].dropna().unique()

    if len(paths) == 0:
        return None

    # Return info about pretrained model
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
    """
    Format resolved configuration for display.

    Args:
        regime: Training regime name
        params: Dict mapping shorthand → full config path
        varied_params: List of shorthand names that varied
        excluded_params: List of shorthand names excluded (single value)
        metrics: Dict with validation/robustness/test metric lists
        pretrained_info: Optional dict with pretrained model info

    Returns:
        Formatted string for display
    """
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("Resolved Configuration")
    lines.append(sep)
    lines.append("")

    # Regime info
    desc = REGIME_DESCRIPTIONS.get(regime, regime)
    lines.append(f"Regime: {regime} ({desc})")
    lines.append("")

    # Parameters
    n_varied = len(varied_params)
    n_excluded = len(excluded_params)
    lines.append(f"Parameters ({n_varied} varied, {n_excluded} excluded):")

    # Show varied params first
    for shorthand in varied_params:
        full_path = params.get(shorthand, "?")
        lines.append(f"  + {shorthand:<16} -> {full_path}")

    # Show excluded params
    for shorthand in excluded_params:
        full_path = params.get(shorthand, "?")
        lines.append(f"  - {shorthand:<16} (excluded: single value)")

    lines.append("")

    # Metrics
    lines.append("Metrics:")

    def format_metric_list(metric_list: List[str], max_show: int = 4) -> str:
        if not metric_list:
            return "(none)"
        # Shorten metric names for display
        short_names = [m.split("/")[-1] for m in metric_list[:max_show]]
        result = ", ".join(short_names)
        if len(metric_list) > max_show:
            result += f", ... (+{len(metric_list) - max_show} more)"
        return result

    val_str = format_metric_list(metrics.get("validation", []))
    rob_str = format_metric_list(metrics.get("robustness", []))
    test_str = format_metric_list(metrics.get("test", []))

    lines.append(f"  Validation: {val_str}")
    if metrics.get("robustness"):
        # Show full robustness metric names since strengths matter
        rob_names = [m.split("/")[-1] for m in metrics["robustness"]]
        lines.append(f"  Robustness: {', '.join(rob_names)} (auto-detected)")
    else:
        lines.append("  Robustness: (none detected)")
    lines.append(f"  Test:       {test_str}")

    lines.append("")

    # Pretrained info
    if pretrained_info:
        n_runs = pretrained_info.get("n_runs_with_pretrained", 0)
        lines.append(f"Pretrained Model: detected in {n_runs} runs")
    else:
        lines.append("Pretrained Model: None detected")

    lines.append(sep)

    return "\n".join(lines)


def get_available_params(regime: str) -> List[str]:
    """
    Get list of available parameter shorthand names for a regime.

    Args:
        regime: Training regime

    Returns:
        List of available shorthand names
    """
    regime = regime.lower().strip()
    if regime not in REGIME_PARAM_MAP:
        return []
    return list(REGIME_PARAM_MAP[regime].keys())


def get_available_regimes() -> List[str]:
    """Get list of available training regimes."""
    return list(REGIME_PARAM_MAP.keys())
