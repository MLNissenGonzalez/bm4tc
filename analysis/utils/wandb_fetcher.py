"""
Utilities for fetching experiment data from Weights & Biases or local outputs.

This module provides convenient wrappers around the wandb API for
retrieving run data, configs, and metrics for post-experiment analysis.
It also supports loading data from local Hydra output directories.

Example usage (wandb):
    from analysis.utils import WandbFetcher

    fetcher = WandbFetcher(entity="my-entity", project="my-project")
    runs_df = fetcher.fetch_runs(filters={"group": "lrwdbs_hpo_spirals_4k"})

Example usage (local):
    from analysis.utils import load_local_hpo_runs

    df = load_local_hpo_runs("outputs/lrwdbs_hpo_spirals_4k_22Jan26")
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
import os
import glob as glob_module

try:
    import wandb
    from wandb.apis.public import Run
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    Run = Any

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


def _get_default_config_keys(regime: Optional[str] = None) -> List[str]:
    """
    Get default config keys for HPO analysis.

    Args:
        regime: Optional training regime ("pre", "gen", "adv", "gan").
                If provided, uses regime-specific params from resolve module.
                If None, uses legacy classification params.

    Returns:
        List of full config paths to extract
    """
    if regime is not None:
        try:
            from analysis.utils.resolve import REGIME_PARAM_MAP
            if regime in REGIME_PARAM_MAP:
                return list(REGIME_PARAM_MAP[regime].values())
        except ImportError:
            pass

    # Legacy default keys (classification-focused)
    return [
        "trainer.classification.optimizer.kwargs.lr",
        "trainer.classification.optimizer.kwargs.weight_decay",
        "trainer.classification.batch_size",
        "born.init_kwargs.bond_dim",
        "born.init_kwargs.in_dim",
        "dataset.name",
        "experiment",
    ]


@dataclass
class WandbFetcher:
    """
    Fetches run data from a wandb project.

    Attributes:
        entity: wandb entity (username or team name)
        project: wandb project name
        api: wandb API instance (created automatically)
    """
    entity: str
    project: str
    api: Any = field(default=None, repr=False)

    def __post_init__(self):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )
        self.api = wandb.Api()

    @property
    def project_path(self) -> str:
        """Returns the full project path: entity/project"""
        return f"{self.entity}/{self.project}"

    def fetch_runs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order: str = "-created_at",
        per_page: int = 100,
    ) -> List[Run]:
        """
        Fetch runs from the project.

        Args:
            filters: MongoDB-style filter dict. Common filters:
                - {"group": "experiment_name"} - filter by group
                - {"state": "finished"} - only finished runs
                - {"config.dataset.name": "moons_4k"} - filter by config
                - {"$and": [{...}, {...}]} - combine filters
            order: Sort order. Use "-" prefix for descending.
                Common: "-created_at", "created_at", "-summary_metrics.acc"
            per_page: Number of runs to fetch per API call

        Returns:
            List of wandb Run objects

        Example:
            # Fetch all runs from a specific HPO experiment
            runs = fetcher.fetch_runs(
                filters={"group": {"$regex": "lrwdbs_hpo.*"}},
                order="-created_at"
            )
        """
        runs = self.api.runs(
            path=self.project_path,
            filters=filters,
            order=order,
            per_page=per_page,
        )
        return list(runs)

    def runs_to_dataframe(
        self,
        runs: List[Run],
        config_keys: Optional[List[str]] = None,
        summary_keys: Optional[List[str]] = None,
        include_history: bool = False,
        history_keys: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert wandb runs to a pandas DataFrame.

        Args:
            runs: List of wandb Run objects
            config_keys: Config keys to extract (dot notation supported).
                If None, extracts common HPO params automatically.
            summary_keys: Summary metric keys to extract.
                If None, extracts all summary metrics.
            include_history: Whether to include full training history
            history_keys: If include_history, which metrics to include

        Returns:
            DataFrame with one row per run, columns for config and metrics
        """
        records = []

        for run in runs:
            record = {
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                "state": run.state,
                "created_at": run.created_at,
            }

            # Extract config
            config = run.config
            if config_keys is None:
                config_keys_to_use = _get_default_config_keys()
            else:
                config_keys_to_use = config_keys

            for key in config_keys_to_use:
                value = self._get_nested(config, key)
                # Shorten key for column name
                short_key = key.split(".")[-1] if "." in key else key
                # Handle duplicate short keys
                if short_key == "kwargs":
                    short_key = key.split(".")[-2] + "_" + key.split(".")[-1]
                record[f"config/{short_key}"] = value

            # Extract summary metrics
            summary = run.summary._json_dict
            if summary_keys is None:
                # Extract all numeric summaries
                for key, value in summary.items():
                    if isinstance(value, (int, float)) and not key.startswith("_"):
                        record[f"summary/{key}"] = value
            else:
                for key in summary_keys:
                    record[f"summary/{key}"] = summary.get(key)

            records.append(record)

        df = pd.DataFrame(records)

        # Convert created_at to datetime
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])

        return df

    def fetch_run_history(
        self,
        run: Union[Run, str],
        keys: Optional[List[str]] = None,
        samples: int = 10000,
    ) -> pd.DataFrame:
        """
        Fetch the full training history of a run.

        Args:
            run: wandb Run object or run_id string
            keys: Specific metric keys to fetch. None for all.
            samples: Max number of history points to fetch

        Returns:
            DataFrame with training history
        """
        if isinstance(run, str):
            run = self.api.run(f"{self.project_path}/{run}")

        history = run.history(samples=samples, keys=keys)
        return history

    def _get_nested(self, d: Dict, key: str, default: Any = None) -> Any:
        """Get nested dict value using dot notation."""
        keys = key.split(".")
        value = d
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


def fetch_hpo_runs(
    entity: str,
    project: str,
    experiment_pattern: str,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to fetch HPO experiment runs.

    Args:
        entity: wandb entity
        project: wandb project name
        experiment_pattern: Experiment name or regex pattern
        dataset_name: Optional dataset name to filter by

    Returns:
        DataFrame with HPO runs and their configs/metrics

    Example:
        df = fetch_hpo_runs(
            entity="my-entity",
            project="gan_train",
            experiment_pattern="lrwdbs_hpo",
            dataset_name="spirals_4k"
        )
    """
    fetcher = WandbFetcher(entity=entity, project=project)

    # Build filter
    filters = {"state": "finished"}

    # Add group filter (experiment name is typically in group)
    if "*" in experiment_pattern or "." in experiment_pattern:
        filters["group"] = {"$regex": experiment_pattern}
    else:
        filters["group"] = {"$regex": f".*{experiment_pattern}.*"}

    # Add dataset filter if specified
    if dataset_name:
        filters["config.dataset.name"] = dataset_name

    runs = fetcher.fetch_runs(filters=filters)

    if not runs:
        print(f"No runs found matching pattern: {experiment_pattern}")
        return pd.DataFrame()

    print(f"Found {len(runs)} runs")

    return fetcher.runs_to_dataframe(runs)


def get_hpo_metrics_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for HPO results.

    Args:
        df: DataFrame from fetch_hpo_runs or runs_to_dataframe

    Returns:
        Dict with metric names as keys and summary stats as values
    """
    summary_cols = [c for c in df.columns if c.startswith("summary/")]

    results = {}
    for col in summary_cols:
        metric_name = col.replace("summary/", "")
        valid_values = df[col].dropna()

        if len(valid_values) > 0:
            results[metric_name] = {
                "mean": valid_values.mean(),
                "std": valid_values.std(),
                "min": valid_values.min(),
                "max": valid_values.max(),
                "median": valid_values.median(),
                "count": len(valid_values),
            }

    return results


# =============================================================================
# LOCAL FILE LOADING (from outputs/ directory)
# =============================================================================

def _get_nested_value(d: Dict, key: str, default: Any = None) -> Any:
    """Get nested dict value using dot notation."""
    keys = key.split(".")
    value = d
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, default)
        else:
            return default
    return value


def _load_hydra_config(run_dir: Path) -> Optional[Dict]:
    """
    Load Hydra config from a run directory.

    Args:
        run_dir: Path to run directory (should contain .hydra/config.yaml)

    Returns:
        Config dict or None if not found
    """
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        return None

    try:
        if OMEGACONF_AVAILABLE:
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)
        else:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return None


def _load_wandb_summary(run_dir: Path) -> Optional[Dict]:
    """
    Load wandb summary metrics from a run directory.

    Wandb stores summary in wandb/<run_dir>/files/wandb-summary.json

    Args:
        run_dir: Path to run directory

    Returns:
        Summary dict or None if not found
    """
    # Look for wandb summary file
    wandb_dirs = list(run_dir.glob("wandb/run-*"))
    if not wandb_dirs:
        # Try looking in parent directory's wandb folder
        wandb_dirs = list(run_dir.glob("*/wandb/run-*"))

    for wandb_dir in wandb_dirs:
        summary_path = wandb_dir / "files" / "wandb-summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load summary from {summary_path}: {e}")

    return None


def load_local_run(
    run_dir: Path,
    config_keys: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Load a single run from a local Hydra output directory.

    Args:
        run_dir: Path to run directory (contains .hydra/config.yaml)
        config_keys: Config keys to extract (dot notation)

    Returns:
        Dict with run info, config, and metrics, or None if loading fails
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        return None

    # Load config
    config = _load_hydra_config(run_dir)
    if config is None:
        return None

    # Load summary metrics
    summary = _load_wandb_summary(run_dir)

    # Build record
    record = {
        "run_id": run_dir.name,
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "state": "finished",  # Assume finished if directory exists
    }

    # Extract config values
    if config_keys is None:
        config_keys = _get_default_config_keys()

    for key in config_keys:
        value = _get_nested_value(config, key)
        short_key = key.split(".")[-1] if "." in key else key
        if short_key == "kwargs":
            short_key = key.split(".")[-2] + "_" + key.split(".")[-1]
        record[f"config/{short_key}"] = value

    # Extract summary metrics
    if summary:
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                record[f"summary/{key}"] = value

    return record


def load_local_hpo_runs(
    sweep_dir: Union[str, Path],
    config_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load all runs from a local Hydra multirun/sweep output directory.

    The expected directory structure is:
        sweep_dir/
        ├── 0/                    # First trial
        │   ├── .hydra/
        │   │   └── config.yaml
        │   └── wandb/           # (optional)
        ├── 1/                    # Second trial
        │   └── ...
        └── multirun.yaml         # (optional)

    Args:
        sweep_dir: Path to sweep output directory
        config_keys: Config keys to extract (dot notation)

    Returns:
        DataFrame with one row per run

    Example:
        df = load_local_hpo_runs("outputs/lrwdbs_hpo_spirals_4k_22Jan26")
    """
    sweep_dir = Path(sweep_dir)

    if not sweep_dir.exists():
        print(f"Directory not found: {sweep_dir}")
        return pd.DataFrame()

    # Find all run directories (numbered subdirectories)
    run_dirs = []
    for item in sweep_dir.iterdir():
        if item.is_dir():
            # Check if it's a numbered directory or has .hydra folder
            if item.name.isdigit() or (item / ".hydra").exists():
                run_dirs.append(item)

    if not run_dirs:
        print(f"No run directories found in: {sweep_dir}")
        return pd.DataFrame()

    # Sort by number if numbered
    run_dirs.sort(key=lambda x: int(x.name) if x.name.isdigit() else x.name)

    print(f"Found {len(run_dirs)} run directories in {sweep_dir}")

    # Load each run
    records = []
    for run_dir in run_dirs:
        record = load_local_run(run_dir, config_keys)
        if record:
            record["group"] = sweep_dir.name
            records.append(record)

    if not records:
        print("No valid runs found (missing config or metrics)")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} runs")

    return df


def find_local_sweep_dirs(
    outputs_dir: Union[str, Path] = "outputs",
    pattern: Optional[str] = None,
) -> List[Path]:
    """
    Find sweep directories in the outputs folder.

    Args:
        outputs_dir: Path to outputs directory
        pattern: Optional pattern to filter directories (glob-style)

    Returns:
        List of sweep directory paths

    Example:
        # Find all HPO sweep directories
        sweeps = find_local_sweep_dirs(pattern="*hpo*")
    """
    outputs_dir = Path(outputs_dir)

    if not outputs_dir.exists():
        return []

    if pattern:
        dirs = list(outputs_dir.glob(pattern))
    else:
        dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]

    # Filter to directories that look like sweeps (have numbered subdirs)
    sweep_dirs = []
    for d in dirs:
        subdirs = list(d.iterdir()) if d.is_dir() else []
        has_numbered = any(s.name.isdigit() for s in subdirs if s.is_dir())
        if has_numbered:
            sweep_dirs.append(d)

    return sorted(sweep_dirs)


# =============================================================================
# BEST RUN FETCHING (for downstream training)
# =============================================================================

def fetch_best_classification_run(
    entity: str,
    project: str,
    group: str,
    dataset_name: Optional[str] = None,
    metric: str = "pre/valid/acc",
    minimize: bool = False,
) -> Optional[Any]:
    """
    Fetch the best classification run from W&B for a given group.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        group: W&B group name (experiment name) to filter by. Required.
        dataset_name: Optional dataset to filter by (e.g., "spirals_4k").
        metric: Summary metric to sort by (default: "pre/valid/acc").
        minimize: If True, sort ascending (for loss metrics). Default False.

    Returns:
        The best wandb Run object, or None if no runs found.

    Example:
        >>> run = fetch_best_classification_run(
        ...     entity="my-entity",
        ...     project="gan_train",
        ...     group="sanity_check_moons_4k_27Jan25",
        ...     dataset_name="moons_4k"
        ... )
        >>> if run:
        ...     print(f"Best run: {run.name}, acc: {run.summary.get('pre/valid/acc')}")
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is required. Install with: pip install wandb")

    fetcher = WandbFetcher(entity=entity, project=project)
    order = f"summary_metrics.{metric}" if minimize else f"-summary_metrics.{metric}"

    # Build filter
    filters = {
        "group": {"$regex": f".*{group}.*"},
        "state": "finished"
    }

    if dataset_name:
        filters["config.dataset.name"] = dataset_name

    try:
        runs = fetcher.fetch_runs(filters=filters, order=order)
        if runs:
            return runs[0]
    except Exception:
        pass

    return None


def extract_classification_config(run: Any) -> Dict[str, Any]:
    """
    Extract classification training config from a W&B run.

    Args:
        run: A wandb Run object.

    Returns:
        Dict containing classification trainer config suitable for use
        in adversarial training HPO experiments.

    Example:
        >>> run = fetch_best_classification_run(...)
        >>> cls_config = extract_classification_config(run)
        >>> print(cls_config["optimizer"]["kwargs"]["lr"])
    """
    config = run.config
    cls_config = _get_nested_value(config, "trainer.classification", {})

    return {
        "max_epoch": cls_config.get("max_epoch", 500),
        "batch_size": cls_config.get("batch_size", 64),
        "optimizer": cls_config.get("optimizer", {
            "name": "adam",
            "kwargs": {"lr": 5e-3, "weight_decay": 0.0}
        }),
        "criterion": cls_config.get("criterion", {
            "name": "negative log-likelihood",
            "kwargs": {"eps": 1e-8}
        }),
        "patience": cls_config.get("patience", 250),
        "stop_crit": cls_config.get("stop_crit", "clsloss"),
        "watch_freq": cls_config.get("watch_freq", 1000),
        "metrics": cls_config.get("metrics", {"clsloss": 1, "acc": 1}),
        "save": False,  # Don't save intermediate models in HPO
        "auto_stack": cls_config.get("auto_stack", True),
        "auto_unbind": cls_config.get("auto_unbind", False),
    }


def extract_born_config(run: Any) -> Dict[str, Any]:
    """
    Extract Born machine config from a W&B run.

    Args:
        run: A wandb Run object.

    Returns:
        Dict containing Born machine configuration.
    """
    config = run.config
    born_config = _get_nested_value(config, "born", {})

    return {
        "init_kwargs": {
            "in_dim": _get_nested_value(born_config, "init_kwargs.in_dim", 30),
            "bond_dim": _get_nested_value(born_config, "init_kwargs.bond_dim", 18),
            "boundary": _get_nested_value(born_config, "init_kwargs.boundary", "obc"),
            "init_method": _get_nested_value(born_config, "init_kwargs.init_method", "randn_eye"),
            "std": _get_nested_value(born_config, "init_kwargs.std", 1e-9),
        },
        "embedding": born_config.get("embedding", "fourier"),
    }


def extract_dataset_config(run: Any) -> Dict[str, Any]:
    """
    Extract dataset config from a W&B run.

    Args:
        run: A wandb Run object.

    Returns:
        Dict containing dataset configuration (name, split_seed, split,
        and gen_dow_kwargs fields).
    """
    config = run.config
    dataset_config = _get_nested_value(config, "dataset", {})

    return {
        "name": dataset_config.get("name"),
        "split_seed": dataset_config.get("split_seed"),
        "split": dataset_config.get("split"),
        "gen_dow_kwargs": {
            "seed": _get_nested_value(dataset_config, "gen_dow_kwargs.seed"),
            "name": _get_nested_value(dataset_config, "gen_dow_kwargs.name"),
            "size": _get_nested_value(dataset_config, "gen_dow_kwargs.size"),
            "noise": _get_nested_value(dataset_config, "gen_dow_kwargs.noise"),
        },
    }


def print_classification_config_yaml(
    entity: str,
    project: str,
    group: str,
    dataset_name: Optional[str] = None,
) -> None:
    """
    Fetch and print best classification config in copy-pasteable YAML format.

    Args:
        entity: W&B entity.
        project: W&B project name.
        group: W&B group name (experiment name) to filter by.
        dataset_name: Optional dataset to filter by.

    Example:
        >>> print_classification_config_yaml(
        ...     entity="my-entity",
        ...     project="gan_train",
        ...     group="sanity_check_moons_4k_27Jan25"
        ... )
        # Output: YAML that can be pasted into experiment configs
    """
    config = get_best_classification_config(entity, project, group, dataset_name)

    print(f"# Best classification config from W&B")
    print(f"# Group: {config['group']}")
    print(f"# Run ID: {config['run_id']}")
    print(f"# Run Name: {config['run_name']}")
    if config['metrics']:
        print(f"# Valid Acc: {config['metrics'].get('valid_acc')}")
        print(f"# Test Acc: {config['metrics'].get('test_acc')}")
    if config.get('tracking_seed') is not None:
        print(f"# Tracking seed: {config['tracking_seed']}")
    ds_cfg = config.get('dataset_config')
    if ds_cfg is not None:
        print(f"# Dataset: {ds_cfg.get('name')}")
        print(f"# Dataset split_seed: {ds_cfg.get('split_seed')}")
        gen_seed = _get_nested_value(ds_cfg, "gen_dow_kwargs.seed")
        print(f"# Dataset gen seed: {gen_seed}")
    print()

    # Tracking section
    if config.get('tracking_seed') is not None:
        print("tracking:")
        print(f"  seed: {config['tracking_seed']}")
        print()

    # Dataset section
    if ds_cfg is not None and ds_cfg.get('split_seed') is not None:
        print("dataset:")
        print(f"  split_seed: {ds_cfg['split_seed']}")
        gen_kwargs = ds_cfg.get('gen_dow_kwargs', {})
        if gen_kwargs and gen_kwargs.get('seed') is not None:
            print("  gen_dow_kwargs:")
            print(f"    seed: {gen_kwargs['seed']}")
        print()

    print("trainer:")
    print("  classification:")

    cls = config['cls_config']
    print(f"    max_epoch: {cls['max_epoch']}")
    print(f"    batch_size: {cls['batch_size']}")
    print(f"    patience: {cls['patience']}")
    print(f"    stop_crit: \"{cls['stop_crit']}\"")
    print(f"    watch_freq: {cls['watch_freq']}")
    print(f"    save: {str(cls['save']).lower()}")
    print("    optimizer:")
    print(f"      name: \"{cls['optimizer']['name']}\"")
    print("      kwargs:")
    print(f"        lr: {cls['optimizer']['kwargs']['lr']}")
    print(f"        weight_decay: {cls['optimizer']['kwargs']['weight_decay']}")
    print("    criterion:")
    print(f"      name: \"{cls['criterion']['name']}\"")
    if cls['criterion'].get('kwargs'):
        print("      kwargs:")
        for k, v in cls['criterion']['kwargs'].items():
            print(f"        {k}: {v}")
    print(f"    metrics: {cls['metrics']}")

    # Born config section
    born = config.get('born_config')
    if born is not None:
        print()
        print("born:")
        init_kwargs = born.get('init_kwargs', {})
        if init_kwargs:
            print("  init_kwargs:")
            for k, v in init_kwargs.items():
                print(f"    {k}: {v}")
        embedding = born.get('embedding')
        if embedding is not None:
            print(f"  embedding: \"{embedding}\"")


def get_best_classification_config(
    entity: str,
    project: str,
    group: str,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch the best classification config from a specific W&B group.

    Args:
        entity: W&B entity.
        project: W&B project name.
        group: W&B group name (experiment name) to filter by. Required.
        dataset_name: Optional dataset to filter by.

    Returns:
        Dict containing: {"run_id", "run_name", "group", "cls_config",
        "born_config", "dataset_config", "tracking_seed", "metrics"}.

    Example:
        >>> config = get_best_classification_config(
        ...     entity="my-entity",
        ...     project="gan_train",
        ...     group="sanity_check_moons_4k_27Jan25"
        ... )
        >>> print(f"lr={config['cls_config']['optimizer']['kwargs']['lr']}")
    """
    run = fetch_best_classification_run(
        entity=entity,
        project=project,
        group=group,
        dataset_name=dataset_name,
    )

    if run:
        summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else {}
        tracking_seed = _get_nested_value(run.config, "tracking.seed")
        return {
            "run_id": run.id,
            "run_name": run.name,
            "group": run.group,
            "cls_config": extract_classification_config(run),
            "born_config": extract_born_config(run),
            "dataset_config": extract_dataset_config(run),
            "tracking_seed": tracking_seed,
            "metrics": {
                "valid_acc": summary.get("pre/valid/acc"),
                "valid_loss": summary.get("pre/valid/loss"),
                "test_acc": summary.get("pre/test/acc"),
            }
        }
    else:
        # Default fallback config
        return {
            "run_id": None,
            "run_name": "default_fallback",
            "group": group,
            "cls_config": {
                "max_epoch": 500,
                "batch_size": 64,
                "optimizer": {"name": "adam", "kwargs": {"lr": 5e-3, "weight_decay": 0.0}},
                "criterion": {"name": "negative log-likelihood", "kwargs": {"eps": 1e-8}},
                "patience": 250,
                "stop_crit": "clsloss",
                "watch_freq": 1000,
                "metrics": {"clsloss": 1, "acc": 1},
                "save": False,
                "auto_stack": True,
                "auto_unbind": False,
            },
            "born_config": {
                "init_kwargs": {"in_dim": 30, "bond_dim": 18, "boundary": "obc"},
                "embedding": "fourier",
            },
            "dataset_config": None,
            "tracking_seed": None,
            "metrics": None,
        }
