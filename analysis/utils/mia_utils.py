"""
Utility functions for MIA (Membership Inference Attack) analysis.

Provides functions for loading run configurations from local Hydra outputs
or wandb, and locating model checkpoints.
"""

from pathlib import Path
from typing import Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


def _config_default_options(defaults: Any) -> dict[str, str]:
    """Extract Hydra default-group selections from a defaults list."""
    options: dict[str, str] = {}
    for entry in defaults or []:
        if not isinstance(entry, dict):
            continue
        for raw_group, value in entry.items():
            group = str(raw_group).removeprefix("override ").lstrip("/")
            if value is None:
                options.pop(group, None)
            else:
                options[group] = str(value)
    return options


def _merge_config_group(cfg, group: str, value):
    """Merge a config-group value at its slash-separated Hydra group path."""
    nested = value
    for key in reversed(group.split("/")):
        nested = {key: nested}
    return OmegaConf.merge(cfg, nested)


def _fallback_experiment_path(run_dir: Path) -> Path:
    """Infer a final experiment YAML from an MPS multirun output path."""
    parts = list(run_dir.parts)
    if "outputs" not in parts:
        raise FileNotFoundError(f"Cannot infer an experiment YAML from {run_dir}")
    layout = parts[parts.index("outputs") + 1 :]
    if len(layout) < 5:
        raise FileNotFoundError(f"Output path is too short to infer its config: {run_dir}")
    dataset, regime, embedding, architecture = layout[:4]
    remainder = layout[4:]
    if remainder[0] == "seed_sweep" and len(remainder) >= 2:
        stage, sweep_dir = "seed_sweep", remainder[1]
    elif remainder[0].startswith("seed_sweep_"):
        stage, sweep_dir = "seed_sweep", remainder[0].removeprefix("seed_sweep_")
    else:
        raise FileNotFoundError(
            f"Cannot infer a final seed-sweep config from output path {run_dir}"
        )

    # MPS's historical AT runs are named ``seed_sweep_DDMM`` without an
    # experiment identifier; the corresponding final config is ``at.yaml``.
    if regime == "at" and sweep_dir.isdigit():
        experiment = "at"
    else:
        experiment = sweep_dir.rsplit("_", 1)[0]
    config_root = Path(__file__).resolve().parents[2] / "configs"
    return (
        config_root
        / "experiments"
        / dataset
        / regime
        / embedding
        / architecture
        / stage
        / f"{experiment}.yaml"
    )


def _load_final_experiment_config(run_dir: Path) -> DictConfig:
    """Rebuild a Hydra run config from its committed final experiment YAML."""
    experiment_path = _fallback_experiment_path(run_dir)
    if not experiment_path.exists():
        raise FileNotFoundError(
            f"No Hydra config below {run_dir} and no final experiment YAML at "
            f"{experiment_path}"
        )
    config_root = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(config_root / "config.yaml")
    experiment = OmegaConf.load(experiment_path)
    base_defaults = OmegaConf.to_container(base.get("defaults"), resolve=False)
    experiment_defaults = OmegaConf.to_container(
        experiment.get("defaults"), resolve=False
    )
    options = _config_default_options(base_defaults)
    options.update(_config_default_options(experiment_defaults))

    cfg = OmegaConf.create(base)
    for group, option in options.items():
        group_path = config_root / group / f"{option}.yaml"
        if group_path.exists():
            cfg = _merge_config_group(cfg, group, OmegaConf.load(group_path))
    cfg = OmegaConf.merge(cfg, experiment)

    params = OmegaConf.select(cfg, "hydra.sweeper.params") or {}
    for key, value in params.items():
        if key == "tracking.seed" and isinstance(value, str) and value.startswith("range("):
            value = int(run_dir.name) + 1 if run_dir.name.isdigit() else 1
        OmegaConf.update(cfg, key, value, merge=True)
    logger.info("Reconstructed config for %s from %s", run_dir, experiment_path)
    return cfg


def load_run_config(run_dir: Union[str, Path]) -> DictConfig:
    """Load a run config from Hydra metadata or its final experiment YAML.

    This loads the complete Hydra configuration used for a training run,
    which can be used to reconstruct the DataHandler and ConditionalBornMachine.

    Args:
        run_dir: Path to the run output directory (e.g., outputs/experiment_name_date).

    Returns:
        OmegaConf DictConfig with the full configuration.

    If a copied output omits ``.hydra/config.yaml``, reconstruct the final
    seed-sweep config from the path under ``outputs/`` and its committed YAML
    below ``configs/experiments``. This is sufficient to analyse checkpoint-only
    runs and preserves the seed override from the sweep directory.

    Example:
        >>> cfg = load_run_config("outputs/classification_2024_01_15")
        >>> print(cfg.dataset.name)
        'spirals_4k'
    """
    run_dir = Path(run_dir)
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        return _load_final_experiment_config(run_dir)

    logger.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    return cfg


def find_model_checkpoint(
    run_dir: Union[str, Path],
    checkpoint_name: Optional[str] = None
) -> Path:
    """Find model checkpoint file in run output folder.

    Searches for checkpoint files in the models/ subdirectory of the run
    output. Looks for .pt files first, then falls back to any file in the
    directory (trainers save checkpoints without a .pt extension).

    Args:
        run_dir: Path to the run output directory.
        checkpoint_name: Specific checkpoint filename to look for.
                         If None, auto-detects the checkpoint file.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint is found.

    Example:
        >>> checkpoint = find_model_checkpoint("outputs/classification_2024_01_15")
        >>> cbm = ConditionalBornMachine.load(str(checkpoint), accumulate=True)
    """
    run_dir = Path(run_dir)
    models_dir = run_dir / "models"

    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found at {models_dir}. "
            f"Expected checkpoint at: {run_dir}/models/"
        )

    if checkpoint_name:
        checkpoint_path = models_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found in {models_dir}"
        )

    # Find .pt files first, then fall back to all files (trainers save
    # checkpoints without extensions)
    checkpoints = list(models_dir.glob("*.pt"))

    if not checkpoints:
        checkpoints = [
            p for p in models_dir.iterdir() if p.is_file()
        ]

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files found in {models_dir}"
        )

    # If multiple, prefer ones with common names
    preferred_names = ["model.pt", "born_machine.pt", "classifier.pt", "final.pt"]
    for name in preferred_names:
        for cp in checkpoints:
            if cp.name == name:
                logger.info(f"Found checkpoint: {cp}")
                return cp

    # Return first found
    checkpoint = checkpoints[0]
    if len(checkpoints) > 1:
        logger.warning(
            f"Multiple checkpoints found: {[cp.name for cp in checkpoints]}. "
            f"Using: {checkpoint.name}"
        )

    logger.info(f"Found checkpoint: {checkpoint}")
    return checkpoint
