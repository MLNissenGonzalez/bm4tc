"""
Utility functions for MIA (Membership Inference Attack) analysis.

Provides functions for loading run configurations from local Hydra outputs
or wandb, and locating model checkpoints.
"""

from pathlib import Path
from typing import Union, Optional
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


def load_run_config(run_dir: Union[str, Path]) -> DictConfig:
    """Load full config from .hydra/config.yaml in run output folder.

    This loads the complete Hydra configuration used for a training run,
    which can be used to reconstruct the DataHandler and ConditionalBornMachine.

    Args:
        run_dir: Path to the run output directory (e.g., outputs/experiment_name_date).

    Returns:
        OmegaConf DictConfig with the full configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.

    Example:
        >>> cfg = load_run_config("outputs/classification_2024_01_15")
        >>> print(cfg.dataset.name)
        'spirals_4k'
    """
    run_dir = Path(run_dir)
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Expected Hydra output directory structure: {run_dir}/.hydra/config.yaml"
        )

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
