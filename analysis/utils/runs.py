"""Load a finished training run: its config and its model checkpoint."""

import json
from pathlib import Path
from typing import Optional, Union
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


def load_run_config(run_dir: Union[str, Path]) -> DictConfig:
    """The resolved config a finished training run was trained with (its ``run.json``)."""
    manifest = Path(run_dir) / "run.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No run.json at {manifest} (not a finished run)")
    return OmegaConf.create(json.loads(manifest.read_text())["config"])


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
