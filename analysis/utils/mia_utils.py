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
    which can be used to reconstruct the DataHandler and BornMachine.

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


def load_run_config_from_wandb(
    entity: str,
    project: str,
    run_id: str
) -> DictConfig:
    """Fetch config from a wandb run.

    This retrieves the configuration logged to Weights & Biases for a
    specific run, allowing analysis without local files.

    Args:
        entity: Wandb entity (username or team name).
        project: Wandb project name.
        run_id: Wandb run ID (the unique identifier, not the run name).

    Returns:
        OmegaConf DictConfig with the run configuration.

    Raises:
        ImportError: If wandb is not installed.
        wandb.errors.CommError: If the run cannot be found.

    Example:
        >>> cfg = load_run_config_from_wandb("my-team", "gan_train", "abc123")
        >>> print(cfg.dataset.name)
        'spirals_4k'
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is required for loading configs from wandb. "
            "Install with: pip install wandb"
        )

    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"

    logger.info(f"Fetching config from wandb run: {run_path}")
    run = api.run(run_path)

    # Convert wandb config dict to OmegaConf
    cfg = OmegaConf.create(run.config)
    return cfg


def find_model_checkpoint(
    run_dir: Union[str, Path],
    checkpoint_name: Optional[str] = None
) -> Path:
    """Find model checkpoint file in run output folder.

    Searches for .pt files in the models/ subdirectory of the run output.

    Args:
        run_dir: Path to the run output directory.
        checkpoint_name: Specific checkpoint filename to look for.
                         If None, returns the first .pt file found.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint is found.

    Example:
        >>> checkpoint = find_model_checkpoint("outputs/classification_2024_01_15")
        >>> bornmachine = BornMachine.load(str(checkpoint))
    """
    run_dir = Path(run_dir)
    models_dir = run_dir / "models"

    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found at {models_dir}. "
            f"Expected checkpoint at: {run_dir}/models/*.pt"
        )

    if checkpoint_name:
        checkpoint_path = models_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found in {models_dir}"
        )

    # Find all .pt files
    checkpoints = list(models_dir.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoint files found in {models_dir}"
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


def download_wandb_checkpoint(
    entity: str,
    project: str,
    run_id: str,
    output_dir: Union[str, Path],
    checkpoint_name: str = "model.pt"
) -> Path:
    """Download model checkpoint from a wandb run.

    Args:
        entity: Wandb entity (username or team name).
        project: Wandb project name.
        run_id: Wandb run ID.
        output_dir: Local directory to save the checkpoint.
        checkpoint_name: Name of the checkpoint file in wandb artifacts.

    Returns:
        Path to the downloaded checkpoint file.

    Raises:
        ImportError: If wandb is not installed.
        FileNotFoundError: If checkpoint not found in wandb run.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is required for downloading checkpoints. "
            "Install with: pip install wandb"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"

    logger.info(f"Fetching checkpoint from wandb run: {run_path}")
    run = api.run(run_path)

    # Try to download from run files
    for file in run.files():
        if file.name.endswith(".pt") or checkpoint_name in file.name:
            local_path = output_dir / file.name
            file.download(root=str(output_dir), replace=True)
            logger.info(f"Downloaded checkpoint to: {local_path}")
            return local_path

    raise FileNotFoundError(
        f"No .pt checkpoint found in wandb run {run_path}. "
        "Available files: " + ", ".join(f.name for f in run.files())
    )
