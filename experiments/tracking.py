"""Experiment tracking: local log.json writer and optional W&B integration."""
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import wandb
import hydra
from hydra.types import RunMode
from omegaconf import OmegaConf

from experiments.config import Config

logger = logging.getLogger(__name__)


def make_logger(output_dir: Path, wandb_run=None) -> Callable[[int, dict], None]:
    """
    Returns an on_epoch_end callback that writes epoch metrics to log.json
    and optionally forwards them to a W&B run.
    """
    log_path = output_dir / "log.json"
    records = []

    def log(epoch: int, metrics: dict) -> None:
        records.append({"epoch": epoch, **metrics})
        log_path.write_text(json.dumps(records, indent=2))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **metrics})

    return log


def _derive_group_key(run_dir: Path, is_multirun: bool) -> str:
    """Derive the W&B group from the Hydra output directory.

    The output dir is the single source of truth for a run's location and is
    identical for every job in a sweep, so deriving the group from it keeps all
    sweep runs in one group. (The previous ``datetime.now()`` approach gave each
    job a different ``_HHMM`` minute, scattering one sweep across many groups.)

    Returns the path components after the ``outputs`` root, i.e.
    ``{dataset}/{nat|at}/{embedding}/{arch}/[{stage}/]{experiment}_{date}``.
    For multiruns the per-job subdir (``hydra.job.num``) is stripped so all jobs
    share the sweep-root group.
    """
    group_dir = run_dir.parent if is_multirun else run_dir
    parts = group_dir.parts
    idx = len(parts) - 1 - parts[::-1].index("outputs")
    return "/".join(parts[idx + 1:])


def init_wandb(cfg: Config) -> wandb.Run:
    """
    Initialize a W&B run from a Hydra config.

    Group: the run's output dir relative to the outputs root, i.e.
    ``{dataset}/{nat|at}/{embedding}/{arch}/[{stage}/]{experiment}_{date}``.
    Run name: job index (0-indexed).
    """
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)

    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(runtime_cfg.runtime.output_dir).resolve()
    job_num = int(runtime_cfg.job.get("num", 0))
    is_multirun = runtime_cfg.mode == RunMode.MULTIRUN

    group_key = _derive_group_key(run_dir, is_multirun)

    run = wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        dir=str(run_dir),
        config=wandb_cfg,
        group=group_key,
        name=str(job_num),
        mode=cfg.tracking.mode,
        reinit="finish_previous"
    )
    return run


def log_dataset_viz(datahandler) -> None:
    """Log a scatter plot of the full dataset to W&B under 'dataset/all'."""
    if datahandler.data_dim != 2:
        logger.info(f"Skipping dataset viz for data_dim={datahandler.data_dim} (only 2D supported)")
        return

    from src.analysis.viz import visualise_samples
    import torch

    all_data = torch.cat([datahandler.data[s] for s in ("train", "valid", "test")], dim=0)
    all_labels = torch.cat([datahandler.labels[s] for s in ("train", "valid", "test")])
    ax = visualise_samples(all_data, all_labels, input_range=datahandler.input_range)
    fig = ax.figure
    wandb.log({"dataset/all": wandb.Image(fig)})
    plt.close(fig)
