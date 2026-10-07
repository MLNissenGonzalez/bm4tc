"""Experiment tracking: local log.json writer and optional W&B integration."""
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import wandb
from omegaconf import OmegaConf

from bm4tc.pipeline.config import Config
from bm4tc.pipeline.metrics import flatten_epoch

logger = logging.getLogger(__name__)


def make_logger(output_dir: Path, wandb_run=None) -> Callable[[int, dict], None]:
    """
    Returns an on_epoch_end callback that writes epoch metrics to log.json
    and optionally forwards them to a W&B run. The trainer passes plain names per
    split; the logged keys come from :mod:`bm4tc.pipeline.metrics`.
    """
    log_path = output_dir / "log.json"
    records = []

    def log(epoch: int, record: dict) -> None:
        metrics = flatten_epoch(record)
        records.append({"epoch": epoch, **metrics})
        log_path.write_text(json.dumps(records, indent=2))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **metrics})

    return log


def init_wandb(cfg: Config, run_dir: Path, names: Dict[str, str]) -> wandb.Run:
    """Start a W&B run in ``run_dir``; ``names`` (group, name, job_type) come from
    :meth:`bm4tc.pipeline.runs.Job.wandb` (D47)."""
    return wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.tracking.mode,
        reinit="finish_previous",
        **names,
    )


def log_dataset_viz(datahandler) -> None:
    """Log a scatter plot of the full dataset to W&B under 'dataset/all'."""
    if datahandler.data_dim != 2:
        logger.info(f"Skipping dataset viz for data_dim={datahandler.data_dim} (only 2D supported)")
        return

    from bm4tc.analysis.viz import visualise_samples
    import torch

    all_data = torch.cat([datahandler.data[s] for s in ("train", "valid", "test")], dim=0)
    all_labels = torch.cat([datahandler.labels[s] for s in ("train", "valid", "test")])
    ax = visualise_samples(all_data, all_labels, input_range=datahandler.input_range)
    fig = ax.figure
    wandb.log({"dataset/all": wandb.Image(fig)})
    plt.close(fig)
