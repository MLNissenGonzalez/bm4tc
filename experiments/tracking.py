"""Experiment tracking: local log.json writer and optional W&B integration."""
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import wandb
import hydra
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


def init_wandb(cfg: Config) -> wandb.Run:
    """
    Initialize a W&B run from a Hydra config.

    Group name: ``{experiment}_{regime}_{archinfo}_{dataset}_{date}``
    Run name: job index (0-indexed).
    """
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)

    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(runtime_cfg.runtime.output_dir)
    job_num = int(runtime_cfg.job.get("num", 0))

    regime_parts = []
    nll_cfg = OmegaConf.select(cfg, "trainer.nll")
    if nll_cfg is not None:
        alpha = float(OmegaConf.select(cfg, "trainer.nll.alpha") or 0.0)
        regime_parts.append(f"nll_a{alpha:.2g}" if alpha > 0 else "nll")
    if OmegaConf.select(cfg, "trainer.adversarial") is not None:
        regime_parts.append("adv")
    regime = "_".join(regime_parts) or "none"

    _dtype = OmegaConf.select(cfg, "born.init_kwargs.dtype")
    _dtype_suffix = {"complex64": "c64", "complex128": "c128"}.get(_dtype, "")
    archinfo = f"d{cfg.born.init_kwargs.in_dim}D{cfg.born.init_kwargs.bond_dim}{_dtype_suffix}{cfg.born.embedding}"

    mode = runtime_cfg.mode.value
    now = datetime.now()
    date_str = now.strftime("%d%m_%H%M") if mode == 1 else now.strftime("%d%m")

    group_key = f"{cfg.experiment}_{regime}_{archinfo}_{cfg.dataset.name}_{date_str}"

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
