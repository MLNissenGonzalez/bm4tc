import src.utils.schemas as schemas
import wandb
import hydra
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def init_wandb(cfg: schemas.Config) -> wandb.Run:
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
    for key, code in [("trainer.discriminative", "dis"), ("trainer.generative", "gen"),
                      ("trainer.adversarial", "adv")]:
        if OmegaConf.select(cfg, key) is not None:
            regime_parts.append(code)
    regime = "".join(regime_parts) or "none"

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


def log_dataset_viz(datahandler):
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
