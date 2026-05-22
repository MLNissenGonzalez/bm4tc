"""
NLL training entry point (discriminative, generative, or mixed).

alpha=0  → pure discriminative (default)
alpha>0  → mixed NLL; alpha=1 → pure generative

Usage:
    python -m experiments.nll +experiments=tests/nll tracking.mode=disabled
    python -m experiments.nll +experiments=nll/fourier/d4r3/hpo/moons tracking.mode=disabled
    python -m experiments.nll --multirun +experiments=nll/legendre/d10r6/seed_sweep/circles
"""
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import logging
from pathlib import Path
from omegaconf import OmegaConf

# geom_lr: geometric LR interpolation for alpha_curve configs.
# Usage in YAML: lr: ${geom_lr:${trainer.nll.alpha},<lr_cls>,<lr_gen>}
# Configs that don't sweep alpha just set lr directly — this resolver is never called.
OmegaConf.register_new_resolver(
    "geom_lr",
    lambda alpha, lr_cls, lr_gen: math.exp(
        (1 - float(alpha)) * math.log(float(lr_cls)) + float(alpha) * math.log(float(lr_gen))
    ),
    replace=True,
)

from experiments.tracking import init_wandb, log_dataset_viz, make_logger
from experiments.config import Config, register
from src.utils import set_seed
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine
from src.train import NLLTrainer
import torch

logger = logging.getLogger(__name__)
register()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):
    """Main entry point for NLL training experiments."""
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading ConditionalBornMachine from {model_path}")
        cbm = ConditionalBornMachine.load(model_path)
        cbm.to(device)
    else:
        cbm = ConditionalBornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(cbm)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)

    trainer = NLLTrainer(cbm, cfg.trainer.nll, datahandler, device)
    trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")

    run.finish()

    stop_crit = cfg.trainer.nll.stop_crit
    objective = trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ("acc", "rob"):
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
