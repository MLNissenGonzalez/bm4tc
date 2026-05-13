"""
Generative Training experiment entry point.

Trains the BornMachine generator using NLL minimization.
Optionally performs classification pretraining first.

Usage:
    # Train generative model (with classification pretraining)
    python -m experiments.generative +experiments=generative/fourier_d30D18/hpo/hpo

    # Quick test
    python -m experiments.generative +experiments=tests/generative tracking.mode=disabled
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import math
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import logging

# Geometric LR interpolation for alpha_curve_mixed configs.
# MixedNLL convention: alpha=0 → pure classification, alpha=1 → pure generative.
OmegaConf.register_new_resolver(
    "geom_lr",
    lambda alpha, lr_cls, lr_gen: math.exp(
        (1 - float(alpha)) * math.log(float(lr_cls)) + float(alpha) * math.log(float(lr_gen))
    ),
    replace=True,
)

from experiments.wandb_utils import init_wandb, log_dataset_viz
from experiments.logging import make_logger
from src.utils import schemas, set_seed, get
from src.data import DataHandler
from src.models import BornMachine
from src.trainer import DiscriminativeTrainer, GenerativeTrainer
import torch

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """Main entry point for generative training experiments."""
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading BornMachine from {model_path}")
        bornmachine = BornMachine.load(model_path)
        bornmachine.to(device)
    else:
        logger.info("Creating new BornMachine.")
        bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(bornmachine)
    log_dataset_viz(datahandler)

    if model_path is not None:
        bornmachine.reset()
        bornmachine.unset_data_nodes()

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)
    models_dir = run_dir / "models"

    if model_path is None:
        if getattr(cfg.trainer, 'discriminative', None) is not None:
            logger.info("Running classification pretraining.")
            pre_trainer = DiscriminativeTrainer(bornmachine, cfg.trainer.discriminative, datahandler, device)
            pre_trainer.train(on_epoch_end=logger_cb, output_dir=models_dir)
            bornmachine.reset()
            bornmachine.unset_data_nodes()
            bornmachine.to(device)
        else:
            logger.info("Skipping classification pretraining.")

    gen_trainer = None
    if cfg.trainer.generative is not None:
        logger.info("Starting generative training.")
        gen_trainer = GenerativeTrainer(bornmachine, cfg.trainer.generative, datahandler, device)
        gen_trainer.train(on_epoch_end=logger_cb, output_dir=models_dir)
    else:
        logger.error("No generative training config provided!")
        raise ValueError("trainer.generative config is required for this experiment.")

    run.finish()

    stop_crit = cfg.trainer.generative.stop_crit
    objective = gen_trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
