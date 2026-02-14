"""
Generative Training experiment entry point.

Trains the BornMachine generator using NLL minimization.
Optionally performs classification pretraining first.

Usage:
    # Train generative model (with classification pretraining)
    python -m experiments.generative +experiments=generative/default

    # Quick test
    python -m experiments.generative +experiments=tests/generative tracking.mode=disabled
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

import hydra
import logging
from src.tracking.wandb_utils import init_wandb
from src.tracking import evaluate_loaded_model
# Think about initializing the generative loss in the trainer?
from src.utils import schemas, set_seed, get
from src.data import DataHandler
from src.models import BornMachine
from src.trainer import ClassificationTrainer, GenerativeTrainer
import torch

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """
    Main entry point for generative training experiments.

    Returns the best validation loss for HPO (Optuna sweeper).
    """
    # Initialize wandb and device
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataHandler (uses gen_dow_kwargs.seed and split_seed only)
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    # Seed training randomness (model init, DataLoader shuffling, PGD, sampling)
    set_seed(cfg.tracking.seed)

    # BornMachine
    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading BornMachine from {model_path}")
        bornmachine = BornMachine.load(model_path)
        bornmachine.to(device)
    else:
        logger.info("Creating new BornMachine.")
        bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    # Preprocessing (uses split_seed, independent of tracking.seed)
    datahandler.split_and_rescale(bornmachine)

    if model_path is not None:
        evaluate_loaded_model(cfg, bornmachine, datahandler, device)

    if model_path is None:
        # Classification pretraining (optional)
        if getattr(cfg.trainer, 'classification', None) is not None:
            logger.info("Running classification pretraining.")
            pre_trainer = ClassificationTrainer(bornmachine, cfg, "pre", datahandler, device)
            pre_trainer.train()
            # Move back to device after pretraining (it moves to CPU at end)
            bornmachine.to(device)
        else:
            logger.info("Skipping classification pretraining.")

    # Generative Training
    gen_trainer = None
    if cfg.trainer.generative is not None:
        logger.info("Starting generative training.")
        gen_trainer = GenerativeTrainer(bornmachine, cfg, datahandler, device)
        gen_trainer.train()
    else:
        logger.error("No generative training config provided!")
        raise ValueError("trainer.generative config is required for this experiment.")

    # Finish
    run.finish()

    # Return objective for HPO (Optuna sweeper uses this)
    # Returns the metric specified by stop_crit in the generative config
    stop_crit = cfg.trainer.generative.stop_crit
    objective = gen_trainer.best.get(stop_crit, float("inf"))
    # Negate accuracy/robustness metrics for minimization (Optuna minimizes by default)
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
