"""
Adversarial Training experiment entry point.

Supports two workflows:
1. Train from scratch: Classification pretraining -> Adversarial training
2. Fine-tune: Load pretrained model -> Adversarial training

Usage:
    # Train from scratch with PGD-AT
    python -m experiments.adversarial trainer/adversarial=pgd_at dataset=moons_2k

    # Train from scratch with TRADES
    python -m experiments.adversarial trainer/adversarial=trades dataset=moons_2k

    # Fine-tune a pretrained model
    python -m experiments.adversarial trainer/adversarial=pgd_at model_path=/path/to/model

    # Quick test
    python -m experiments.adversarial +experiments=tests/adversarial tracking.mode=disabled
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

import hydra
import logging
from src.tracking.wandb_utils import init_wandb
from src.tracking import evaluate_loaded_model
from src.utils import schemas, set_seed
from src.data import DataHandler
from src.models import BornMachine
from src.trainer import ClassificationTrainer, AdversarialTrainer
import torch

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """
    Main entry point for adversarial training experiments.

    If model_path is provided in config, loads a pretrained model.
    Otherwise, trains a new model with classification first.

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

    # BornMachine - either load or create new
    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading pretrained BornMachine from {model_path}")
        bornmachine = BornMachine.load(model_path)
        bornmachine.to(device)
    else:
        logger.info("Creating new BornMachine and running classification pretraining.")
        bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    # Preprocessing (uses split_seed, independent of tracking.seed)
    datahandler.split_and_rescale(bornmachine)

    if model_path is not None:
        evaluate_loaded_model(cfg, bornmachine, datahandler, device)

    if model_path is None:
        # Classification pretraining
        if cfg.trainer.classification is not None:
            pre_trainer = ClassificationTrainer(bornmachine, cfg, "pre", datahandler, device)
            pre_trainer.train()
            # Move back to device after pretraining (it moves to CPU at end)
            bornmachine.to(device)
        else:
            logger.warning("No classification config provided, starting adversarial training from random init.")

    # Adversarial Training
    adv_trainer = None
    if cfg.trainer.adversarial is not None:
        logger.info(f"Starting adversarial training with method: {cfg.trainer.adversarial.method}")
        adv_trainer = AdversarialTrainer(bornmachine, cfg, "adv", datahandler, device)
        adv_trainer.train()
    else:
        logger.error("No adversarial training config provided!")
        raise ValueError("trainer.adversarial config is required for this experiment.")

    # Finish
    run.finish()

    # Return objective for HPO (Optuna sweeper uses this)
    # Uses the metric specified by stop_crit in the adversarial config
    stop_crit = cfg.trainer.adversarial.stop_crit
    objective = adv_trainer.best.get(stop_crit, float("inf"))
    # Negate accuracy/robustness metrics for minimization (Optuna minimizes by default)
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
