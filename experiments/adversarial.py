"""
Adversarial Training experiment entry point.

Supports two workflows:
1. Train from scratch: Classification pretraining -> Adversarial training
2. Fine-tune: Load pretrained model -> Adversarial training

Usage:
    # Train from scratch with PGD-AT
    python -m experiments.adversarial trainer/adversarial=pgd_at dataset=moons_2k

    # Fine-tune a pretrained model
    python -m experiments.adversarial trainer/adversarial=pgd_at model_path=/path/to/model

    # HPO on moons dataset (fourier d30D18 architecture)
    python -m experiments.adversarial --multirun +experiments=adversarial/fourier_d30D18/hpo/moons

    # Quick test
    python -m experiments.adversarial +experiments=tests/adversarial tracking.mode=disabled
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import logging
from pathlib import Path
from experiments.wandb_utils import init_wandb, log_dataset_viz
from experiments.logging import make_logger
from src.utils import schemas, set_seed
from src.data import DataHandler
from src.models import BornMachine
from src.trainer import DiscriminativeTrainer, AdversarialTrainer
import torch

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """Main entry point for adversarial training experiments."""
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading pretrained BornMachine from {model_path}")
        bornmachine = BornMachine.load(model_path)
        bornmachine.to(device)
    else:
        logger.info("Creating new BornMachine and running classification pretraining.")
        bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(bornmachine)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)
    models_dir = run_dir / "models"

    if model_path is None:
        if cfg.trainer.discriminative is not None:
            pre_trainer = DiscriminativeTrainer(bornmachine, cfg.trainer.discriminative, datahandler, device)
            pre_trainer.train(on_epoch_end=logger_cb, output_dir=models_dir)
            bornmachine.to(device)
        else:
            logger.warning("No classification config provided, starting adversarial training from random init.")

    adv_trainer = None
    if cfg.trainer.adversarial is not None:
        logger.info(f"Starting adversarial training with method: {cfg.trainer.adversarial.method}")
        adv_trainer = AdversarialTrainer(bornmachine, cfg.trainer.adversarial, datahandler, device)
        adv_trainer.train(on_epoch_end=logger_cb, output_dir=models_dir)
    else:
        logger.error("No adversarial training config provided!")
        raise ValueError("trainer.adversarial config is required for this experiment.")

    run.finish()

    stop_crit = cfg.trainer.adversarial.stop_crit
    objective = adv_trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
