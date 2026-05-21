"""
Adversarial Training experiment entry point.

Supports two workflows:
1. Train from scratch: NLL pretraining (alpha=0) -> Adversarial training
2. Fine-tune: Load pretrained model -> Adversarial training

Usage:
    # Train from scratch with PGD-AT
    python -m experiments.adversarial +experiments=tests/adversarial tracking.mode=disabled

    # Fine-tune a pretrained model
    python -m experiments.adversarial trainer/adversarial=pgd_at model_path=/path/to/model

    # HPO on moons dataset (fourier d30r18 architecture)
    python -m experiments.adversarial --multirun +experiments=adversarial/fourier/d30r18/hpo/moons
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import logging
from pathlib import Path
from experiments.tracking import init_wandb, log_dataset_viz, make_logger
from experiments.config import Config, register
from src.utils import set_seed
from src.data import DataHandler
from src.models.cbm import ConditionalBornMachine
from src.trainer import NLLTrainer, AdversarialTrainer
import torch

logger = logging.getLogger(__name__)
register()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):
    """Main entry point for adversarial training experiments."""
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading pretrained ConditionalBornMachine from {model_path}")
        cbm = ConditionalBornMachine.load(model_path)
        cbm.to(device)
    else:
        logger.info("Creating new ConditionalBornMachine.")
        cbm = ConditionalBornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(cbm)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)
    models_dir = run_dir / "models"

    if model_path is None:
        if cfg.trainer.nll is not None:
            pre_trainer = NLLTrainer(cbm, cfg.trainer.nll, datahandler, device)
            pre_trainer.train(on_epoch_end=logger_cb, output_dir=models_dir)
        else:
            logger.warning("No NLL config provided, starting adversarial training from random init.")

    adv_trainer = None
    if cfg.trainer.adversarial is not None:
        logger.info("Starting adversarial training.")
        adv_trainer = AdversarialTrainer(cbm, cfg.trainer.adversarial, datahandler, device)
        adv_trainer.train(on_epoch_end=logger_cb, output_dir=models_dir)
    else:
        logger.error("No adversarial training config provided!")
        raise ValueError("trainer.adversarial config is required for this experiment.")

    run.finish()

    stop_crit = cfg.trainer.adversarial.stop_crit
    objective = adv_trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ("acc", "rob"):
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
