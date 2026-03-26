import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

import hydra
import logging
from src.tracking.wandb_utils import init_wandb
from src.tracking import log_dataset_viz
from src.utils import schemas, set_seed
from src.data import DataHandler
from src.models import BornMachine
from src.trainer import ClassificationTrainer
import torch

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """
    Main entry point for classification training.

    Returns the best validation loss for HPO (Optuna sweeper).
    """
    # Initialising wandb and device
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataHandler (uses gen_dow_kwargs.seed and split_seed only)
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    # Seed training randomness (model init, DataLoader shuffling, PGD, sampling)
    set_seed(cfg.tracking.seed)

    # BornMachine
    bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    # Preprocessing (uses split_seed, independent of tracking.seed)
    datahandler.split_and_rescale(bornmachine)
    log_dataset_viz(datahandler)

    # Trainer
    trainer = ClassificationTrainer(bornmachine, cfg, "pre", datahandler, device)

    # Train
    trainer.train()

    # Finish
    run.finish()

    # Return objective for HPO (Optuna sweeper uses this)
    # Uses the metric specified by stop_crit in the classification config
    stop_crit = cfg.trainer.classification.stop_crit
    objective = trainer.best.get(stop_crit, float("inf"))
    # Negate accuracy/robustness metrics for minimization (Optuna minimizes by default)
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective

if __name__ == "__main__":
    main()