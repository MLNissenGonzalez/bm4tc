import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

import hydra
import logging
from src.tracking.wandb_utils import init_wandb
from src.utils import schemas, set_seed
from src.data import DataHandler
from src.models import BornMachine, Critic
from src.trainer import ClassificationTrainer, GANStyleTrainer
from src.tracking import PerformanceEvaluator, evaluate_loaded_model, log_dataset_viz
import torch

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """
    Main entry point for GAN-style training.

    Returns the best validation metric (determined by stop_crit) for HPO.
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
    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading BornMachine from {model_path}")
        bornmachine = BornMachine.load(model_path)
        bornmachine.to(device)
        pre_trainer = None
    else:
        logger.info("Creating and training new BornMachine.")
        bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    # Preprocessing (uses split_seed, independent of tracking.seed)
    datahandler.split_and_rescale(bornmachine)
    log_dataset_viz(datahandler)

    if model_path is not None:
        evaluate_loaded_model(cfg, bornmachine, datahandler, device)

    if model_path is None:
        # Pretrain
        pre_trainer = ClassificationTrainer(bornmachine, cfg, "pre", datahandler, device)
        pre_trainer.train()


    # Critic TODO: Think about how to actually use backbone flexibility
    critic = Critic(cfg.trainer.ganstyle, datahandler, device=device)


    # Gantrain
    if pre_trainer is None:
        logger.info("Evaluating former best performance.")
        evaluator = PerformanceEvaluator(cfg, datahandler, cfg.trainer.ganstyle, device)
        former_best = evaluator.evaluate(bornmachine, split="valid", step=0) # evaluator is not needed after this
    else:
        former_best = pre_trainer.best

    gan_trainer = GANStyleTrainer(bornmachine, cfg, datahandler, critic, device, former_best)
    gan_trainer.train()

    # Finish
    run.finish()

    # Return objective for HPO (Optuna sweeper uses this)
    # Returns the metric specified by stop_crit in the ganstyle config
    stop_crit = cfg.trainer.ganstyle.stop_crit
    objective = gan_trainer.best.get(stop_crit, float("inf"))
    # Negate accuracy/robustness metrics for minimization (Optuna minimizes by default)
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()