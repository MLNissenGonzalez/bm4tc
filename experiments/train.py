"""
Training entry point for NAT and AT (one Trainer; AT is the run with an attack,
``trainer.evasion`` set).

Returns the best validation objective (D8), which Optuna minimises.

Usage:
    python -m experiments.train +experiments=tests/nat tracking.mode=disabled
    python -m experiments.train +experiments=tests/at tracking.mode=disabled
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import logging
from pathlib import Path

from experiments.resolvers import register_resolvers
register_resolvers()

from experiments.tracking import init_wandb, log_dataset_viz, make_logger
from experiments.config import Config, register
from src.utils import set_seed
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine
from src.train import Trainer
import torch

logger = logging.getLogger(__name__)
register()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config) -> float:
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    set_seed(cfg.tracking.seed)

    model_path = cfg.model_path
    if model_path is not None:
        logger.info(f"Loading ConditionalBornMachine from {model_path}")
        cbm = ConditionalBornMachine.load(model_path, accumulate=cfg.born.accumulate)
        cbm.to(device)
    else:
        cbm = ConditionalBornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(cbm)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)

    trainer = Trainer(cbm, cfg.trainer, datahandler, device)
    trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")

    run.finish()
    return trainer.best["objective"]


if __name__ == "__main__":
    main()
