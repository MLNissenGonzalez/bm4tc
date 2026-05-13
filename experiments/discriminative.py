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
from src.trainer import ClassificationTrainer
import torch

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    """Main entry point for discriminative training."""
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(bornmachine)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)

    trainer = ClassificationTrainer(bornmachine, cfg, "pre", datahandler, device)
    trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")

    run.finish()

    stop_crit = cfg.trainer.discriminative.stop_crit
    objective = trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
