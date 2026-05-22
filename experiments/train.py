"""
Unified training entry point (NLL discriminative/generative, adversarial).

Mode is inferred from which trainer config is set:
  cfg.trainer.adversarial is not None  →  adversarial (AdversarialTrainer only)
  cfg.trainer.nll is not None          →  nll (NLLTrainer)

Usage:
    python -m experiments.train +experiments=tests/nll tracking.mode=disabled
    python -m experiments.train +experiments=tests/adversarial tracking.mode=disabled
    python -m experiments.train --multirun +experiments=nll/gen/legendre/d10r6/seed_sweep/circles
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
from src.train import NLLTrainer, AdversarialTrainer
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

    if cfg.trainer.adversarial is not None:
        trainer = AdversarialTrainer(cbm, cfg.trainer.adversarial, datahandler, device)
        trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")
        stop_crit = cfg.trainer.adversarial.stop_crit
    elif cfg.trainer.nll is not None:
        trainer = NLLTrainer(cbm, cfg.trainer.nll, datahandler, device)
        trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")
        stop_crit = cfg.trainer.nll.stop_crit
    else:
        raise ValueError("No trainer config: set trainer.nll or trainer.adversarial")

    run.finish()

    objective = trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ("acc", "rob"):
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
