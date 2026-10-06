"""
Unified training entry point (NLL discriminative/generative, adversarial).

Regime is inferred from which trainer config is set:
  cfg.trainer.at is not None   →  at  (AdversarialTrainer)
  cfg.trainer.nat is not None  →  nat (NLLTrainer)

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
from src.train import NLLTrainer, AdversarialTrainer
import torch
from omegaconf import OmegaConf

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
        cbm = ConditionalBornMachine.load(model_path)
        # accumulate is an inference-path toggle, not part of the checkpoint
        # weights — honor the current run's born.accumulate on loaded models
        # (fresh models read it at construction). Lets pretrained/fine-tune runs
        # opt into the overflow-safe path even though the a0 checkpoint predates it.
        cbm.accumulate = cfg.born.accumulate
        # Persist the override into the model's own config so the checkpoint
        # saved after training records the flag actually used — otherwise save()
        # would serialize the stale flag inherited from the loaded checkpoint,
        # and later analysis loads would read the wrong value.
        OmegaConf.set_struct(cbm.cfg, False)
        cbm.cfg.accumulate = cbm.accumulate
        OmegaConf.set_struct(cbm.cfg, True)
        cbm.to(device)
    else:
        cbm = ConditionalBornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(cbm)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)

    if cfg.trainer.get("at") is not None:
        trainer = AdversarialTrainer(cbm, cfg.trainer.at, datahandler, device)
    elif cfg.trainer.get("nat") is not None:
        trainer = NLLTrainer(cbm, cfg.trainer.nat, datahandler, device)
    else:
        raise ValueError("No trainer config: set trainer.nat or trainer.at")
    trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")

    run.finish()
    return trainer.best["objective"]


if __name__ == "__main__":
    main()
