"""
Softmax sanity-check experiment entry point.

Trains the CBM classifier using raw amplitudes as softmax logits,
matching tutorial MPS implementations. Used to sanity-check that the
MPS architecture can learn when trained with a standard softmax loss.

Two variants (configured via +experiments=tests/softmax/<config>):
  - legendre_mnist  : Legendre d3r10 embedding, MNIST
  - simp_mnist      : SimpEmbedding (1, x, 1-x) d3r10, MNIST

Run:
    python -m experiments.softmax_sanity +experiments=tests/softmax/legendre_mnist
    python -m experiments.softmax_sanity +experiments=tests/softmax/simp_mnist
"""

import hydra
import torch
import logging
from pathlib import Path
from experiments.tracking import init_wandb, log_dataset_viz, make_logger
from experiments.config import Config, register
from src.utils import set_seed
from src.data import DataHandler
from src.model import CBMConfig, ConditionalBornMachine
from src.train.softmax import SoftmaxTrainer

logger = logging.getLogger(__name__)
register()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    cbm_cfg = CBMConfig(
        embedding=cfg.born.embedding,
        init_kwargs=cfg.born.init_kwargs,
    )
    cbm = ConditionalBornMachine(cbm_cfg, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(cbm)
    log_dataset_viz(datahandler)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger_cb = make_logger(run_dir, wandb_run=run)

    trainer = SoftmaxTrainer(cbm, cfg.trainer.discriminative, datahandler, device)
    trainer.train(on_epoch_end=logger_cb, output_dir=run_dir / "models")

    run.finish()

    stop_crit = cfg.trainer.discriminative.stop_crit
    objective = trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
