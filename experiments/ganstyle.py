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
import torch

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):

    # Initialising wandb and device
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.tracking.seed)

    # DataHandler
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    # BornMachine
    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        logger.info(f"Loading BornMachine from {cfg.model_path}")
        born = BornMachine.load(cfg.model_path)
        born.to(device)
        # Preprocessing
        datahandler.split_and_rescale(born)
    else:
        logger.info("Creating and training new BornMachine.")
        bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)
        # Preprocessing
        datahandler.split_and_rescale(bornmachine)
        # Pretrain
        pre_trainer = ClassificationTrainer(bornmachine, cfg, "pre", datahandler, device)
        pre_trainer.train()
    

    # Critic TODO: Think about how to actually use backbone flexibility
    critic = Critic(cfg.trainer.ganstyle, datahandler) 
    
    # Gantrain
    gan_trainer = GANStyleTrainer(bornmachine, cfg, datahandler, critic, device, pre_trainer.best)
    gan_trainer.train()

    # Finish
    run.finish()


if __name__ == "__main__":
    main()