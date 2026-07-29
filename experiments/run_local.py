"""
Self-contained training script for ConditionalBornMachine.
No Hydra, no W&B. Edit the CONFIG BLOCK below, then run:

    python -m experiments.run_local
"""

import logging
import sys

from pathlib import Path
import torch

from src.datahandler import DataHandler, DatasetConfig, DataGenDowConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.train import NLLTrainer, NLLConfig, NormControlConfig, AdversarialTrainer, AdversarialConfig
from src.utils import set_seed
from src.utils.train import OptimizerConfig
from src.utils.evasion import EvasionConfig

# =============================================================================
# CONFIG BLOCK — edit here, leave the rest untouched
# =============================================================================

# Regime: "nll" (discriminative or generative) | "adversarial" (PGD-AT)
REGIME = "nll"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset ------------------------------------------------------------------
# 2-D toy:  "moons" | "circles" | "spirals"
# Image:    "mnist"
# UCR time-series: "ecg200" | "italypowerdemand" | "chlorineconcentration"
#                  "syntheticcontrol" | "cricketx" | "crickety" | "cricketz"
DATASET_CFG = DatasetConfig(
    name="spirals",
    gen_dow_kwargs=DataGenDowConfig(
        name="spirals",
        size=500,         # samples per class → 1000 total; set overwrite=True below once to regen
        seed=42,
        noise=0.1,        # label noise fraction (spirals need an explicit value)
    ),
    split=(0.7, 0.15, 0.15),
    split_seed=42,
    overwrite=True,       # always regenerate so size/noise/seed changes take effect immediately
)

# --- Architecture -------------------------------------------------------------
# Arch presets (in_dim × bond_dim):
#   d4r3  → MPSInitConfig(in_dim=4,  bond_dim=3)
#   d6r4  → MPSInitConfig(in_dim=6,  bond_dim=4)
#   d10r6 → MPSInitConfig(in_dim=10, bond_dim=6)   ← default
#   d30r18→ MPSInitConfig(in_dim=30, bond_dim=18)
#
# Embedding: "fourier" | "legendre" | "hermite" | "chebychev1" | "chebychev2"
CBM_CFG = CBMConfig(
    init_kwargs=MPSInitConfig(
        in_dim=10,
        bond_dim=6,
        init_method="randn_eye",  # "randn" | "randn_eye" | "canonical"
    ),
    embedding="legendre",
)

# --- NLL trainer (used when REGIME="nll") -------------------------------------
# alpha controls the loss mix:
#   alpha=0.0 → pure discriminative  (-log p(c|x))
#   alpha=0.5 → mixed NLL            (50 % dis + 50 % gen)
#   alpha=1.0 → pure generative      (-log p(x,c))
#
# stop_crit: "acc" | "dis_loss" | "gen_loss" | "rob"
NLL_CFG = NLLConfig(
    alpha=0.5,
    max_epoch=80,
    batch_size=64,
    optimizer=OptimizerConfig(kwargs={"lr": 1e-5}),
    stop_crit="acc",
    patience=250,
    norm_control=NormControlConfig(target=1.0, hard_every=1, soft_strength=0.0),
    save=False,  # set True + output_dir in run_nll() to checkpoint the best model
)

# --- Adversarial trainer (used when REGIME="adversarial") --------------------
# method: "FGM" (fast, single-step) | "PGD" (multi-step, stronger)
# eps_rel: list of relative ε values (fractions of the input domain width)
# eval_rob_freq=0 disables robustness eval (faster epochs)
ADV_CFG = AdversarialConfig(
    max_epoch=30,
    batch_size=64,
    optimizer=OptimizerConfig(kwargs={"lr": 1e-4}),
    evasion=EvasionConfig(
        method="PGD",
        norm="inf",
        eps_rel=[0.1],   # fraction of the input domain; abs 0.2 on legendre
        num_steps=10,
        random_start=True,
    ),
    stop_crit="acc",
    patience=100,
    eval_rob_freq=5,   # evaluate robustness every N epochs (0 = never)
    clean_weight=0.0,  # weight of clean loss in adversarial objective (TRADES: 1.0)
    save=False,
)

# Checkpoint to load before adversarial training (fine-tune a pre-trained CBM).
# None → train from scratch.
# Example: MODEL_PATH = Path("outputs/seed_sweep/dis/fourier/d10r6/spirals_2205/0/models/model.pt")
MODEL_PATH = None

# =============================================================================
# END CONFIG BLOCK
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _make_callback(regime: str):
    def on_epoch_end(epoch: int, metrics: dict):
        nan = float("nan")
        if regime == "nll":
            logger.info(
                f"epoch {epoch:4d} | "
                f"nll/train={metrics.get('nll/train', nan):.4f}  "
                f"dis_loss={metrics.get('dis_loss/valid', nan):.4f}  "
                f"gen_loss={metrics.get('gen_loss/valid', nan):.4f}  "
                f"acc={metrics.get('acc/valid', nan):.4f}"
            )
        else:
            rob_str = (
                f"  rob={metrics['rob/valid']:.4f}" if "rob/valid" in metrics else ""
            )
            logger.info(
                f"epoch {epoch:4d} | "
                f"dis_loss={metrics.get('dis_loss/valid', nan):.4f}  "
                f"acc={metrics.get('acc/valid', nan):.4f}"
                f"{rob_str}"
            )
    return on_epoch_end


def run_nll():
    device = torch.device(DEVICE)
    dh = DataHandler(DATASET_CFG)
    dh.load()
    set_seed(SEED)
    cbm = ConditionalBornMachine(CBM_CFG, dh.data_dim, dh.num_cls, device)
    dh.split_and_rescale(cbm)
    trainer = NLLTrainer(cbm, NLL_CFG, dh, device)
    trainer.train(on_epoch_end=_make_callback("nll"))
    logger.info(f"Best: {trainer.best}")


def run_adversarial():
    device = torch.device(DEVICE)
    dh = DataHandler(DATASET_CFG)
    dh.load()
    set_seed(SEED)
    if MODEL_PATH is not None:
        logger.info(f"Loading checkpoint from {MODEL_PATH}")
        cbm = ConditionalBornMachine.load(MODEL_PATH)
        cbm.to(device)
    else:
        cbm = ConditionalBornMachine(CBM_CFG, dh.data_dim, dh.num_cls, device)
    dh.split_and_rescale(cbm)
    trainer = AdversarialTrainer(cbm, ADV_CFG, dh, device)
    trainer.train(on_epoch_end=_make_callback("adversarial"))
    logger.info(f"Best: {trainer.best}")


if __name__ == "__main__":
    if REGIME == "nll":
        run_nll()
    elif REGIME == "adversarial":
        run_adversarial()
    else:
        raise ValueError(f"Unknown REGIME: {REGIME!r}. Use 'nll' or 'adversarial'.")
