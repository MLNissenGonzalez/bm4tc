# TODO: Implement adversarial trainer here, training classifier, or in this repository, the bornmachine, for robustness

import torch
import torch.nn as nn
import tensorkrowch as tk
from src.evasion.minimal import RobustnessEvaluation
from dataclasses import dataclass
from src.models import BornMachine

# TODO: Implement one or more protocolls.

@dataclass
class AdTrainConfig:
    epochs: int = 200
    batch_size: int = 128
    relative_strength: float = 0.1 # strength = diam_norm(dataspace) * relative_strength
    norm: int | str = 2

def train(
        bornmachine: BornMachine,
        cfg: AdTrainConfig
):
    return "not implemented"