# TODO: Implement adversarial trainer here, training classifier, or in this repository, the bornmachine, for robustness

import torch
import torch.nn as nn
import tensorkrowch as tk
from src.utils.evasion.minimal import RobustnessEvaluation
from dataclasses import dataclass
from src.models import BornMachine
import src.utils.schemas as schemas

# TODO: Implement one or more protocolls.

class Trainer:
    def __init__(self):
        pass
    def train(
        bornmachine: BornMachine,
        cfg: schemas.AdversarialConfig
        ):
        return NotImplementedError