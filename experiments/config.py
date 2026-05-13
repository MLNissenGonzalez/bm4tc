"""Experiment-level configuration: dataclasses and Hydra registration.

Call register() once per process before @hydra.main to register all
structured configs with the Hydra ConfigStore.
"""
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from src.data.gen_n_load import DatasetConfig
from src.models.born import BornMachineConfig
from src.trainer.discriminative import DiscriminativeConfig
from src.trainer.generative import GenerativeConfig
from src.trainer.adversarial import AdversarialConfig
from src.utils.evasion import EvasionConfig


@dataclass
class TrainerConfig:
    discriminative: Optional[DiscriminativeConfig] = None
    adversarial: Optional[AdversarialConfig] = None
    generative: Optional[GenerativeConfig] = None


@dataclass
class TrackingConfig:
    project: str = "bm4tc"
    entity: str = ""
    mode: str = "disabled"
    seed: int = 42
    evasion: EvasionConfig = field(default_factory=EvasionConfig)


@dataclass
class Config:
    """Top-level configuration for an experiment."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    born: BornMachineConfig = field(default_factory=BornMachineConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment: str = "default"
    descriptor: str = ""


def register():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="dataset", name="schema", node=DatasetConfig)
    cs.store(group="model/born", name="schema", node=BornMachineConfig)
    cs.store(group="trainer", name="wrapper", node=TrainerConfig)
    cs.store(group="trainer/discriminative", name="schema", node=DiscriminativeConfig)
    cs.store(group="trainer/generative", name="schema", node=GenerativeConfig)
    cs.store(group="trainer/adversarial", name="schema", node=AdversarialConfig)
    cs.store(group="tracking", name="schema", node=TrackingConfig)
