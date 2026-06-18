"""Experiment-level configuration: dataclasses and Hydra registration.

Call register() once per process before @hydra.main to register all
structured configs with the Hydra ConfigStore.
"""
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from src.datahandler import DatasetConfig
from src.model import CBMConfig
from src.train.nll import NLLConfig
from src.train.adversarial import AdversarialConfig
from src.utils.evasion import EvasionConfig


@dataclass
class TrainerConfig:
    nll: Optional[NLLConfig] = None
    adversarial: Optional[AdversarialConfig] = None


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
    born: CBMConfig = field(default_factory=CBMConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment: str = "default"
    descriptor: str = ""
    model_path: Optional[str] = None


def register():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="dataset", name="schema", node=DatasetConfig)
    cs.store(group="model/born", name="schema", node=CBMConfig)
    cs.store(group="trainer/nll", name="schema", node=NLLConfig)
    cs.store(group="trainer/adversarial", name="schema", node=AdversarialConfig)
    cs.store(group="tracking", name="schema", node=TrackingConfig)
