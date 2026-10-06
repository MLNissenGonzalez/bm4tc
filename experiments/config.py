"""Experiment-level configuration: dataclasses and Hydra registration.

Call register() once per process before @hydra.main to register all
structured configs with the Hydra ConfigStore.
"""
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from src.datahandler import DatasetConfig
from src.model import CBMConfig
from src.train.trainer import TrainConfig


@dataclass
class TrackingConfig:
    project: str = "bm4tc"
    entity: str = ""
    mode: str = "disabled"
    seed: int = 42


@dataclass
class Config:
    """Top-level configuration for an experiment."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    born: CBMConfig = field(default_factory=CBMConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment: str = "default"
    descriptor: str = ""
    stage: str = ""
    model_path: Optional[str] = None


def register():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="dataset", name="schema", node=DatasetConfig)
    cs.store(group="model/born", name="schema", node=CBMConfig)
    cs.store(group="trainer", name="schema", node=TrainConfig)
    cs.store(group="tracking", name="schema", node=TrackingConfig)
