"""The run-config schema and its Hydra registration.

Every model and trainer constant is a dataclass default here or in the module that
owns it (D25); ``configs/config.yaml`` only picks group options. Call register()
once per process before composing (bm4tc/pipeline/runs.py does).
"""
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from bm4tc.pipeline.data import DatasetConfig
from bm4tc.core.jem.train import JEMConfig
from bm4tc.core.model import CBMConfig
from bm4tc.core.train import TrainConfig


@dataclass
class TrackingConfig:
    project: str = "bm4tc"
    entity: str = ""
    mode: str = "disabled"
    seed: int = 42  # set per job from the study's seeds


@dataclass
class Config:
    """The configuration of one training run. ``model`` picks the model: ``mps``
    reads ``born``, ``jem`` reads ``jem``; ``trainer`` is shared."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: str = "mps"
    born: CBMConfig = field(default_factory=CBMConfig)
    jem: JEMConfig = field(default_factory=JEMConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def register():
    ConfigStore.instance().store(name="base_config", node=Config)
