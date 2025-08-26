from dataclasses import dataclass
from typing import List, Optional, Tuple
from hydra.core.config_store import ConfigStore


# --- Sub-configs ---

@dataclass
class DatasetConfig:
    title: str
    n_feat: int
    n_cls: int
    size: int
    split: Tuple[float, float, float]
    seed: int
    noise: float
    factor: Optional[float]

@dataclass
class MPSConfig:
    phys_dim: int
    bond_dim: int
    phys_dim: int
    boundary: str = "obc"  # "obc" or "pbc"
    design: bool = True # central tensor or not
    out_dim: Optional[int] # dimension of visible leg of central tensor. usually equal to num_cls
    out_position: Optional[int]
    init_method: str
    std: float
    embedding: str

@dataclass
class DisConfig:
    hidden_dims: List[int]
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] # for leaky relu
    input_dim: int

@dataclass
class PretrainMPSConfig:
    optimizer: str
    lr: float
    max_epochs: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    patience: int
    weight_decay: Optional[float]
    auto_stack: bool = True
    auto_unbind: bool = False
    title: str
    goal_acc: Optional[float]
    print_early_stop: bool = True
    print_updates: bool = True

@dataclass
class PretrainDisConfig:
    optimizer: str
    lr: float
    max_epochs: int
    n_real_samples: int # per class or not? per batch
    n_synth_samples: int # n_real_samples + n_synth_samples = batch_size of dataloader.
    loss_fn: str
    patience: int

# TODO: Add adtraining schema and config file for test case.

# --- Top-level config ---

@dataclass
class Config:
    dataset: DatasetConfig
    model_mps: MPSConfig
    model_dis: DisConfig
    pretrain_mps: PretrainMPSConfig
    pretrain_dis: PretrainDisConfig


# --- Register schemas with Hydra ---

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

# Register subgroups so Hydra can override them
cs.store(group="dataset", name="moons_1", node=DatasetConfig)
cs.store(group="model/mps", name="test", node=MPSConfig)
cs.store(group="model/dis", name="test", node=DisConfig)
cs.store(group="pretrain/mps", name="test", node=PretrainMPSConfig)
cs.store(group="pretrain/dis", name="test", node=PretrainDisConfig)
