from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Text, Sequence
import torch
from hydra.core.config_store import ConfigStore
import tensorkrowch as tk

mps = tk.models.MPSLayer()

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

# TODO: Make compatible with ensemble design
# TODO: Implement the below in the scripts
@dataclass
class MPSInitConfig:
    n_features: int | None = None
    in_dim: int | Sequence[int] | None = None
    out_dim: int | None = None
    bond_dim: int | Sequence[int] | None = None
    out_position: int | None = None
    boundary: Text = 'obc'
    tensors: Sequence[torch.Tensor] | None = None
    n_batches: int = 1
    init_method: Text = 'randn'
    dtype: torch.dtype | None = None

@dataclass
class MPSConfig:
    init_kwargs: MPSInitConfig
    design: bool = True # central tensor or not
    std: float
    embedding: str

@dataclass
class DisConfig:
    hidden_dims: List[int]
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] # for leaky relu
    input_dim: int

@dataclass
class OptimizerConfig:
    name: str # e.g. "adam"
    kwargs: Optional[Dict[str, Any]]  # e.g. {"lr": 1e-4, "weight_decay": 0.01}

@dataclass
class CriterionConfig:
    name: str # e.g nlll
    kwargs: Optional[Dict[str, Any]] = None # e.g. {"eps": 1e-12}

# TODO: Add some interpolations in configs here (mps.phys_dim, dataset.dim) and so on.
@dataclass
class PretrainMPSConfig:
    optimizer_cfg: OptimizerConfig
    criterion_cfg: CriterionConfig
    max_epochs: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    patience: int
    auto_stack: bool = True
    auto_unbind: bool = False
    goal_acc: Optional[float]
    print_early_stop: bool = True
    print_updates: bool = True

@dataclass
class PretrainDisConfig:
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    max_epochs: int
    n_real_samples: int # per class or not? per batch
    n_synth_samples: int # n_real_samples + n_synth_samples = batch_size of dataloader.
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
