from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Text, Sequence
import torch
from hydra.core.config_store import ConfigStore

# --- Low-level-configs ---

# TODO: Make compatible with ensemble design
# TODO: Implement the below in the scripts

# Initialize MPSLayer either with hyperparams or tensors, MPS only with tensors.
@dataclass
class MPSInitConfig:
    n_features: Optional[int] = None
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None
    bond_dim: Optional[int] = None
    out_position: Optional[int] = None
    boundary: Text = 'obc'
    tensors: Optional[str] = None # path to where tensors are stored
    n_batches: int = 1
    init_method: Text = 'randn'
    dtype: Optional[str]=None #Optional[torch.dtype] = None

@dataclass
class MPSConfig:
    design: bool # Central tensor or not?
    init_kwargs: MPSInitConfig
    std: float
    embedding: str

@dataclass
class DisConfig:
    hidden_dims: List[int]
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] = None # for leaky relu
    input_dim: Optional[int] = None # derived from dataset usually

@dataclass
class OptimizerConfig:
    name: str # e.g. "adam"
    kwargs: Optional[Dict[str, Any]] # e.g. {"lr": 1e-4, "weight_decay": 0.01}

@dataclass
class CriterionConfig:
    name: str # e.g nlll
    kwargs: Optional[Dict[str, Any]]# e.g. {"eps": 1e-12}

# TODO: Add some interpolations in configs here (mps.phys_dim, dataset.dim) and so on.
@dataclass
class PretrainMPSConfig:
    optimizer_cfg: OptimizerConfig
    criterion_cfg: CriterionConfig
    max_epochs: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    patience: int
    goal_acc: Optional[float] = None
    auto_stack: bool = True
    auto_unbind: bool = False
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

# --- Mid-level config ---
@dataclass
class DataGenDowConfig:
    name: str
    size: Optional[int]
    seed: Optional[int]
    noise: Optional[float]
    circ_factor: Optional[float]
    dow_link: Optional[List[str]]

@dataclass
class DatasetConfig:
    name: str
    gen_dow_kwargs: DataGenDowConfig
    split: Tuple[float, float, float]
    split_seed: int

@dataclass
class ModelConfig:
    mps: MPSConfig
    dis: DisConfig

@dataclass
class PretrainConfig:
    mps: PretrainMPSConfig
    dis: PretrainDisConfig

@dataclass
class GANStyleConfig:
    n_real: int
    n_synth: int
    d_loss: str
    g_loss: str
    d_optim: OptimizerConfig
    g_optim: OptimizerConfig
    max_steps: int
    retrain_crit: str
    stopping_crit: str
    patience: Optional[int]
    smoothing: float = 0.0

#--- Top-level config ---

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    pretrain: PretrainConfig


# --- Register schemas with Hydra ---

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

# Register subgroups so Hydra can override them
cs.store(group="dataset", name="schema", node=DatasetConfig)
cs.store(group="model/mps", name="schema", node=MPSConfig)
# cs.store(group="model/mps", name="init_schema", node=MPSInitConfig)
cs.store(group="model/dis", name="schema", node=DisConfig)
cs.store(group="pretrain/mps", name="schema", node=PretrainMPSConfig)
cs.store(group="pretrain/dis", name="schema", node=PretrainDisConfig)
cs.store(group="ad_train", name="schema", node=GANStyleConfig)
