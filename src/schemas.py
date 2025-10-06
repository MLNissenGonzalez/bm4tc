from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Text, Sequence
from hydra.core.config_store import ConfigStore
import omegaconf
import wandb
import hydra

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
    sampler: str

@dataclass
class DisConfig:
    hidden_multipliers: List[float]
    mode: str
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] = None # for leaky relu

@dataclass
class OptimizerConfig:
    name: str # e.g. "adam"
    kwargs: Optional[Dict[str, Any]] # e.g. {"lr": 1e-4, "weight_decay": 0.01}

@dataclass
class CriterionConfig:
    name: str # e.g nlll
    kwargs: Optional[Dict[str, Any]]# e.g. {"eps": 1e-12}

@dataclass
class PretrainMPSConfig:
    max_epoch: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    stop_crit: str # loss / acc 
    patience: int
    watch_freq: int
    update_freq: int
    toViz: bool
    auto_stack: bool = True
    auto_unbind: bool = False

@dataclass
class PretrainDisConfig:
    max_epoch: int
    batch_size: int
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    info_freq: int
    patience: int
    stop_crit: str # loss / acc 

@dataclass
class WandbConfig:
    entity: str
    project: str
    mode: str

@dataclass
class SaveConfig:
    pre_mps: bool
    pre_dis: bool
    gan_dis: bool
    gan_mps: bool

@dataclass
class RandomConfig:
    seed: int
    random_state: int

@dataclass
class SamplingConfig:
    num_spc: int # total number of samples to be sampled
    num_bins: int # machine accuracy per feature
    batch_spc: int # number of samples to be sampled per batch (important for memory management)

# --- Mid-level config ---
@dataclass
class DataGenDowConfig:
    name: str
    size: Optional[int] # per class? yes. consider renaming to n_spc
    seed: Optional[int]
    noise: Optional[float]
    circ_factor: Optional[float]
    dow_link: Optional[List[str]]

@dataclass
class DatasetConfig:
    name: str
    gen_dow_kwargs: DataGenDowConfig
    split: Tuple[float, float, float]
    split_seed: int # unused / replaced in reproducibilty group

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
    max_epoch: int
    r_real: float # in (0.0, infty). n_real = n_synth * r_synth
    d_criterion: CriterionConfig
    g_criterion: CriterionConfig
    d_optimizer: OptimizerConfig
    g_optimizer: OptimizerConfig
    check_freq: int
    toViz: bool
    info_freq: int
    watch_freq: int
    acc_drop_tol: float
    retrain: PretrainMPSConfig
    smoothing: float = 0.0

@dataclass
class ReproducibilityConfig:
    save: SaveConfig
    random: RandomConfig

#--- Top-level config ---

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    sampling: SamplingConfig
    pretrain: PretrainConfig
    gantrain: GANStyleConfig
    wandb: WandbConfig
    reproduce: ReproducibilityConfig
    experiment: str = "default"

# --- Register schemas with Hydra ---

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

# Register subgroups so Hydra can override them
cs.store(group="dataset", name="schema", node=DatasetConfig)
cs.store(group="model", name="wrapper", node=ModelConfig)
cs.store(group="model/mps", name="schema", node=MPSConfig)
# cs.store(group="model/mps", name="init_schema", node=MPSInitConfig)
cs.store(group="model/dis", name="schema", node=DisConfig)
cs.store(group="pretrain", name="wrapper", node=PretrainConfig)
cs.store(group="pretrain/mps", name="schema", node=PretrainMPSConfig)
cs.store(group="pretrain/dis", name="schema", node=PretrainDisConfig)
cs.store(group="ad_train", name="schema", node=GANStyleConfig)
cs.store(group="wandb", name="schema", node=WandbConfig)
cs.store(group="reproduce", name="schema", node=ReproducibilityConfig)
cs.store(group="reproduce/save", name="schema", node=SaveConfig)
cs.store(group="reproduce/random", name="schema", node=RandomConfig)
cs.store(group="sampling", name="schema", node=SamplingConfig)

