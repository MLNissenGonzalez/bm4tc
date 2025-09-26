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
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    max_epoch: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    patience: int
    auto_stack: bool = True
    auto_unbind: bool = False
    print_early_stop: bool = True
    print_updates: bool = True

@dataclass
class PretrainDisConfig:
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    info_freq: int
    max_epoch: int
    batch_size: int
    patience: int

@dataclass
class WandbSetupConfig:
    entity: str
    project: str

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
    num_bins: int
    n_real: int
    r_synth: float # in (0.0, infty). n_synth = n_real * r_synth
    d_criterion: CriterionConfig
    g_criterion: CriterionConfig
    d_optimizer: OptimizerConfig
    g_optimizer: OptimizerConfig
    max_epoch: int
    stopping_crit: str
    check_freq: int
    info_freq: int
    retrain_crit: str
    acc_drop_tol: float
    retrain_cfg: PretrainMPSConfig
    patience: Optional[int]
    smoothing: float = 0.0

@dataclass
class WandbConfig:
    setup: WandbSetupConfig
    watch: Dict[str, Any]
    gen_viz: int
    isWatch: bool

#--- Top-level config ---

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    pretrain: PretrainConfig
    gantrain: GANStyleConfig
    wandb: WandbConfig


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

# TODO: Move this to src._utils
def init_wandb(cfg: Config):
    # Convert only loggable types
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    
    # Optional: add Hydra job info to group for multiruns
    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    job_num = runtime_cfg.job.get("num", 0)
    
    group_key = f"{cfg.dataset.name}-{cfg.model.dis.mode}"
    run_name = f"j{job_num}-D{cfg.model.mps.init_kwargs.bond_dim}-d{cfg.model.mps.init_kwargs.in_dim}-pre{cfg.pretrain.mps.max_epoch}-gan{cfg.gantrain.max_epoch}"
    run = wandb.init(
        project=cfg.wandb.setup.project,
        entity=cfg.wandb.setup.entity,
        config=wandb_cfg,
        group=group_key,
        name=run_name,
        reinit="finish_previous"
    )
    return run