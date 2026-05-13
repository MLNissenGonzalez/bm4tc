from dataclasses import dataclass, field
from typing import *
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()


# --- Shared configs ---

@dataclass
class OptimizerConfig:
    """
    Parameters
    ----------
    name : str
        Optimizer name, e.g., 'adam', 'sgd'.
    kwargs : Optional[Dict[str, Any]]
        Keyword arguments for the optimizer constructor.
    """
    name: str
    kwargs: Optional[Dict[str, Any]]


@dataclass
class CriterionConfig:
    """
    Parameters
    ----------
    name : str
        Loss function identifier, e.g., 'nll'.
    kwargs : Optional[Dict[str, Any]]
        Keyword arguments for the loss constructor.
    """
    name: str
    kwargs: Optional[Dict[str, Any]]


@dataclass
class SamplingConfig:
    """
    Configuration for post-training MPS sampling.

    Parameters
    ----------
    method : str
        Sampling method identifier, e.g., 'multinomial'.
    num_spc : int
        Total number of samples to generate per class.
    num_bins : int
        Number of discrete bins per feature (controls resolution/accuracy).
    batch_spc : int
        Number of samples per batch; controls memory usage.
    """
    method: str
    num_spc: int
    num_bins: int
    batch_spc: int


@dataclass
class EvasionConfig:
    method: str = "FGM"
    norm: int | str = "inf"
    criterion: CriterionConfig = field(default_factory=lambda: CriterionConfig(name="nll", kwargs=None))
    strengths: list = field(default_factory=lambda: [0.1, 0.3])
    num_steps: int = 10
    step_size: float | None = None
    random_start: bool = True


@dataclass
class PurificationConfig:
    """Configuration for likelihood-based purification of adversarial examples.

    Parameters
    ----------
    norm : int | str
        Lp norm for perturbation ball ("inf" or int >= 1).
    num_steps : int
        Number of gradient descent iterations for purification.
    step_size : float | None
        Step size per iteration. If None, defaults to 2.5 * radius / num_steps.
    random_start : bool
        Whether to start from random point within the radius ball.
    radii : list
        List of purification radii to evaluate.
    eps : float
        Clamping floor for numerical stability in log p(x).
    """
    norm: int | str = "inf"
    num_steps: int = 20
    step_size: float | None = None
    random_start: bool = False
    radii: list = field(default_factory=lambda: [0.1, 0.2, 0.3])
    eps: float = 1e-12


# --- Data config ---

@dataclass
class DataGenDowConfig:
    """
    Parameters
    ----------
    name : str
        Name of the dataset or generation method.
    size : Optional[int]
        Number of samples per class.
    seed : Optional[int]
        Random seed for data generation.
    noise : Optional[float]
        Magnitude of added noise.
    circ_factor : Optional[float]
        Circular factor for generating cyclic patterns.
    dow_link : Optional[List[str]]
        Optional download links.
    dow_password : Optional[str]
        Password for protected zip downloads.
    """
    name: str
    size: Optional[int]
    seed: Optional[int]
    noise: Optional[float]
    circ_factor: Optional[float]
    dow_link: Optional[List[str]]
    dow_password: Optional[str] = None


@dataclass
class DatasetConfig:
    """
    Parameters
    ----------
    name : str
        Dataset identifier.
    gen_dow_kwargs : DataGenDowConfig
        Parameters for data generation/download.
    split : Tuple[float, float, float]
        Ratios for train, validation, and test splits (must sum to 1).
    split_seed : int
        Random seed for train/validation/test split.
    overwrite : bool
        If True, regenerate dataset even if it already exists on disk.
    use_ucr_split : bool
        If True, honour original UCR train/test boundary.
    scaler : str
        Scaler to apply: "minmax" or "linear".
    """
    name: str
    gen_dow_kwargs: DataGenDowConfig
    split: Tuple[float, float, float]
    split_seed: int
    overwrite: bool = False
    use_ucr_split: bool = False
    scaler: str = "minmax"


cs.store(group="dataset", name="schema", node=DatasetConfig)


# --- Model configs ---

@dataclass
class MPSInitConfig:
    in_dim: int
    bond_dim: int
    out_position: int | None = None
    boundary: Text = 'obc'
    init_method: Text = 'randn'
    std: float = 1e-9
    n_features: int | None = None
    out_dim: int | None = None
    dtype: Optional[str] = None


@dataclass
class BornMachineConfig:
    """
    Parameters
    ----------
    init_kwargs : MPSInitConfig
        Initialization parameters for the MPS.
    embedding : str
        Embedding type identifier.
    model_path : Optional[str]
        Path to a saved model checkpoint to load.
    """
    init_kwargs: MPSInitConfig
    embedding: str
    model_path: Optional[str] = None


cs.store(group="model/born", name="schema", node=BornMachineConfig)


# --- Trainer configs ---

@dataclass
class DiscriminativeConfig:
    max_epoch: int
    batch_size: int
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    stop_crit: str = "acc"
    patience: int = 250
    watch_freq: int = 0
    save: bool = False
    auto_stack: bool = True
    auto_unbind: bool = False


cs.store(group="trainer/discriminative", name="schema", node=DiscriminativeConfig)


@dataclass
class AdversarialConfig:
    """
    Configuration for adversarial training of the BornMachine classifier.

    Supports two methods:
    - "pgd_at": PGD Adversarial Training (Madry et al.)
    - "trades": TRADES (Zhang et al.) — L(x,y) + beta * KL(p(x) || p(x_adv))

    Parameters
    ----------
    max_epoch : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for training.
    method : str
        Adversarial training method: "pgd_at" or "trades".
    optimizer : OptimizerConfig
        Optimizer configuration for training.
    criterion : CriterionConfig
        Base classification loss function.
    evasion : EvasionConfig
        Attack configuration (method, norm, strengths, etc.).
    stop_crit : str
        Metric to monitor for early stopping: "acc", "dis_loss", "gen_loss", or "rob".
    patience : int
        Number of epochs without improvement before early stopping.
    watch_freq : int
        Frequency of gradient logging. 0 = disabled.
    metrics : Dict[str, int]
        Metrics to evaluate and their frequencies.
    trades_beta : float
        Trade-off parameter for TRADES (ignored for pgd_at). Default 6.0.
    clean_weight : float
        Weight for clean examples in pgd_at (0.0 = pure adversarial). Default 0.0.
    curriculum : bool
        Whether to use curriculum learning over epsilon. Default False.
    curriculum_start : float
        Starting epsilon for curriculum training. Default 0.0.
    curriculum_end_epoch : int | None
        Epoch by which to reach full epsilon. Default None (use max_epoch).
    save : bool
        Whether to save the trained model.
    auto_stack : bool
        tensorkrowch auto_stack option. Default True.
    auto_unbind : bool
        tensorkrowch auto_unbind option. Default False.
    """
    max_epoch: int
    batch_size: int
    method: str
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    evasion: EvasionConfig
    stop_crit: str
    patience: int
    watch_freq: int
    eval_rob_freq: int = 5
    trades_beta: float = 6.0
    clean_weight: float = 0.0
    curriculum: bool = False
    curriculum_start: float = 0.0
    curriculum_end_epoch: int | None = None
    save: bool = False
    auto_stack: bool = True
    auto_unbind: bool = False


cs.store(group="trainer/adversarial", name="schema", node=AdversarialConfig)


@dataclass
class NormControlConfig:
    """
    Unified control for Born Machine norm management during generative training.

    Parameters
    ----------
    target : float | str | None
        Target partition function Z. Three forms accepted:
        - None: capture Z from the pretrained BornMachine at train start.
        - float: use this literal value.
        - str: Python expression evaluated with n_features, data_dim, sqrt, log, exp in scope.
    hard_every : int
        Hard-renormalize toward target every N optimizer steps. 0 = disabled.
    soft_strength : float
        Coefficient for the soft penalty strength * (Z - target)². 0.0 = disabled.
    """
    target: Optional[Union[float, str]] = 1.0
    hard_every: int = 1
    soft_strength: float = 0.0


@dataclass
class GenerativeConfig:
    """
    Configuration for generative training using NLL minimization.
    Trains p(x|c) by minimizing negative log-likelihood.
    """
    max_epoch: int
    batch_size: int
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    stop_crit: str
    patience: int
    watch_freq: int
    norm_control: NormControlConfig
    save: bool = False
    auto_stack: bool = True
    auto_unbind: bool = False


cs.store(group="trainer/generative", name="schema", node=GenerativeConfig)


@dataclass
class TrainerConfig:
    discriminative: DiscriminativeConfig | None = None
    adversarial: AdversarialConfig | None = None
    generative: GenerativeConfig | None = None


cs.store(group="trainer", name="wrapper", node=TrainerConfig)


# --- Tracking config ---

@dataclass
class TrackingConfig:
    project: str
    entity: str
    mode: str  # 'online' | 'offline' | 'disabled'
    seed: int
    evasion: EvasionConfig


cs.store(group="tracking", name="schema", node=TrackingConfig)


@dataclass
class Config:
    """Top-level configuration for an experiment."""
    dataset: DatasetConfig
    born: BornMachineConfig
    trainer: TrainerConfig
    tracking: TrackingConfig
    experiment: str = "default"
    descriptor: str = ""


cs.store(name="base_config", node=Config)
