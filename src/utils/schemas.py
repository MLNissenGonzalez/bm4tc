from dataclasses import dataclass, field
from typing import *
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()

# --- Shared configs---

@dataclass
class OptimizerConfig:
    """
    Configuration for an optimizer.

    Parameters
    ----------
    name : str
        Optimizer name, e.g., 'adam', 'sgd'.
    kwargs : Optional[Dict[str, Any]]
        Keyword arguments for the optimizer constructor, e.g., `{"lr": 1e-4, "weight_decay": 0.01}`.
    """
    name: str  # e.g. "adam"
    kwargs: Optional[Dict[str, Any]]  # e.g. {"lr": 1e-4, "weight_decay": 0.01}


@dataclass
class CriterionConfig:
    """
    Configuration for a loss function.

    Parameters
    ----------
    name : str
        Loss function identifier, e.g., 'nll', 'bce'.
    kwargs : Optional[Dict[str, Any]]
        Keyword arguments for the loss constructor, e.g., `{"eps": 1e-12}` for numerical stability.
    """
    name: str  # e.g nlll, wgan
    kwargs: Optional[Dict[str, Any]]  # e.g. {"eps": 1e-12, "lamb": 10.0, "swapped": False}


@dataclass
class SamplingConfig:
    """
    Configuration for MPS or GAN-style sampling.

    Parameters
    ----------
    method : str
        Sampling method identifier, e.g., 'secant' or 'spline'.
    num_spc : int
        Total number of samples to generate per class.
    num_bins : int
        Number of discrete bins per feature (controls resolution/accuracy).
    batch_spc : int
        Number of samples per batch to generate, useful for memory management.
    """
    method: str
    num_spc: int  # total number of samples to be sampled
    num_bins: int  # machine accuracy per feature
    # number of samples to be sampled per batch (important for memory management)
    batch_spc: int

@dataclass
class EvasionConfig:
    method: str = "FGM",
    norm: int | str = "inf",
    criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None)
    strengths: list = field(default_factory=lambda: [0.1, 0.3]) # not relative, default for dataspaces [0,1]^n
    # PGD-specific parameters (ignored for FGM)
    num_steps: int = 10
    step_size: float | None = None  # defaults to 2.5 * strength / num_steps if None
    random_start: bool = True


# --- Data config ---
@dataclass
class DataGenDowConfig:
    """
    Configuration for downloading or generating dataset.

    Parameters
    ----------
    name : str
        Name of the dataset or generation method.
    size : Optional[int]
        Number of samples per class (consider renaming to n_spc for clarity).
    seed : Optional[int]
        Random seed for data generation.
    noise : Optional[float]
        Magnitude of added noise in the dataset.
    circ_factor : Optional[float]
        Circular factor for generating cyclic patterns in data (if relevant).
    dow_link : Optional[List[str]]
        Optional list download links.
    """
    name: str
    size: Optional[int]  # per class? yes. consider renaming to n_spc
    seed: Optional[int]
    noise: Optional[float]
    circ_factor: Optional[float]
    dow_link: Optional[List[str]]

@dataclass
class DatasetConfig:
    """
    High-level dataset configuration.

    Parameters
    ----------
    name : str
        Dataset identifier.
    gen_dow_kwargs : DataGenDowConfig
        Parameters for data generation/download.
    split : Tuple[float, float, float]
        Ratios for train, validation, and test splits (must sum to 1).
    """
    name: str
    gen_dow_kwargs: DataGenDowConfig
    split: Tuple[float, float, float]
    split_seed: int

cs.store(group="dataset", name="schema", node=DatasetConfig)

# --- Model configs ---
# TODO: Add more documentation for the Configs (ADDED AS ISSUE)
# TODO: Make compatible with ensemble design (ADDED AS ISSUE)

@dataclass
class MPSInitConfig:
    # TODO: Add documentation (ADDED AS ISSUE)
    in_dim: int
    bond_dim: int
    out_position: int | None = None # dynamically assigned, if None in Config, tries to find middle
    boundary: Text = 'obc'
    init_method: Text = 'randn'
    std: float = 1e-9
    n_features: int | None = None # dynamically assigned, depends on dataset.
    out_dim: int | None = None # dynamically assigned, depends on dataset.
    dtype: Optional[str] = None  # Optional[torch.dtype] = None


@dataclass
class BornMachineConfig:
    """
    Configuration for a full MPS model, including design choices, initialization, and embedding.

    Parameters
    ----------
    init_kwargs : MPSInitConfig
        Initialization parameters for the MPS (see `MPSInitConfig`).
    embedding : str
        Identifier for the embedding type used to map input values to physical dimensions.
    """
    init_kwargs: MPSInitConfig
    embedding: str
    model_path: Optional[str] = None # Where the model is stored. 

cs.store(group="model/born", name="schema", node=BornMachineConfig)

@dataclass
class MLPModelKwargs:
    hidden_multipliers: List[float] 
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] = None  # for leaky relu

@dataclass
class ConvModelKwargs:
    placeholder: str

# maybe try also autoencoder and other feature extraction things.
@dataclass 
class BackBoneConfig:
    architecture: str # see above for examples
    model_kwargs: dict = field(default_factory=lambda: {})

@dataclass 
class HeadConfig:
    class_aware: bool # of the classes
    architecture: str # see above for examples
    model_kwargs: dict = field(default_factory=lambda: {})

# Inner optimization of DisTrainer with two phases (pretrain and in contest with generator orchestrated by GAN style training config)
@dataclass
class DiscriminationConfig:
    max_epoch_pre: int
    max_epoch_gan: int
    batch_size: int
    optimizer: OptimizerConfig
    patience: int

@dataclass
class CriticConfig:
    """
    Contains architectual design of the backbone and class heads. 

    Parameters
    ----------
    architecture: str
        name of the architecture
    model_kwargs: dict
        dictionary of model kwargs. see above for some examples
    criterion: Criterion
        the type of criterion/loss function used, which implies which distance is minimised implicetly
    """
    backbone: BackBoneConfig # encoder
    head: HeadConfig # decoder
    discrimination: DiscriminationConfig
    criterion: CriterionConfig # wgan, bce

cs.store(group="trainer/ganstyle/critic", name="schema", node=CriticConfig)
# --- Trainer configs ---  

@dataclass 
class ClassificationConfig:
    max_epoch: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    optimizer: OptimizerConfig
    criterion: CriterionConfig 
    stop_crit: str  # loss / acc
    patience: int
    watch_freq: int
    metrics: Dict[str, int] # to eval, values give evaluation frequency of given metric
    save: bool
    auto_stack: bool = True
    auto_unbind: bool = False
    

cs.store(group="trainer/classification", name="schema", node=ClassificationConfig)
     
@dataclass
class GANStyleConfig:
    """
    Configuration for GAN-style training of the MPS generator and discriminator.

    Parameters
    ----------
    max_epoch : int
        Maximum number of GAN training epochs.
    r_real : float
        Ratio of real samples to generated samples per batch (n_real = n_synth * r_real).
    d_criterion : CriterionConfig
        Loss function configuration for the discriminator.
    g_criterion : CriterionConfig
        Loss function configuration for the generator.
    d_optimizer : OptimizerConfig
        Optimizer configuration for the discriminator.
    g_optimizer : OptimizerConfig
        Optimizer configuration for the generator.
    check_freq : int
        Frequency (in epochs) to check classification performance and potentially retrain.
    toViz : bool
        Whether to visualize generated samples during training.
    info_freq : int
        Frequency (in epochs) to log progress information.
    watch_freq : int
        Step interval for gradient logging and monitoring.
    acc_drop_tol : float
        Accuracy drop tolerance; triggers retraining if validation accuracy falls below (best_acc - acc_drop_tol).
    retrain : PretrainMPSConfig
        Pretraining configuration used for retraining generator when needed.
    smoothing : float, default=0.0
        Optional label smoothing applied to targets for the generator/discriminator losses.
    """
    max_epoch: int
    critic: CriticConfig
    sampling: SamplingConfig
    r_real: float  # in (0.0, infty). n_real = n_synth * r_synth
    optimizer: OptimizerConfig
    watch_freq: int
    metrics: Dict[str, int] # to eval, values give evaluation frequency of given metric
    retrain_crit: str # "acc" or "loss"
    tolerance: float # retrain for acc, if: tolerance < (goal_acc - current_acc), retrain for loss, if: tolerance < (current_loss - goal_loss) / goal_loss     
    retrain: ClassificationConfig
    save: bool = False

cs.store(group="trainer/gantrain", name="schema", node=GANStyleConfig)

@dataclass
class AdversarialConfig:
    """
    Configuration for adversarial training of the BornMachine classifier.

    Supports two methods:
    - "pgd_at": PGD Adversarial Training (Madry et al.) - trains on adversarial examples
    - "trades": TRADES (Zhang et al.) - L(x,y) + beta * KL(p(x) || p(x_adv))

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
        Metric to monitor for early stopping: "loss", "acc", or "rob".
    patience : int
        Number of epochs without improvement before early stopping.
    watch_freq : int
        Frequency of gradient logging to W&B.
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
    method: str  # "pgd_at" or "trades"
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    evasion: EvasionConfig
    stop_crit: str  # "loss", "acc", or "rob"
    patience: int
    watch_freq: int
    metrics: Dict[str, int]
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
class GenerativeConfig:
    """
    Configuration for generative training using NLL minimization.

    Trains p(x|c) by minimizing negative log-likelihood.
    User must implement the criterion with their normalization approach.
    """
    max_epoch: int
    batch_size: int
    optimizer: OptimizerConfig
    criterion: CriterionConfig  # generative NLL criterion (user implements)
    stop_crit: str  # "loss" or "fid"
    patience: int
    watch_freq: int
    metrics: Dict[str, int]  # {"loss": 1, "fid": 10, "viz": 10}
    save: bool = False
    auto_stack: bool = True
    auto_unbind: bool = False

cs.store(group="trainer/generative", name="schema", node=GenerativeConfig)


@dataclass
class TrainerConfig:
    classification: ClassificationConfig | None = None
    ganstyle: GANStyleConfig | None = None
    adversarial: AdversarialConfig | None = None
    generative: GenerativeConfig | None = None

cs.store(group="trainer", name="wrapper", node=TrainerConfig)

# --- Utils configs ---

@dataclass
class TrackingConfig:
    project: str
    entity: str
    mode: str # Literal['online', 'offline', 'disabled', 'shared'] | None
    seed: int
    random_state: int
    sampling: SamplingConfig
    evasion: EvasionConfig

cs.store(group="tracking", name="schema", node=TrackingConfig)

@dataclass
class Config:
    """
    Top-level configuration integrating dataset, model, sampling, training, logging, and reproducibility.

    Parameters
    ----------
    dataset : DatasetConfig
        Configuration of the dataset and data generation.
    model : ModelConfig
        Configuration for generator (MPS) and discriminator networks.
    sampling : SamplingConfig
        Configuration of the sampling method, batch size, and resolution.
    pretrain : PretrainConfig
        Pretraining parameters for generator and discriminator.
    gantrain : GANStyleConfig
        GAN-style training configuration.
    wandb : WandbConfig
        Weights & Biases logging configuration.
    reproduce : ReproducibilityConfig
        Experiment reproducibility and save configuration.
    experiment : str, default='default'
        Name of the experiment.
    """
    dataset: DatasetConfig
    born: BornMachineConfig
    trainer: TrainerConfig
    tracking: TrackingConfig
    experiment: str = "default"

cs.store(name="base_config", node=Config)

