from dataclasses import dataclass
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
    name: str  # e.g nlll
    kwargs: Optional[Dict[str, Any]]  # e.g. {"eps": 1e-12}


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
    criterion: str = "crossentropy",
    strengths: List[float] = [0.1, 0.3] # not relative, default for dataspaces [0,1]^n


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
class CriticConfig:
    """
    Configuration for a discriminator neural network.

    Parameters
    ----------
    hidden_multipliers : List[float]
        List of multipliers defining the hidden layer sizes relative to the input dimension.
        For example, `[1.0, 2.0, 1.0]` will scale the input size to determine hidden layers.
    mode : str
        Type of discriminator architecture, e.g., 'mlp', 'cnn', or custom modes.
    nonlinearity : str, default='relu'
        Nonlinear activation function to use between hi
    gantrain: GANStyleConfigdden layers.
        Common options: 'relu', 'leaky_relu', 'tanh', etc.
    negative_slope : Optional[float], default=None
        Slope parameter for leaky ReLU (ignored if nonlinearity is not 'leaky_relu').
    """
    architecture: str
    mode: str
    hidden_multipliers: List[float] | None = None
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] = None  # for leaky relu
    criterion: CriterionConfig

cs.store(group="model/crit", name="schema", node=CriticConfig)

@dataclass
class ModelsConfig:
    """
    Configuration for model components.

    Parameters
    ----------
    mps : MPSConfig
        Configuration for the MPS.
    dis : DisConfig
        Configuration for the discriminator network.
    """
    born: BornMachineConfig
    crit: CriticConfig

cs.store(group="models", name="wrapper", node=ModelsConfig)

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
    auto_stack: bool = True
    auto_unbind: bool = False
    save: bool

cs.store(group="trainer/classification", name="schema", node=ClassificationConfig)

# Inner optimization of DisTrainer with two phases (pretrain and in contest with generator orchestrated by GAN style training config)
@dataclass
class DiscriminationConfig:
    """
    Configuration for pretraining a discriminator network.

    Parameters
    ----------
    max_epoch : int
        Maximum number of epochs for pretraining.
    batch_size : int
        Number of samples per batch.
    optimizer : OptimizerConfig
        Configuration of the optimizer for discriminator pretraining.
    criterion : CriterionConfig
        Loss function configuration for the discriminator.
    info_freq : int
        Frequency (in epochs) at which to print training information.
    patience : int
        Number of epochs without improvement before early stopping.
    stop_crit : str
        Criterion for early stopping: 'loss' or 'acc'.
    """
    max_epoch_pre: int
    max_epoch_gan: int
    batch_size: int
    optimizer: OptimizerConfig
    patience: int
    stop_crit: str  # loss / acc
     
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
    discrimination: DiscriminationConfig
    criterion: CriterionConfig # Kind of interacts with Discriminator, doesn't it.
    sampling: SamplingConfig
    r_real: float  # in (0.0, infty). n_real = n_synth * r_synth
    optimizer: OptimizerConfig
    watch_freq: int
    acc_drop_tol: float
    retrain: ClassificationConfig
    save: bool = False

cs.store(group="trainer/ganstyle", name="schema", node=GANStyleConfig)

@dataclass
class AdversarialConfig:
    max_epoch: int
    batch_size: int
    evasion: EvasionConfig
    save: bool
    # and more.

cs.store(group="trainer/adversarial", name="schema", node=AdversarialConfig)

@dataclass 
class TrainerConfig:
    classification: ClassificationConfig
    ganstyle: GANStyleConfig
    adversarial: AdversarialConfig

cs.store(group="trainer", name="wrapper", node=TrainerConfig)

# --- Utils configs ---

@dataclass
class TrackingConfig:
    project: str
    entity: str
    mode: str
    seed: int
    random_state: int
    metrics: Dict[str, int]
    sampling: SamplingConfig | None = None
    evasion: EvasionConfig | None = None

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
    models: ModelsConfig
    trainer: TrainerConfig
    tracking: TrackingConfig
    experiment: str = "default"

cs.store(name="base_config", node=Config)

