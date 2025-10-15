from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Text, Sequence
from hydra.core.config_store import ConfigStore

# TODO: Add more documentation for the Configs (ADDED AS ISSUE)

# --- Low-level-configs ---

# TODO: Make compatible with ensemble design (ADDED AS ISSUE)
# Initialize MPSLayer either with hyperparams or tensors, MPS only with tensors.


@dataclass
class MPSInitConfig:
    # TODO: Add documentation (ADDED AS ISSUE)
    n_features: Optional[int] = None
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None
    bond_dim: Optional[int] = None
    out_position: Optional[int] = None
    boundary: Text = 'obc'
    tensors: Optional[str] = None  # path to where tensors are stored
    n_batches: int = 1
    init_method: Text = 'randn'
    dtype: Optional[str] = None  # Optional[torch.dtype] = None


@dataclass
class MPSConfig:
    """
    Configuration for a full MPS model, including design choices, initialization, and embedding.

    Parameters
    ----------
    design : bool
        Indicates whether the MPS includes a central tensor (True) or is an ensemble (False).
    init_kwargs : MPSInitConfig
        Initialization parameters for the MPS (see `MPSInitConfig`).
    std : float
        Standard deviation used for weight initialization (if applicable).
    embedding : str
        Identifier for the embedding type used to map input values to physical dimensions.
    """
    design: bool
    init_kwargs: MPSInitConfig
    std: float
    embedding: str


@dataclass
class DisConfig:
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
        Nonlinear activation function to use between hidden layers.
        Common options: 'relu', 'leaky_relu', 'tanh', etc.
    negative_slope : Optional[float], default=None
        Slope parameter for leaky ReLU (ignored if nonlinearity is not 'leaky_relu').
    """
    hidden_multipliers: List[float]
    mode: str
    nonlinearity: str = "relu"  # could also use Literal if you want stricter typing
    negative_slope: Optional[float] = None  # for leaky relu


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
class PretrainMPSConfig:
    """
    Configuration for pretraining an MPS model as a classifier.

    Parameters
    ----------
    max_epoch : int
        Maximum number of epochs for pretraining.
    batch_size : int
        Number of samples loaded per categorization step across all classes.
    optimizer : OptimizerConfig
        Configuration of the optimizer used for pretraining.
    criterion : CriterionConfig
        Configuration of the loss function used for pretraining.
    stop_crit : str
        Criterion for early stopping: 'loss' or 'acc'.
    patience : int
        Number of epochs to wait without improvement before early stopping.
    watch_freq : int
        Frequency (in steps) at which to log gradient histograms.
    update_freq : int
        Frequency (in steps) at which optimizer updates are applied.
    toViz : bool
        Whether to visualize generated samples during pretraining.
    auto_stack : bool, default=True
        See tensorkrowch documentation.
    auto_unbind : bool, default=False
        See tensorkrowch documentation.
    """
    max_epoch: int
    batch_size: int  # samples loaded per categorisation step for all classes involved
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    stop_crit: str  # loss / acc
    patience: int
    watch_freq: int
    update_freq: int
    toViz: bool
    auto_stack: bool = True
    auto_unbind: bool = False


@dataclass
class PretrainDisConfig:
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
    max_epoch: int
    batch_size: int
    optimizer: OptimizerConfig
    criterion: CriterionConfig
    info_freq: int
    patience: int
    stop_crit: str  # loss / acc


@dataclass
class WandbConfig:
    """
    Configuration for Weights & Biases (wandb) logging.

    Parameters
    ----------
    entity : str
        Wandb entity (team or user) under which the project will be logged.
    project : str
        Name of the wandb project.
    mode : str
        Wandb mode, e.g., 'online', 'offline', 'disabled'.
    """
    entity: str
    project: str
    mode: str


@dataclass
class SaveConfig:
    """
    Flags for saving trained models.

    Parameters
    ----------
    pre_mps : bool
        Whether to save the pretrained MPS model.
    pre_dis : bool
        Whether to save the pretrained discriminator(s).
    gan_dis : bool
        Whether to save the GAN discriminator(s) after training.
    gan_mps : bool
        Whether to save the GAN generator (MPS) after training.
    """
    pre_mps: bool
    pre_dis: bool
    gan_dis: bool
    gan_mps: bool


@dataclass
class RandomConfig:
    """
    Configuration for random seeds to ensure reproducibility.

    Parameters
    ----------
    seed : int
        Global seed for Python, NumPy, and PyTorch.
    random_state : int
        Additional integer to initialize pseudo-random generators, e.g., for dataset splitting.
    """
    seed: int
    random_state: int


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

# --- Mid-level config ---


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


@dataclass
class ModelConfig:
    """
    Configuration for model components.

    Parameters
    ----------
    mps : MPSConfig
        Configuration for the MPS.
    dis : DisConfig
        Configuration for the discriminator network.
    """
    mps: MPSConfig
    dis: DisConfig


@dataclass
class PretrainConfig:
    """
    Configuration for pretraining both MPS classifier and discriminator.

    Parameters
    ----------
    mps : PretrainMPSConfig
        Pretraining parameters for the MPS classifier.
    dis : PretrainDisConfig
        Pretraining parameters for the discriminator network.
    """
    mps: PretrainMPSConfig
    dis: PretrainDisConfig

# TODO: Move Smoothing to GAN criterion config


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
    r_real: float  # in (0.0, infty). n_real = n_synth * r_synth
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
    """
    Configuration for experiment reproducibility and saving.

    Parameters
    ----------
    save : SaveConfig
        Flags controlling which model components and training states to save.
    random : RandomConfig
        Random seed configuration to ensure deterministic behavior.
    """
    save: SaveConfig
    random: RandomConfig

# --- Top-level config ---


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
