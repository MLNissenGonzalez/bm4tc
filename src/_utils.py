# utils that are practical for experiments
from schemas import OptimizerConfig
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from typing import Optional, Dict, Any
import schemas
from pathlib import Path
from datetime import datetime
import random
import os
import hydra
import logging
import wandb
import omegaconf
from torch.autograd import Function
import scipy.linalg

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------Criterion-----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Using classes instead of functions in case I want to you use loss functions with more hyperparameters and/or learnable parameters


class MPSNLLL(nn.Module):
    """
    Negative Log-Likelihood Loss (custom implementation for MPS models).

    This loss computes the mean negative log-likelihood of the true class probabilities
    predicted by an MPS classifier. It is equivalent to the categorical cross-entropy
    loss for one-hot targets, but implemented explicitly to ensure numerical stability
    and control over small-value clamping.

    Parameters
    ----------
    eps : float, optional
        Small positive constant used to clamp probabilities from below before
        applying the logarithm to prevent numerical underflow. Default is 1e-12.

    Forward Pass
    -------------
    Given predicted probabilities `p` of shape (batch_size, num_classes) and integer
    targets `t` of shape (batch_size,), the loss is computed as:

        L = -mean( log( p[i, t[i]] ) )

    Returns a scalar tensor representing the average NLL across the batch.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = p.clamp(min=self.eps)
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()


_LOSS_MAP = {
    "nll": MPSNLLL,
    "nlll": MPSNLLL,
    "negativeloglikelihood": MPSNLLL,
    "negloglikelihood": MPSNLLL,
    "bce": nn.BCELoss,  # BCE loss expects probabilities and target is float between 0 and 1
    "binarycrossentropy": nn.BCELoss,
    # TODO: Has to be adapted to the swapped logarithm (ADDED AS ISSUE)
    "vanilla": nn.BCELoss
}


def get_criterion(cfg: schemas.CriterionConfig) -> nn.Module:
    """
    Instantiates a loss function based on the configuration specification.

    The function looks up a registered loss in `_LOSS_MAP` using a normalized
    version of the name provided in `cfg.name`. It supports flexible name
    variants (case- and delimiter-insensitive) and automatically handles
    optional keyword arguments.

    Parameters
    ----------
    cfg : schemas.CriterionConfig
        Configuration object specifying:
        - `name`: name or alias of the desired loss function (e.g. "nll", "bce").
        - `kwargs`: optional dictionary of keyword arguments for initialization.

    Returns
    -------
    nn.Module
        A PyTorch loss module ready to be used for training.

    Raises
    ------
    ValueError
        If the specified loss name does not match any entry in `_LOSS_MAP`.

    Notes
    -----
    - The `MPSNLLL` loss is tailored for MPS-based classifiers with discrete labels.
    - The `BCELoss` is included for binary tasks with probabilistic outputs.
    - Unrecognized loss names will raise an explicit error.
    """
    key = cfg.name.replace(" ", "").replace("-", "").lower()
    if key not in _LOSS_MAP:
        raise ValueError(f"Loss '{cfg.name}' not recognised")
    # use empty dict if kwargs is None
    return _LOSS_MAP[key](**(cfg.kwargs or {}))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------Optimizer-----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


_OPTIMIZER_MAP = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adamax": optim.Adamax,
    "nadam": optim.NAdam,
}


def get_optimizer(params, config: OptimizerConfig) -> optim.Optimizer:
    """
    Select and instantiate a PyTorch optimizer.

    Parameters
    ----------
    params : iterable
        Parameters to optimize, e.g. model.parameters()
    config.name : str
        Name of the optimizer, e.g. "adam"
    config.kwargs : dict, optional
        Extra arguments passed to the optimizer, e.g. {"lr": 1e-3}

    Returns
    -------
    optim.Optimizer
        Instantiated optimizer.
    """
    key = config.name.replace("-", "").replace("_", "").lower()
    try:
        optimizer_cls = _OPTIMIZER_MAP[key]
    except KeyError:
        raise ValueError(f"Optimizer {config.name} not recognised. "
                         f"Available: {list(_OPTIMIZER_MAP.keys())}")

    return optimizer_cls(params, **config.kwargs)


def _class_wise_dataset_size(t: torch.LongTensor, num_cls: int) -> list:
    return torch.bincount(input=t, minlength=num_cls).tolist()


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------wandb tracking setup-----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------


def init_wandb(cfg: schemas.Config) -> wandb.Run:
    """
    Initialize a Weights & Biases (wandb) run with configuration and runtime context.

    This function prepares a structured and reproducible WandB logging environment
    by converting the Hydra/OmegaConf configuration into a flat dict, determining
    runtime information (e.g. run directory, mode, job index), and generating a
    descriptive group and run name for both single and multi-run jobs.

    Behavior
    --------
    - Supports both single and multirun Hydra modes.
    - Constructs descriptive `group` and `name` fields that encode dataset,
      experiment name, date, and key model hyperparameters.
    - Ensures that runs are properly reinitialized when multiple jobs are spawned
      in the same process (via `reinit="finish_previous"`).

    Parameters
    ----------
    cfg : schemas.Config
        Full experiment configuration object (Hydra/OmegaConf structured config).
        Must include the following nested sections:
        - `cfg.experiment` (experiment identifier)
        - `cfg.dataset.name`
        - `cfg.model.mps.init_kwargs` (contains `bond_dim`, `in_dim`)
        - `cfg.pretrain.mps.max_epoch`
        - `cfg.gantrain.max_epoch`
        - `cfg.wandb` (contains `project`, `entity`, `mode`)

    Returns
    -------
    wandb.Run
        The active wandb run instance.

    Raises
    ------
    TypeError
        If any Hydra sweeper parameter group has an unexpected type.
    """

    # Convert only loggable types
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    # Job Info
    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(runtime_cfg.runtime.output_dir)
    job_num = int(runtime_cfg.job.get("num", 0)) + 1

    # Job Mode
    mode = runtime_cfg.mode.value
    total_num = 1
    if mode == 1:  # single run
        now = datetime.now().strftime("%d%b%y_%I%p%M")
    else:  # multirun
        now = datetime.now().strftime("%d%b%y")
        params = runtime_cfg.sweeper.params
        for group in params.values():
            # handle both list and string forms
            if isinstance(group, (list, tuple, omegaconf.ListConfig)):
                options = group
            elif isinstance(group, str):
                options = group.split(",")
            else:
                raise TypeError(
                    f"Unexpected sweeper param type: {type(group)} ({group})")
            total_num *= len(options)

    # Group (folder) and run name
    group_key = f"{cfg.experiment}_{cfg.dataset.name}_{now}"
    run_name = f"job{job_num}/{total_num}_D{cfg.model.mps.init_kwargs.bond_dim}-d{cfg.model.mps.init_kwargs.in_dim}-pre{cfg.pretrain.mps.max_epoch}-gan{cfg.gantrain.max_epoch}"

    # Initializing the wandb object
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=str(run_dir),
        config=wandb_cfg,
        group=group_key,
        name=run_name,
        mode=cfg.wandb.mode,
        reinit="finish_previous"
    )
    return run


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------Model weights save and transfer-----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# General saving function
def save_model(model: torch.nn.Module, run_name: str, model_type: str):
    """
    Save model inside the Hydra run's output directory:
    ${hydra:run.dir}/models/{model_type}_{run_name}.pt
    """
    assert model_type in ["pre_mps", "pre_dis", "gan_mps", "gan_dis"], \
        f"Invalid model_type {model_type}"

    # Hydra's current run dir
    run_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Models subfolder inside it
    folder = run_dir / "models"
    folder.mkdir(parents=True, exist_ok=True)

    # Save path
    filename = f"{model_type}_{run_name}.pt"
    save_path = folder / filename

    # Save state dict
    torch.save(model.state_dict(), save_path)

    return str(save_path)

# Verification of model transfer


def verify_tensors(model1, model2, name1="Original", name2="Copy"):
    """
    Verify correct transfer of tensor parameters between two MPS models.

    This utility compares the internal tensors of two `tk.models.MPS` objects
    element-wise to ensure they are numerically identical (within floating-point
    tolerance). It logs detailed information about mismatches for debugging
    purposes, and reports successful transfers otherwise.

    Parameters
    ----------
    model1 : tk.models.MPS
        Reference (source) model whose tensors serve as the ground truth.
    model2 : tk.models.MPS
        Target (copied) model whose tensors are checked for equality.
    name1 : str, optional
        Display name for the first model in log output (default: "Original").
    name2 : str, optional
        Display name for the second model in log output (default: "Copy").

    Logging
    -------
    - Logs `"Verifying tensor transfer..."` at the start.
    - Logs an `ERROR` for each tensor that differs between the two models,
      including the index and maximum absolute difference.
    - Logs an `INFO` message for each successfully transferred tensor.

    Notes
    -----
    - Uses `torch.allclose` for element-wise comparison with default tolerances.
    - Intended primarily for post-checks after weight transfer or checkpoint reload.

    Returns
    -------
    None
    """
    logger.info("Verifying tensor transfer...")
    for i, (t1, t2) in enumerate(zip(model1.tensors, model2.tensors)):
        if not torch.allclose(t1, t2):
            logger.error(f"Tensor {i} mismatch between {name1} and {name2}")
            logger.error(f"Max difference: {(t1 - t2).abs().max().item()}")
        else:
            logger.info(f"Tensor {i} successfully transferred")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------VISUALISATIONS-----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: Implement this also for other visualisable datatypes (ADDED AS ISSUE)
def visualise_samples(samples: torch.FloatTensor, labels: Optional[torch.LongTensor] = None, gen_viz: Optional[int] = None):
    """
    Visualise real or synthetised samples. 
    If t==None, then samples are synthesised and 
    expected to be of the shape (n, num classes, data dim), 
    else data is real and of shape (N, data dim).
    gen_viz tells us how many samples should be visualised for higher dimensional cases (MNIST).

    Returns
    -------
    ax
        axis object of matplotlib (either image or scatter plot)
    """
    if labels is None:
        n, num_classes, data_dim = samples.shape
        # (n*num_classes, data_dim)
        samples = samples.reshape(n*num_classes, data_dim)
        labels = torch.arange(num_classes).repeat(n)    # (n*num_classes,)

    if samples.shape[1] == 2:
        return create_2d_scatter(X=samples, t=labels)
    else:
        if gen_viz is None:
            # Can be used to visualise only a limited amount of examples
            gen_viz = samples.shape[0]
        raise ValueError("Higher data dimension not yet implemented.")


def create_2d_scatter(
    X: torch.FloatTensor,
    t: torch.LongTensor,
    title=None,
    ax=None,
    show_legend=True
):
    """
    Create a 2D scatter plot that handles both numpy arrays and torch tensors
    and can be embedded in larger figures.
    """
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    classes = np.unique(t)
    for cls in classes:
        idx = (t == cls)
        ax.scatter(X[idx, 0], X[idx, 1], s=5, label=f'Class {cls}')

    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.grid(True)
    if show_legend:
        ax.legend(title="Class")
    return ax

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------Random Seed -----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducible results.

    This function ensures that experiments are deterministic to the extent
    possible, including behavior on GPU and multi-GPU setups.

    Parameters
    ----------
    seed : int
        Integer value used to seed all random number generators.

    Notes
    -----
    - Seeds Python's `random` module, NumPy, and PyTorch (CPU and GPU).
    - For PyTorch, also sets `torch.backends.cudnn.deterministic=True` to
      enforce deterministic algorithms in cuDNN.
    - Disables `torch.backends.cudnn.benchmark` to avoid non-deterministic
      optimizations.
    - Sets the `PYTHONHASHSEED` environment variable for hash-based operations.
    - May reduce performance due to disabling some GPU optimizations.

    Examples
    --------
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior in cuDNN (can slow things down!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------Metrics for generative capabilities -----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Matrix square root implementation for pytorch,
# from https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py ---

class MatrixSquareRoot(Function):
    """
    Custom PyTorch autograd function to compute the matrix square root.

    Computes X = sqrt(A) such that X @ X ≈ A. Backpropagation uses the
    Sylvester equation to compute the gradient w.r.t. the input matrix.

    Notes
    -----
    - Forward pass uses `scipy.linalg.sqrtm`.
    - Backward pass solves the Sylvester equation: sqrtm @ dX + dX @ sqrtm = d(sqrtm).
    - Inputs are assumed to be square matrices.
    - Input and output tensors are on the same device and dtype as the input.

    Usage
    -----
    >>> X = torch.randn(3, 3)
    >>> sqrtX = MatrixSquareRoot.apply(X)
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float64)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.detach().cpu().numpy().astype(np.float64)
            gm = grad_output.detach().cpu().numpy().astype(np.float64)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def mean_n_cov(data: torch.FloatTensor):
    """
    Compute the mean vector and covariance matrix of a dataset.

    Parameters
    ----------
    data : torch.FloatTensor
        Tensor of shape (N, d) where N is the number of samples and d the feature dimension.

    Returns
    -------
    mu : torch.Tensor
        Mean vector of shape (d,).
    cov : torch.Tensor
        Covariance matrix of shape (d, d), computed as torch.cov(data.T).
    """
    mu = data.mean(dim=0)
    cov = torch.cov(data.T)
    return mu, cov

# --- FID-like metric ---


class FIDLike(nn.Module):
    """
    FID-like metric for evaluating the quality of generated samples.

    Measures the similarity between the distributions of real and generated samples
    using a Gaussian approximation: differences in mean and covariance, including
    a matrix square root term. Similar to the Fréchet Inception Distance (FID) in spirit.

    Parameters
    ----------
    eps : float
        Small regularization constant added to the diagonal of covariance matrices
        for numerical stability.

    Methods
    -------
    lazy_forward(mu_r, cov_r, generated)
        Computes the FID-like score given precomputed real mean/covariance and generated samples.
    forward(real, generated)
        Computes the FID-like score given raw real and generated samples.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def lazy_forward(self, mu_r, cov_r, generated):
        mu_g, cov_g = mean_n_cov(generated)

        # Regularize for numerical stability
        eye = torch.eye(cov_r.shape[0], device=cov_r.device)
        cov_r = cov_r + self.eps * eye
        cov_g = cov_g + self.eps * eye

        # Mean difference
        diff_mu = torch.sum((mu_r - mu_g) ** 2)

        # Matrix square root term using custom function
        covmean = sqrtm(cov_r @ cov_g)

        # Ensure real part (numerical precision can introduce tiny imaginary parts)
        if torch.is_complex(covmean):
            covmean = covmean.real

        diff_cov = torch.trace(cov_r + cov_g - 2 * covmean)

        return diff_mu + diff_cov

    def forward(self, real, generated):
        mu_r, cov_r = mean_n_cov(real)
        return self.lazy_forward(mu_r, cov_r, generated)


def sample_quality_control(synths: torch.FloatTensor,
                           upper: float, lower: float):
    """
    Inspect generated samples for out-of-bound values.

    Parameters
    ----------
    synths : torch.FloatTensor
        Tensor of generated samples, shape (num_samples, num_classes, num_features).
    upper : float
        Upper bound for acceptable sample values.
    lower : float
        Lower bound for acceptable sample values.

    Notes
    -----
    - Logs the number of "bad" positions where values exceed bounds.
    - Reports the first 200 offending indices and values.
    - Logs per-class and per-feature maximum absolute value and mean.
    - Useful for debugging numerical instabilities in generative models.
    """
    bad_idx = (
        (synths.abs() > upper) | (synths < lower)
    ).nonzero(as_tuple=False)  # (sample_idx, class_idx, feat_idx)
    logger.info(f"bad positions count = {bad_idx.shape[0]}")
    if bad_idx.shape[0] > 0:
        # show first few offending indices and their values
        for i in range(min(200, bad_idx.shape[0])):
            s, c, f = bad_idx[i].tolist()
            val = synths[s, c, f].item()
            logger.info(f"BAD value at sample={s}, class={c}, feat={f}: {val}")
        # show global per-dim & per-class stats
        logger.info("per-class-per-dim max abs:")
        for c in range(synths.shape[1]):
            for f in range(synths.shape[2]):
                m = synths[:, c, f].abs().max().item()
                logger.info(
                    f" class={c}, feat={f}, max_abs={m:.4g}, mean={synths[:,c,f].mean().item():.4g}")
    else:
        logger.info("No values above threshold found.")
