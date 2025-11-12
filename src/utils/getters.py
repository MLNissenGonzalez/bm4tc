import torch 
import tensorkrowch as tk
from typing import *
import numpy as np
import src.utils.schemas as schemas
import torch.optim as optim
import torch.nn as nn

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------Embeddings--------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

def legendre_embedding(data: torch.Tensor, degree: int = 2, axis: int = -1):
    """
    Compute Legendre polynomial embedding of input data.

    Generates Legendre polynomials up to the specified degree for each
    element along the given axis, stacking them along the same axis.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of arbitrary shape, e.g., (batch_size, n_features).
        Each element is a scalar value in [-1, 1] (typical for Legendre polynomials).
    degree : int, default=2
        Maximum degree of Legendre polynomials.
    axis : int, default=-1
        Axis along which to stack polynomial values. Can be negative.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(degree+1, ...)` where polynomials are stacked along
        `axis`. Output dtype matches input `data.dtype`.

    Notes
    -----
    Uses the standard recursive formula:
        P_0(x) = 1
        P_1(x) = x
        P_n(x) = ((2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n
    """

    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')

    polies = [1, data]
    for i in range(2, degree + 1):
        p_i = ((2*i-1) * data * polies[-1] - (i-1)*polies[-2]) / i
        polies.append(p_i)
    return torch.stack(polies, dim=axis)


_EMBEDDING_MAP = {
    "fourier": tk.embeddings.fourier,
    "poly": tk.embeddings.poly,
    "polynomial": tk.embeddings.poly,
    "legendre": legendre_embedding
}


def embedding(name: str) -> Callable[[torch.FloatTensor, int], torch.FloatTensor]:
    """
    Retrieve embedding function by name.

    Parameters
    ----------
    name : str
        Embedding identifier (case-insensitive). Valid options:
        - "fourier"
        - "poly" or "polynomial"
        - "legendre"

    Returns
    -------
    callable
        Embedding function that maps a tensor to its embedding.
        Signature: `tensor -> tensor`. E.g., `legendre_embedding`.

    Raises
    ------
    ValueError
        If the embedding name is not recognized.

    Notes
    -----
    The returned function may expect inputs within specific ranges,
    defined in `_EMBEDDING_TO_RANGE`.
    """
    key = name.lower()
    try:
        embedding = _EMBEDDING_MAP[key]
    except KeyError:
        raise ValueError(f"Embedding {name} not recognised. "
                         f"Available: {list(_EMBEDDING_MAP.keys())}")
    return embedding


_EMBEDDING_TO_RANGE = {
    "fourier": (0., 1.),
    "legendre": (-1., 1.)
}


def range_from_embedding(embedding: str):
    """
    Given embedding identifier, return the associated domain of that embedding
    using the `_EMBEDDING_TO_RANGE` dictionary.
    """
    key = embedding.replace(" ", "").lower()
    try:
        rang = _EMBEDDING_TO_RANGE[key]
    except KeyError:
        raise ValueError(f"Embedding {embedding} not recognised. "
                         f"Available: {list(_EMBEDDING_TO_RANGE.keys())}")
    return rang


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#----------Tensor Network properties.-----------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

def indim_and_ncls(mps: tk.models.MPS) -> Tuple[int, int]:
    """
    Extracts `in_dim`, the input embedding dimension, and `n_cls`, the number of classes assuming same input embedding dimension.

    Returns
    -------
    Tuple[int, int]
        (in_dim, n_cls)
    """
    phys = np.array(mps.phys_dim)
    val, counts = np.unique(ar=phys, return_counts=True)
    if np.max(counts) < phys.size-1:
        raise ValueError(
            "Can only handle same dimensional input embedding dimensions.")
    in_dim = val[np.argmax(counts)]
    if len(val) == 2:
        n_cls = val[val != in_dim][0]
    else:
        n_cls = in_dim
    return int(in_dim), int(n_cls)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#----------Optimizers.--------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

_OPTIMIZER_MAP = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adamax": optim.Adamax,
    "nadam": optim.NAdam,
}


def optimizer(params, config: schemas.OptimizerConfig) -> optim.Optimizer:
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

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#----------Critierions.--------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

# Using classes instead of functions in case I want to you use loss functions with more hyperparameters and/or learnable parameters
# TODO: maybe differentiate between different training modes and models?

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


def criterion(cfg: schemas.CriterionConfig) -> nn.Module:
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