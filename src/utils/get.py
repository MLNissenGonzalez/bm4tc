import tensorkrowch as tk
from typing import *
import numpy as np
import src.utils.schemas as schemas
import torch.optim as optim


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


from .criterions import criterion
from .embeddings import embedding, range_from_embedding