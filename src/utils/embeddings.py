
"""
Embeddings of the input for the BornMachines. This is here because...
"""

import torch
import tensorkrowch as tk
from typing import * 


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

    polies = [torch.ones_like(data), data]
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
