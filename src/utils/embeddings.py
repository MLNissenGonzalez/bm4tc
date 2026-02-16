
"""
Embeddings of the input for the BornMachines. This is here because...
"""

import torch
import tensorkrowch as tk
from typing import * 


def legendre_embedding(data: torch.Tensor, dim: int = 2, axis: int = -1):
    """
    Compute Legendre polynomial embedding of input data.

    Generates Legendre polynomials P_0 through P_{dim-1} for each element,
    producing exactly ``dim`` components to match the MPS physical dimension.
    This follows the same convention as tensorkrowch's fourier/poly embeddings,
    where the second argument is the output dimension, not the degree.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of arbitrary shape, e.g., (batch_size, n_features).
        Each element is a scalar value in [-1, 1] (typical for Legendre polynomials).
    dim : int, default=2
        Number of embedding components (= max degree + 1).
    axis : int, default=-1
        Axis along which to stack polynomial values. Can be negative.

    Returns
    -------
    torch.Tensor
        Tensor with ``dim`` polynomials stacked along ``axis``.
        Output dtype matches input ``data.dtype``.

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
    for i in range(2, dim):
        p_i = ((2*i-1) * data * polies[-1] - (i-1)*polies[-2]) / i
        polies.append(p_i)
    return torch.stack(polies[:dim], dim=axis)


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
