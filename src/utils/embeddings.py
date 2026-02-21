
"""
Embeddings of the input for the BornMachines. This is here because...
"""

import torch
import tensorkrowch as tk
from typing import * 


def hermite_embedding(data: torch.Tensor, dim: int = 2, axis: int = -1):
    """
    Compute normalized physicist's Hermite embedding.

    n-th component: f_n(x) = H_n(x) * exp(-x²/2) / sqrt(sqrt(π) * 2^n * n!)
    Satisfies ∫_ℝ f_n(x) f_m(x) dx = δ_{nm}.

    Input range: any real-valued input (pipeline scales to [0, 1] via MinMaxScaler).

    Uses numerically stable "damped recurrence" on g_n(x) = H_n(x) * exp(-x²/2)
    to avoid overflow for large dim.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of arbitrary shape. Values typically in [0, 1] after scaling.
    dim : int, default=2
        Number of embedding components (polynomials 0 through dim-1).
    axis : int, default=-1
        Axis along which to stack components.

    Returns
    -------
    torch.Tensor
        Components stacked along `axis`, same shape as `data` + one new axis.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')

    import math
    # Precompute normalisation constants c_n = sqrt(sqrt(pi) * 2^n * n!) in log space
    log_sqrt_pi_quarter = 0.25 * math.log(math.pi)
    log2 = math.log(2.0)

    # Damped recurrence: g_n = H_n(x) * exp(-x^2/2), stays bounded
    exp_neg_x2_half = torch.exp(-0.5 * data ** 2)
    g_prev = exp_neg_x2_half                  # g_0
    g_curr = 2.0 * data * exp_neg_x2_half     # g_1

    # log_c_n = 0.5 * (0.25*log(pi) + n*log2 + lgamma(n+1))
    def log_c(n):
        return 0.5 * (log_sqrt_pi_quarter + n * log2 + math.lgamma(n + 1))

    components = [g_prev * math.exp(-log_c(0))]
    if dim > 1:
        components.append(g_curr * math.exp(-log_c(1)))

    for n in range(2, dim):
        g_next = 2.0 * data * g_curr - 2.0 * (n - 1) * g_prev
        components.append(g_next * math.exp(-log_c(n)))
        g_prev, g_curr = g_curr, g_next

    return torch.stack(components[:dim], dim=axis)


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
    "legendre": legendre_embedding,
    "hermite": hermite_embedding,
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
        - "hermite"

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
    "legendre": (-1., 1.),
    "hermite": (0., 1.),
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
