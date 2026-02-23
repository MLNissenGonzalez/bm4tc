
"""
Embeddings of the input for the BornMachines. This is here because...
"""

import math
import torch
from math import sqrt, pi
from typing import *


class FourierEmbedding:
    """Fourier basis embedding of L²[0,1], matching tk.embeddings.fourier."""

    def __init__(self, dim: int):
        self.dim = dim
        # Precompute (op, scale, freq) for i = 1..dim (1-indexed as in tk source)
        self._triples: List[Tuple[str, float, float]] = []
        for i in range(1, dim + 1):
            k = i // 2
            if i == 1:
                self._triples.append(('ones', 1.0, 0.0))
            elif i % 2 == 0:
                self._triples.append(('cos', sqrt(2), 2 * pi * k))
            else:
                self._triples.append(('sin', sqrt(2), 2 * pi * k))

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        components = []
        for op, scale, freq in self._triples:
            if op == 'ones':
                components.append(torch.ones_like(data))
            elif op == 'cos':
                components.append(scale * (freq * data).cos())
            else:
                components.append(scale * (freq * data).sin())
        return torch.stack(components, dim=axis)


class PolyEmbedding:
    """Polynomial (monomial) embedding, matching tk.embeddings.poly."""

    def __init__(self, dim: int):
        self.dim = dim
        # tk.embeddings.poly(data, degree=dim) produces dim+1 components
        self._powers: List[int] = list(range(dim + 1))

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        return torch.stack([data.pow(i) for i in self._powers], dim=axis)


class LegendreEmbedding:
    """
    Legendre polynomial embedding.

    Generates Legendre polynomials P_0 through P_{dim-1} for each element,
    producing exactly ``dim`` components to match the MPS physical dimension.
    """

    def __init__(self, dim: int):
        self.dim = dim
        # Precompute recurrence coefficients alpha[i] = (2i-1)/i, beta[i] = (i-1)/i
        # for i = 2..dim-1
        self._alpha: List[float] = [0.0, 0.0] + [(2*i - 1) / i for i in range(2, dim)]
        self._beta: List[float]  = [0.0, 0.0] + [(i - 1) / i     for i in range(2, dim)]

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        polies = [torch.ones_like(data), data]
        for i in range(2, self.dim):
            p_i = self._alpha[i] * data * polies[-1] - self._beta[i] * polies[-2]
            polies.append(p_i)
        return torch.stack(polies[:self.dim], dim=axis)


class HermiteEmbedding:
    """
    Normalized physicist's Hermite embedding.

    n-th component: f_n(x) = H_n(x) * exp(-x²/2) / sqrt(sqrt(π) * 2^n * n!)
    Satisfies ∫_ℝ f_n(x) f_m(x) dx = δ_{nm}.

    Uses numerically stable "damped recurrence" on g_n(x) = H_n(x) * exp(-x²/2)
    to avoid overflow for large dim.
    """

    def __init__(self, dim: int):
        self.dim = dim
        log_sqrt_pi_quarter = 0.25 * math.log(math.pi)
        log2 = math.log(2.0)
        self._norms: List[float] = [
            math.exp(-0.5 * (log_sqrt_pi_quarter + n * log2 + math.lgamma(n + 1)))
            for n in range(dim)
        ]

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        exp_neg_x2_half = torch.exp(-0.5 * data ** 2)
        g_prev = exp_neg_x2_half                  # g_0
        g_curr = 2.0 * data * exp_neg_x2_half     # g_1
        components = [g_prev * self._norms[0]]
        if self.dim > 1:
            components.append(g_curr * self._norms[1])
        for n in range(2, self.dim):
            g_next = 2.0 * data * g_curr - 2.0 * (n - 1) * g_prev
            components.append(g_next * self._norms[n])
            g_prev, g_curr = g_curr, g_next
        return torch.stack(components, dim=axis)


_EMBEDDING_MAP = {
    "fourier":    FourierEmbedding,
    "poly":       PolyEmbedding,
    "polynomial": PolyEmbedding,
    "legendre":   LegendreEmbedding,
    "hermite":    HermiteEmbedding,
}


def embedding(name: str, dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Instantiate an embedding callable by name and physical dimension.

    Parameters
    ----------
    name : str
        Embedding identifier (case-insensitive). Valid options:
        - "fourier"
        - "poly" or "polynomial"
        - "legendre"
        - "hermite"
    dim : int
        Physical dimension (number of embedding components).

    Returns
    -------
    callable
        Embedding instance with signature ``(data: Tensor) -> Tensor``.

    Raises
    ------
    ValueError
        If the embedding name is not recognized.
    """
    key = name.lower()
    try:
        cls = _EMBEDDING_MAP[key]
    except KeyError:
        raise ValueError(f"Embedding {name} not recognised. "
                         f"Available: {list(_EMBEDDING_MAP.keys())}")
    return cls(dim)


_EMBEDDING_TO_RANGE = {
    "fourier": (0., 1.),
    "legendre": (-1., 1.),
    "hermite": (-4., 4.),
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
