
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
    Normalized Legendre polynomial embedding, orthonormal on L²[-1,1].

    φ_k(x) = sqrt((2k+1)/2) · P_k(x),  so ∫_{-1}^{1} φ_m φ_n dx = δ_{mn}.

    Generates normalized polynomials φ_0 through φ_{dim-1} for each element,
    producing exactly ``dim`` components to match the MPS physical dimension.

    Note: ∫ P_k² dx = 2/(2k+1), so without the normalization factor the
    higher-order components are systematically under-represented (their
    L²-norm decreases as 1/k), biasing the Born Machine toward low-frequency
    features.
    """

    def __init__(self, dim: int):
        self.dim = dim
        # Recurrence coefficients: P_i = alpha[i]*x*P_{i-1} - beta[i]*P_{i-2}
        self._alpha: List[float] = [0.0, 0.0] + [(2*i - 1) / i for i in range(2, dim)]
        self._beta: List[float]  = [0.0, 0.0] + [(i - 1) / i     for i in range(2, dim)]
        # Orthonormality scales: ||P_k||² = 2/(2k+1)  →  scale = sqrt((2k+1)/2)
        self._norms: List[float] = [sqrt((2*k + 1) / 2) for k in range(dim)]

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        polies = [torch.ones_like(data), data]
        for i in range(2, self.dim):
            p_i = self._alpha[i] * data * polies[-1] - self._beta[i] * polies[-2]
            polies.append(p_i)
        return torch.stack(
            [p * n for p, n in zip(polies[:self.dim], self._norms)], dim=axis
        )


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


class ChebyshevT1Embedding:
    """Chebyshev functions of the first kind, orthonormal on L²[-1,1].
    φ_n(x) = T_n(x) · sqrt(k_n / (π √(1−x²))),  k_0=1, k_n=2 for n≥1.

    IMPORTANT — domain is restricted to (-0.99, 0.99), NOT the full (-1, 1).
    The weight factor (1−x²)^{−1/4} is the square root of the Chebyshev
    measure and diverges at x = ±1.  Restricting the data range to ±0.99
    bounds the weight to at most (1−0.99²)^{−1/4} ≈ 2.24, preventing the
    Born Machine from placing artificially high probability mass at the
    boundaries due to the embedding magnitude alone.  See src/utils/GUIDE.md
    "Chebyshev T1 boundary artefact" for a full explanation.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self._scales: List[float] = [sqrt(1.0 / pi)] + [sqrt(2.0 / pi)] * (dim - 1)

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        # clamp(min=1e-6) is a numerical safety net only; the data range (-0.99, 0.99)
        # already keeps (1-x²) >= 0.0199, so w <= ~2.24 under normal operation.
        w = (1.0 - data ** 2).clamp(min=1e-6).pow(-0.25)
        T_prev = torch.ones_like(data)
        T_curr = data.clone()
        components = [T_prev * self._scales[0] * w]
        if self.dim > 1:
            components.append(T_curr * self._scales[1] * w)
        for n in range(2, self.dim):
            T_next = 2.0 * data * T_curr - T_prev
            components.append(T_next * self._scales[n] * w)
            T_prev, T_curr = T_curr, T_next
        return torch.stack(components, dim=axis)


class ChebyshevT2Embedding:
    """Chebyshev functions of the second kind, orthonormal on L²[-1,1].
    ψ_n(x) = U_n(x) · sqrt(2/π · √(1−x²)).
    """
    def __init__(self, dim: int):
        self.dim = dim
        self._scale: float = sqrt(2.0 / pi)

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        w = (1.0 - data ** 2).clamp(min=0.0).pow(0.25)
        U_prev = torch.ones_like(data)
        U_curr = 2.0 * data
        components = [U_prev * self._scale * w]
        if self.dim > 1:
            components.append(U_curr * self._scale * w)
        for n in range(2, self.dim):
            U_next = 2.0 * data * U_curr - U_prev
            components.append(U_next * self._scale * w)
            U_prev, U_curr = U_curr, U_next
        return torch.stack(components, dim=axis)


class SimpEmbedding:
    """
    Simple 3-component embedding: φ(x) = (1, x, 1-x).

    Maps x ∈ [0, 1] to a 3-dimensional vector. Used as a reference
    baseline matching tutorial MPS softmax classifiers.

    NOTE: This embedding is NOT orthonormal. The three components
    (1, x, 1-x) are linearly dependent (col 0 = col 1 + col 2) and
    not orthogonal under any standard L² inner product. It is provided
    purely as a sanity-check baseline, not for production use.

    The `dim` parameter is accepted for compatibility with the Hydra
    config system (which passes `in_dim` from born config) but is
    ignored — the output always has exactly 3 components.
    """

    def __init__(self, dim: int):
        # dim is absorbed for Hydra compatibility; output is always 3-dimensional.
        pass

    def __call__(self, data: torch.Tensor, axis: int = -1) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError('`data` should be torch.Tensor type')
        return torch.stack([torch.ones_like(data), data, 1.0 - data], dim=axis)


_EMBEDDING_MAP = {
    "fourier":     FourierEmbedding,
    "poly":        PolyEmbedding,
    "polynomial":  PolyEmbedding,
    "legendre":    LegendreEmbedding,
    "hermite":     HermiteEmbedding,
    "chebychev1":  ChebyshevT1Embedding,
    "chebyshev1":  ChebyshevT1Embedding,
    "chebychev2":  ChebyshevT2Embedding,
    "chebyshev2":  ChebyshevT2Embedding,
    "simp":        SimpEmbedding,
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
    "fourier":    (0., 1.),
    "legendre":   (-1., 1.),
    "hermite":    (-4., 4.),
    # (-0.99, 0.99) rather than (-1, 1): the T1 weight (1-x²)^{-1/4} diverges
    # at ±1 (w → ∞), creating a strong implicit boundary prior in the Born
    # Machine.  Restricting to ±0.99 caps w ≈ 2.24 vs ~31.6 at the raw
    # boundary.  T2 has the opposite weight (1-x²)^{+1/4} → 0 at ±1, so the
    # full (-1,1) range is safe for T2 (boundary = zero embedding, not ∞).
    "chebychev1": (-0.99, 0.99),
    "chebyshev1": (-0.99, 0.99),
    "chebychev2": (-1., 1.),
    "chebyshev2": (-1., 1.),
    "simp":       (0., 1.),
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
