import pytest
import torch
from bm4tc.core.embeddings import (
    FourierEmbedding,
    LegendreEmbedding,
    HermiteEmbedding,
    ChebyshevT1Embedding,
    _EMBEDDING_TO_RANGE,
    attack_range_from_embedding,
    embedding,
    range_from_embedding,
)

BATCH = 8
IN_DIM = 4
DATA_DIM = 6


@pytest.fixture
def fourier_emb():
    return FourierEmbedding(IN_DIM)


@pytest.fixture
def x_fourier():
    return torch.rand(BATCH, DATA_DIM)


@pytest.fixture
def x_legendre():
    return torch.rand(BATCH, DATA_DIM) * 2 - 1


@pytest.fixture
def x_hermite():
    return torch.rand(BATCH, DATA_DIM) * 8 - 4


# ---- output shapes ----

def test_fourier_output_shape(x_fourier):
    emb = FourierEmbedding(IN_DIM)
    out = emb(x_fourier)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


def test_legendre_output_shape(x_legendre):
    emb = LegendreEmbedding(IN_DIM)
    out = emb(x_legendre)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


def test_hermite_output_shape(x_hermite):
    emb = HermiteEmbedding(IN_DIM)
    out = emb(x_hermite)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


def test_chebychev1_output_shape():
    emb = ChebyshevT1Embedding(IN_DIM)
    x = torch.rand(BATCH, DATA_DIM) * 1.98 - 0.99
    out = emb(x)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


# ---- Fourier specific ----

def test_fourier_phi0_is_one():
    emb = FourierEmbedding(IN_DIM)
    x = torch.zeros(1)
    out = emb(x)
    assert abs(out.view(-1)[0].item() - 1.0) < 1e-5


def test_fourier_complex_dtype():
    emb = FourierEmbedding(IN_DIM, dtype=torch.complex64)
    x = torch.rand(BATCH, DATA_DIM)
    out = emb(x)
    assert out.dtype == torch.complex64


def test_legendre_float_dtype():
    emb = LegendreEmbedding(IN_DIM, dtype=torch.float32)
    x = torch.rand(BATCH, DATA_DIM) * 2 - 1
    out = emb(x)
    assert out.dtype == torch.float32


# ---- factory function ----

def test_embedding_factory_fourier():
    emb = embedding("fourier", IN_DIM)
    x = torch.rand(BATCH, DATA_DIM)
    out = emb(x)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


def test_embedding_factory_legendre():
    emb = embedding("legendre", IN_DIM)
    x = torch.rand(BATCH, DATA_DIM) * 2 - 1
    out = emb(x)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


def test_embedding_factory_hermite():
    emb = embedding("hermite", IN_DIM)
    x = torch.rand(BATCH, DATA_DIM) * 8 - 4
    out = emb(x)
    assert out.shape == (BATCH, DATA_DIM, IN_DIM)


def test_embedding_factory_unknown_raises():
    with pytest.raises(ValueError):
        embedding("no_such_embedding", IN_DIM)


# ---- range_from_embedding ----

def test_range_from_embedding_fourier():
    lo, hi = range_from_embedding("fourier")
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_range_from_embedding_legendre():
    lo, hi = range_from_embedding("legendre")
    assert lo == pytest.approx(-1.0) and hi == pytest.approx(1.0)


def test_range_from_embedding_hermite():
    lo, hi = range_from_embedding("hermite")
    assert lo == pytest.approx(-4.0) and hi == pytest.approx(4.0)


def test_range_from_embedding_chebychev1():
    lo, hi = range_from_embedding("chebychev1")
    assert lo == pytest.approx(-0.99) and hi == pytest.approx(0.99)


def test_range_from_embedding_chebychev2():
    """T2 is restricted to ±0.99 like T1, for the opposite reason: its weight
    (1-x²)^{1/4} vanishes at ±1, so φ(±1) is the zero vector (the reasons are
    in ``_EMBEDDING_TO_RANGE``)."""
    lo, hi = range_from_embedding("chebychev2")
    assert lo == pytest.approx(-0.99) and hi == pytest.approx(0.99)


# ---- no embedding may vanish inside its own declared domain ----

@pytest.mark.parametrize(
    "name", ["fourier", "cosine", "legendre", "hermite", "chebychev1", "chebychev2"]
)
def test_embedding_never_vanishes_in_its_own_range(name):
    """‖φ(x)‖ > 0 for every x in the declared domain.

    A zero basis vector means |ψ(x,c)|² = 0 for *every* parameter setting — a
    hard zero in the density that no training can move — and it drove the
    accumulate-path contraction into 0/0. Chebyshev T2 declared (-1, 1) and
    vanished exactly at the endpoints, which minmax rescaling maps data onto by
    construction. Guard the whole family, not just T2.
    """
    lo, hi = range_from_embedding(name)
    emb = embedding(name, IN_DIM)
    x = torch.linspace(lo, hi, 2001)
    norms = emb(x).norm(dim=-1)
    assert torch.isfinite(norms).all(), f"{name}: non-finite ‖φ‖ inside its range"
    assert norms.min() > 1e-3, (
        f"{name}: ‖φ‖ drops to {norms.min().item():.3g} at "
        f"x={x[norms.argmin()].item():+.4f}, inside its declared range "
        f"({lo}, {hi})"
    )


# ---- batch independence ----

def test_embedding_batch_independence():
    emb = FourierEmbedding(IN_DIM)
    x = torch.rand(2, 1)
    out_full = emb(x)
    out0 = emb(x[0:1])
    out1 = emb(x[1:2])
    assert torch.allclose(out_full[0], out0[0], atol=1e-5)
    assert torch.allclose(out_full[1], out1[0], atol=1e-5)


def test_embedding_dim_1():
    emb = FourierEmbedding(1)
    x = torch.rand(BATCH, DATA_DIM)
    out = emb(x)
    assert out.shape == (BATCH, DATA_DIM, 1)


def test_embedding_no_nan_inf():
    emb = FourierEmbedding(IN_DIM)
    x = torch.rand(BATCH, DATA_DIM)
    out = emb(x)
    assert torch.isfinite(out).all()


# ---- orthonormality: Z = Σ_c ∫ |ψ|² is the MPS contracted with itself only if ∫ φ φᵀ = I ----

DIMS = [1, 2, 3, 5, 10]   # mnist12_pdim's d and below
QUAD_NODES = 400

# Embeddings orthonormal on a larger domain than their input range: that domain, and the
# largest |Gram − I| on the input range at the dims above (the density is normalised on
# the larger domain, so it puts some mass outside the data's range). Every other embedding
# with a range must be orthonormal on that range.
ORTHONORMAL_ELSEWHERE = {
    "hermite":    ((-12.0, 12.0), 0.25),   # ℝ; [-4, 4] cuts the tails: 2e-1 at dim 10
    "chebychev1": ("(-1, 1)", 0.2),        # ±0.99 cuts the diverging weight: 0.18
    "chebyshev1": ("(-1, 1)", 0.2),
    "chebychev2": ("(-1, 1)", 0.1),        # 0.08 at dim 10
    "chebyshev2": ("(-1, 1)", 0.1),
}


def _gram(name, dim, domain):
    """∫ φ φᵀ over ``domain`` in float64: Gauss-Legendre quadrature on an interval;
    ``"(-1, 1)"`` substitutes x = cos θ and takes the midpoint rule in θ. Both Chebyshev
    integrands then become trigonometric polynomials (cos mθ cos nθ, sin mθ sin nθ), which
    that rule integrates exactly, and no node comes near the embedding's clamp
    (1 − x² ≥ 1e-6) at the ends."""
    import numpy as np
    if domain == "(-1, 1)":
        theta = torch.tensor(np.pi * (np.arange(QUAD_NODES) + 0.5) / QUAD_NODES)
        x, w = theta.cos(), np.pi / QUAD_NODES * theta.sin()
    else:
        t, w = np.polynomial.legendre.leggauss(QUAD_NODES)
        lo, hi = domain
        x, w = torch.tensor((hi - lo) / 2 * t + (hi + lo) / 2), torch.tensor((hi - lo) / 2 * w)
    phi = embedding(name, dim, dtype=torch.float64)(x)
    return (phi * w[:, None]).T @ phi


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", list(_EMBEDDING_TO_RANGE))
def test_embedding_is_orthonormal(name, dim):
    domain = ORTHONORMAL_ELSEWHERE.get(name, (range_from_embedding(name),))[0]
    gram = _gram(name, dim, domain)
    assert torch.allclose(gram, torch.eye(dim, dtype=gram.dtype), atol=1e-10), (
        f"{name} at dim {dim} on {domain}: max |Gram − I| = "
        f"{(gram - torch.eye(dim, dtype=gram.dtype)).abs().max().item():.3g}")


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", list(_EMBEDDING_TO_RANGE))
def test_embedding_gram_on_its_input_range(name, dim):
    tol = ORTHONORMAL_ELSEWHERE.get(name, (None, 1e-10))[1]
    gram = _gram(name, dim, range_from_embedding(name))
    deviation = (gram - torch.eye(dim, dtype=gram.dtype)).abs().max().item()
    assert deviation <= tol, f"{name} at dim {dim}: max |Gram − I| = {deviation:.3g} > {tol}"


# ---- cosine vs Fourier at the ends of [0, 1] ----

def test_cosine_ends_differ_where_fourier_ends_collide():
    """Fourier's period equals its range, so a pixel at 0 and one at 1 embed the same at
    every dim; the cosine basis has period 2, and φ_1 tells the ends apart."""
    ends = torch.tensor([0.0, 1.0])
    fourier, cosine = embedding("fourier", 5)(ends), embedding("cosine", 5)(ends)
    assert torch.allclose(fourier[0], fourier[1], atol=1e-5)
    assert not torch.allclose(cosine[0], cosine[1], atol=1e-2)


def test_cosine_is_flat_at_both_ends():
    """dφ_k/dx = −√2 πk sin(πkx) is 0 at 0 and 1 for every k (Legendre's grows as k²)."""
    x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    phi = embedding("cosine", 10, dtype=torch.float64)(x)
    for k in range(10):
        (grad,) = torch.autograd.grad(phi[:, k].sum(), x, retain_graph=True)
        assert grad.abs().max() < 1e-12


@pytest.mark.parametrize("name", list(_EMBEDDING_TO_RANGE))
def test_attack_range_is_the_input_range_unless_the_embedding_is_flat_at_its_ends(name):
    """Attacks on every other embedding stay bit-identical (D178)."""
    lo, hi = range_from_embedding(name)
    if name == "cosine":
        assert attack_range_from_embedding(name) == pytest.approx((lo + 1e-3, hi - 1e-3))
    else:
        assert attack_range_from_embedding(name) == (lo, hi)
