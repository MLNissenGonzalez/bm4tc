import pytest
import torch
from src.utils.embeddings import (
    FourierEmbedding,
    LegendreEmbedding,
    HermiteEmbedding,
    ChebyshevT1Embedding,
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
