import pytest
import torch
from src.utils.purification.minimal import LikelihoodPurification, normalizing

BATCH = 8
DIM = 6


@pytest.fixture
def delta():
    return torch.randn(BATCH, DIM)


# ---- normalizing ----

def test_normalizing_linf_is_sign(delta):
    out = normalizing(delta, "inf")
    signs = delta.sign()
    assert torch.allclose(out, signs)


def test_normalizing_l2_unit_vector(delta):
    out = normalizing(delta, 2)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


def test_normalizing_l2_zero_clamped():
    x = torch.zeros(BATCH, DIM)
    out = normalizing(x, 2)
    assert torch.isfinite(out).all()


def test_normalizing_invalid_norm_raises(delta):
    with pytest.raises(ValueError):
        normalizing(delta, 0)


# ---- LikelihoodPurification._project ----

def test_project_linf_clamps(delta):
    eps = 0.3
    purif = LikelihoodPurification(norm="inf")
    projected = purif._project(delta, eps)
    assert (projected >= -eps - 1e-6).all()
    assert (projected <= eps + 1e-6).all()


def test_project_l2_bounded(delta):
    eps = 0.5
    purif = LikelihoodPurification(norm=2)
    projected = purif._project(delta, eps)
    norms = projected.norm(p=2, dim=1)
    assert (norms <= eps + 1e-5).all()


def test_project_zero_unchanged():
    purif = LikelihoodPurification(norm="inf")
    zero = torch.zeros(BATCH, DIM)
    projected = purif._project(zero, radius=0.5)
    assert torch.allclose(projected, zero)


# ---- LikelihoodPurification._random_init ----

def test_random_init_shape():
    purif = LikelihoodPurification(norm="inf")
    init = purif._random_init((BATCH, DIM), radius=0.2, device=torch.device("cpu"))
    assert init.shape == (BATCH, DIM)


def test_random_init_within_l2_ball():
    purif = LikelihoodPurification(norm=2)
    radius = 0.5
    init = purif._random_init((BATCH, DIM), radius=radius, device=torch.device("cpu"))
    norms = init.norm(p=2, dim=1)
    assert (norms <= radius + 1e-5).all()


def test_random_init_linf_bounded():
    purif = LikelihoodPurification(norm="inf")
    radius = 0.2
    init = purif._random_init((BATCH, DIM), radius=radius, device=torch.device("cpu"))
    assert (init.abs() <= radius + 1e-6).all()
