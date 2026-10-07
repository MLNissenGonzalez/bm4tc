import pytest
import torch
from bm4tc.core.attacks import normalizing, project, random_in_ball

CPU = torch.device("cpu")

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


# ---- project (attacks and purification share it) ----

def test_project_linf_clamps(delta):
    eps = 0.3
    norm = "inf"
    projected = project(delta, norm, eps)
    assert (projected >= -eps - 1e-6).all()
    assert (projected <= eps + 1e-6).all()


def test_project_l2_bounded(delta):
    eps = 0.5
    norm = 2
    projected = project(delta, norm, eps)
    norms = projected.norm(p=2, dim=1)
    assert (norms <= eps + 1e-5).all()


def test_project_zero_unchanged():
    norm = "inf"
    zero = torch.zeros(BATCH, DIM)
    projected = project(zero, norm, 0.5)
    assert torch.allclose(projected, zero)


# ---- random_in_ball ----

def test_random_init_shape():
    norm = "inf"
    init = random_in_ball((BATCH, DIM), norm, 0.2, CPU)
    assert init.shape == (BATCH, DIM)


def test_random_init_within_l2_ball():
    norm = 2
    radius = 0.5
    init = random_in_ball((BATCH, DIM), norm, radius, CPU)
    norms = init.norm(p=2, dim=1)
    assert (norms <= radius + 1e-5).all()


def test_random_init_linf_bounded():
    norm = "inf"
    radius = 0.2
    init = random_in_ball((BATCH, DIM), norm, radius, CPU)
    assert (init.abs() <= radius + 1e-6).all()
