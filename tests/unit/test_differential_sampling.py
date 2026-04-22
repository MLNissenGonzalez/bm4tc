import pytest
import torch
from src.models.generator.differential_sampling import (
    pre_select,
    os_secant,
    multinomial_sampling,
    main,
)

BATCH = 8
BINS = 20


@pytest.fixture
def grid():
    return torch.linspace(0.0, 1.0, BINS)


@pytest.fixture
def uniform_p():
    return torch.ones(BATCH, BINS)


# ---- pre_select ----

def test_pre_select_cdf_monotone(uniform_p):
    cdf, _, _ = pre_select(uniform_p)
    diffs = cdf[:, 1:] - cdf[:, :-1]
    assert (diffs >= -1e-6).all()


def test_pre_select_nu_shape_and_range(uniform_p):
    _, nu, _ = pre_select(uniform_p)
    assert nu.shape == (BATCH, 1)
    assert (nu >= 0).all() and (nu < 1).all()


def test_pre_select_ids_valid_range(uniform_p):
    _, _, ids = pre_select(uniform_p)
    assert ids.shape == (BATCH,)
    assert (ids >= 0).all() and (ids < BINS).all()


# ---- multinomial_sampling ----

def test_multinomial_output_in_grid(uniform_p, grid):
    samples = multinomial_sampling(uniform_p, grid)
    grid_set = set(grid.tolist())
    for s in samples.tolist():
        assert any(abs(s - g) < 1e-5 for g in grid_set)


def test_multinomial_output_shape(uniform_p, grid):
    samples = multinomial_sampling(uniform_p, grid)
    assert samples.shape == (BATCH,)


def test_multinomial_zero_row_fallback(grid):
    p = torch.zeros(BATCH, BINS)
    samples = multinomial_sampling(p, grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_multinomial_single_nonzero_bin(grid):
    p = torch.zeros(BATCH, BINS)
    p[:, 5] = 1.0
    samples = multinomial_sampling(p, grid)
    expected = grid[5].item()
    assert all(abs(s - expected) < 1e-5 for s in samples.tolist())


def test_multinomial_nan_probs_handled(grid):
    p = torch.full((BATCH, BINS), float("nan"))
    samples = multinomial_sampling(p, grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_multinomial_posinf_handled(grid):
    p = torch.full((BATCH, BINS), float("inf"))
    samples = multinomial_sampling(p, grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


# ---- os_secant ----

def test_os_secant_output_in_range(uniform_p, grid):
    samples = os_secant(uniform_p, grid)
    step = (grid[-1] - grid[0]) / (len(grid) - 1)
    assert (samples >= grid.min() - step).all()
    assert (samples <= grid.max() + step).all()


def test_os_secant_output_shape(uniform_p, grid):
    samples = os_secant(uniform_p, grid)
    assert samples.shape == (BATCH,)


# ---- main dispatcher ----

def test_main_multinomial_dispatch(uniform_p, grid):
    samples = main(uniform_p, grid, "multinomial")
    assert samples.shape == (BATCH,)


def test_main_secant_dispatch(uniform_p, grid):
    samples = main(uniform_p, grid, "secant")
    assert samples.shape == (BATCH,)


def test_main_unknown_method_raises(uniform_p, grid):
    with pytest.raises(ValueError):
        main(uniform_p, grid, "nonexistent_method")


def test_main_batch_size_1(grid):
    p = torch.ones(1, BINS)
    samples = main(p, grid, "multinomial")
    assert samples.shape == (1,)
