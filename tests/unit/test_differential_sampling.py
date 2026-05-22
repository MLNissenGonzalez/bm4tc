import pytest
import torch
from src.utils.sampling import draw_from_grid

BATCH = 8
BINS = 20


@pytest.fixture
def grid():
    return torch.linspace(0.0, 1.0, BINS)


@pytest.fixture
def uniform_p():
    return torch.ones(BATCH, BINS)


# ---- draw_from_grid ----

def test_draw_from_grid_output_in_grid(uniform_p, grid):
    samples = draw_from_grid(uniform_p, grid)
    grid_set = set(grid.tolist())
    for s in samples.tolist():
        assert any(abs(s - g) < 1e-5 for g in grid_set)


def test_draw_from_grid_output_shape(uniform_p, grid):
    samples = draw_from_grid(uniform_p, grid)
    assert samples.shape == (BATCH,)


def test_draw_from_grid_zero_row_fallback(grid):
    p = torch.zeros(BATCH, BINS)
    samples = draw_from_grid(p, grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_draw_from_grid_single_nonzero_bin(grid):
    p = torch.zeros(BATCH, BINS)
    p[:, 5] = 1.0
    samples = draw_from_grid(p, grid)
    expected = grid[5].item()
    assert all(abs(s - expected) < 1e-5 for s in samples.tolist())


def test_draw_from_grid_nan_probs_handled(grid):
    p = torch.full((BATCH, BINS), float("nan"))
    samples = draw_from_grid(p, grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_draw_from_grid_posinf_handled(grid):
    p = torch.full((BATCH, BINS), float("inf"))
    samples = draw_from_grid(p, grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_draw_from_grid_batch_size_1(grid):
    p = torch.ones(1, BINS)
    samples = draw_from_grid(p, grid)
    assert samples.shape == (1,)
