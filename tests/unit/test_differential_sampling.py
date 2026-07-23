import pytest
import torch
from src.model import draw_from_grid, draw_from_grid_log

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


# ---- draw_from_grid_log ----

def test_draw_from_grid_log_output_in_grid_and_shape(grid):
    samples = draw_from_grid_log(torch.zeros(BATCH, BINS), grid)
    assert samples.shape == (BATCH,)
    grid_set = grid.tolist()
    for s in samples.tolist():
        assert any(abs(s - g) < 1e-5 for g in grid_set)


def test_draw_from_grid_log_matches_linear_on_benign_weights(grid):
    """log-domain and linear leaves agree bin-for-bin on the same weights."""
    torch.manual_seed(0)
    p = torch.rand(BATCH, BINS) + 1e-3
    torch.manual_seed(42)
    lin = draw_from_grid(p / p.amax(-1, keepdim=True), grid)
    torch.manual_seed(42)
    log = draw_from_grid_log(p.log(), grid)
    assert torch.equal(lin, log)


def test_draw_from_grid_log_survives_overflowing_scale(grid):
    """Weights whose linear form would overflow float32 still sample correctly.

    log-weights ~1e4 => exp overflows to inf in linear space; the row-max
    subtraction keeps the relative magnitudes exact."""
    log_p = torch.full((BATCH, BINS), -1e4)
    log_p[:, 3] = 1e4        # single dominant bin
    samples = draw_from_grid_log(log_p, grid)
    assert torch.allclose(samples, grid[3].expand(BATCH))


def test_draw_from_grid_log_neg_inf_bins_never_chosen(grid):
    """-inf is the log-space mask: those bins must have zero probability."""
    log_p = torch.zeros(BATCH, BINS)
    log_p[:, 5:] = float("-inf")
    samples = draw_from_grid_log(log_p, grid)
    assert (samples < grid[5]).all()


def test_draw_from_grid_log_all_neg_inf_row_uniform_fallback(grid):
    """A fully-masked row has no admissible bin -> uniform, not NaN."""
    samples = draw_from_grid_log(torch.full((BATCH, BINS), float("-inf")), grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_draw_from_grid_log_nan_handled(grid):
    samples = draw_from_grid_log(torch.full((BATCH, BINS), float("nan")), grid)
    assert samples.shape == (BATCH,)
    assert torch.isfinite(samples).all()


def test_draw_from_grid_log_batch_size_1(grid):
    samples = draw_from_grid_log(torch.zeros(1, BINS), grid)
    assert samples.shape == (1,)
