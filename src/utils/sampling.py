import torch
import logging
logger = logging.getLogger(__name__)


def draw_from_grid(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Shared multinomial leaf: draw one grid value per batch element proportional to p.

    Hard (non-differentiable) inverse-CDF sampling via torch.multinomial. Handles
    degenerate inputs (NaN, inf, all-zero rows) gracefully.

    Parameters
    ----------
    p : torch.Tensor
        Unnormalized probability weights, shape (batch, num_bins). Must be >= 0.
    z : torch.Tensor
        Grid of candidate values, shape (num_bins,).

    Returns
    -------
    torch.Tensor
        Sampled grid values, shape (batch,).
    """
    p_clean = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    row_sums = p_clean.sum(dim=-1, keepdim=True)
    p_clean = torch.where(row_sums > 0, p_clean, torch.ones_like(p_clean))
    indices = torch.multinomial(p_clean, num_samples=1).squeeze(1)
    return z[indices]


if __name__ == "__main__":
    p = torch.tensor([[0.1, 0.5, 0.3, 0.1], [0.25, 0.25, 0.25, 0.25]])
    z = torch.linspace(0, 1, 4)
    out = draw_from_grid(p, z)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    print("sampling.py OK:", out)
