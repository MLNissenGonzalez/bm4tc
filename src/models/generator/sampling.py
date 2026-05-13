import torch
import logging
logger = logging.getLogger(__name__)


def pre_select(p: torch.Tensor):
    """
    Sample from a discrete distribution, returning CDF, uniform draw, and bin index.

    Parameters
    ----------
    p : torch.Tensor
        Unnormalized probability mass function, shape (batch, num_bins).

    Returns
    -------
    cdf_norm : torch.Tensor, shape (batch, num_bins)
    nu : torch.Tensor, shape (batch, 1) — uniform draw
    ids : torch.Tensor, shape (batch,) — bin indices
    """
    cdf = p.cumsum(dim=-1)
    cdf_norm = cdf / cdf[:, -1:]
    nu = torch.rand(p.size(0), 1, device=p.device)
    cmp = (nu < cdf_norm.detach())
    ids = cmp.float().argmax(dim=-1)
    return cdf_norm, nu, ids


def multinomial_sampling(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Sample from a discretized conditional distribution using torch.multinomial.

    Hard (non-differentiable) inverse-CDF sampling. Draws one bin index per batch
    element proportional to p, then returns the corresponding grid value from z.

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


_SAMPLING_MAP = {
    "multinomial": multinomial_sampling,
}


def sample(p: torch.Tensor, input_space: torch.Tensor, method: str = "multinomial") -> torch.Tensor:
    """
    Draw samples from a probability distribution tensor using the specified method.

    Parameters
    ----------
    p : torch.Tensor
        Probability vector for one site, shape (batch, num_bins).
    input_space : torch.Tensor
        1D tensor of candidate values corresponding to bins.
    method : str
        Sampling method name. Currently supported: "multinomial".

    Returns
    -------
    torch.Tensor
        Sampled values, shape (batch,).
    """
    if method not in _SAMPLING_MAP:
        raise ValueError(
            f"Sampling method '{method}' not recognized. Available: {list(_SAMPLING_MAP.keys())}")
    return _SAMPLING_MAP[method](p, input_space)


if __name__ == "__main__":
    p = torch.tensor([[0.1, 0.5, 0.3, 0.1], [0.25, 0.25, 0.25, 0.25]])
    z = torch.linspace(0, 1, 4)
    out = sample(p, z)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    print("sampling.py OK:", out)
