# script with sampling functions
import torch
import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# --------------Single-shot secant sampling (sss_sampling)-------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def pre_select(p: torch.Tensor):
    """ 
    Basically code of sampling from discrete distribution but 
    returning cdf, random sample from uniform(0,1) and position of containing bin

    Parameters
    ----------
    p: tensor
        (unnormalized) probability mass function
        shape: (batch, num_bins)

    Returns
    -------
    cdf_norm: tensor
        shape (batch, num_bins)
    nu: tensor (no gradient)
        shape (batch, 1). The samples from the uniform dist
    ids: tensor (no gradient)
        shape (batch,). ids[b] = argmin_j (nu[b]<cdf[b,j])
    """
    cdf = p.cumsum(dim=-1)  # cumulative sum
    cdf_norm = cdf / cdf[:, -1:]  # normalisation of cdf

    nu = torch.rand(p.size(0), 1, device=p.device)  # 0 <= nu[b,0] < 1

    # extract index j for which nu in (CDF(z_j-1), CDF(z_j))
    # ids = p.detach().size(1) - torch.le(nu, cdf_norm.detach()).sum(dim=1) # batch length number of indices j
    cmp = (nu < cdf_norm.detach())
    # torch.argmax returns index of first (by the index count of the dimension to be reduced) maximal value
    ids = cmp.float().argmax(dim=-1)
    return cdf_norm, nu, ids


def os_secant(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Single-shot secant sampling on a predefined grid of x-values. 
    Interpolation step gives gradient signal.

    Parameters
    ----------
    p: tensor
        (unnormalized) probability density function evaluated on grid
        shape: (batch, num_bins)
    z: tensor
        the grid of a univariate variable
        shape: (num_bins,)

    Returns
    -------
    samples: tensor
        shape (batch,)
    """
    cdf, nu, ids_b = pre_select(p=p)

    ids_b = torch.clamp(ids_b, min=1)  # Prevent -1, shape (batch,)
    b = torch.gather(input=z, dim=0, index=ids_b).detach()  # shape (batch,)
    a = torch.gather(input=z, dim=0, index=ids_b-1).detach()
    idx = ids_b.unsqueeze(1)                    # shape: (batch, 1)
    idx_prev = (ids_b - 1).unsqueeze(1)

    cdf_b = torch.gather(cdf, dim=1, index=idx).squeeze(
        1)       # shape: (batch,)
    cdf_a = torch.gather(cdf, dim=1, index=idx_prev).squeeze(1)
    denom = (cdf_b - cdf_a).clamp(min=1e-12)
    samples = a + (b - a) * (nu.squeeze(1) - cdf_a) / denom

    return samples  # shape (batch, )


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# --------------Gumbel-softmax sampling-------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# TODO: Add Grumbel-softmax sampling


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# --------------Spline-interpolation based sampling-------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# TODO: Add Spline-based sampling


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# --------------main function-------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

_SAMPLING_MAP = {
    "secant": os_secant,
    # add more methods here
}


def main(p: torch.Tensor, input_space: torch.Tensor, method: str = "os_secant") -> torch.Tensor:
    """
    Draw samples from a probability distribution tensor using the specified method.

    Parameters
    ----------
    p : torch.Tensor
        Probability vector for one site (shape: batch_size, num_bins)
    input_space : torch.Tensor
        1D tensor of candidate values corresponding to bins
    method : str
        Sampling method name (must exist in _SAMPLING_MAP)

    Returns
    -------
    torch.Tensor
        Sampled values (shape: batch_size,)
    """

    if method not in _SAMPLING_MAP:
        raise ValueError(
            f"Sampling method {method} not recognized. Available: {list(_SAMPLING_MAP.keys())}")
    return _SAMPLING_MAP[method](p, input_space)
