# script with sampling functions
import torch

# TODO: Add documentation
# Single-shot secant sampling (sss_sampling)
def pre_select(p: torch.Tensor):
    """  
    Input
    p: (unnormalized) probability mass function, 
        torch.Tensor: batch x num_bins
    Output
    samples: array of samples of length batch
    """
    cdf = p.cumsum(dim=-1) # cumulative sum
    cdf_norm = cdf / cdf[:, -1:] # normalisation of cdf
    
    nu = torch.rand(p.size(0), 1) # batch x 1 samples nu of uniform([0,1))
    
    # extract index j for which nu in (CDF(z_j-1), CDF(z_j))
    ids = p.detach().size(1) - torch.le(nu, cdf_norm.detach()).sum(dim=1) # batch length number of indices j
    return cdf_norm, nu, ids

# TODO: Add documentation
def sss_sampling(p: torch.Tensor,
                    z: torch.Tensor):
    cdf, nu, ids_b = pre_select(p=p)
    ids_b = torch.clamp(ids_b, min=1)  # Prevent -1
    # The following values might be computed once for z, given that it has linspace structure
    b = torch.gather(input=z, dim=0, index=ids_b).detach()
    a = torch.gather(input=z, dim=0, index=ids_b-1).detach()

    idx = ids_b.unsqueeze(1)                    # shape: (batch, 1)
    idx_prev = (ids_b - 1).unsqueeze(1)

    cdf_b = torch.gather(cdf, dim=1, index=idx).squeeze(1)       # shape: (batch,)
    cdf_a = torch.gather(cdf, dim=1, index=idx_prev).squeeze(1)

    samples = a + (b - a) * (nu.squeeze(1) - cdf_a) / (cdf_b - cdf_a)
    return samples


# Grumbel-softmax sampling
# TODO: Add Grumbel-softmax sampling

# Invertible and smooth (wrt model parameters) 
# Spline Interpolation for sampling
# TODO: Add Spline-based sampling