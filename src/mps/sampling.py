# script with sampling functions
import torch


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#--------------Single-shot secant sampling (sss_sampling)-------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

# TODO: Add tensor shapes description
def pre_select(p: torch.Tensor):
    """ 
    Basically code of sampling from discrete distribution but 
    returning cdf, random sample from uniform(0,1) and position of containing bin

    Parameters
    ----------
    p: tensor
        (unnormalized) probability mass function
    
    Returns
    -------
    tuple(tensor, tensor, int)
        normalised cdf, random float in (0,1), index j=inf(u<F(x_j))
    """
    cdf = p.cumsum(dim=-1) # cumulative sum
    cdf_norm = cdf / cdf[:, -1:] # normalisation of cdf
    
    nu = torch.rand(p.size(0), 1) # batch x 1 samples nu of uniform([0,1))
    
    # extract index j for which nu in (CDF(z_j-1), CDF(z_j))
    ids = p.detach().size(1) - torch.le(nu, cdf_norm.detach()).sum(dim=1) # batch length number of indices j
    return cdf_norm, nu, ids

# TODO: Add tensor shapes description
def sss_sampling(   p: torch.Tensor,
                    z: torch.Tensor
                    ) -> torch.Tensor:
    """
    Single-shot secant sampling on a predefined grid of x-values. 
    Interpolation step gives gradient signal.

    Parameters
    ----------
    p: tensor
        (unnormalized) probability mass function on grid
    z: tensor
        the grid of a univariate variable

    Returns
    -------
    tensor
        sample(s)
    """
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


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#--------------Gumbel-softmax sampling-------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
# TODO: Add Grumbel-softmax sampling


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#--------------Spline-interpolation based sampling-------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
# TODO: Add Spline-based sampling