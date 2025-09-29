import torch
import tensorkrowch as tk
from typing import Dict, Tuple
import torch.nn as nn
import wandb
import numpy as np


import logging
logger = logging.getLogger(__name__)



#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#---------------Getter functions for MPS-----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

# TODO: Rewrite this to fit papers definition
def legendre_embedding(x: torch.Tensor, dim: int):
    return tk.embeddings.poly

_EMBEDDING_MAP = {
    "fourier": tk.embeddings.fourier,
    "poly": tk.embeddings.poly,
    "polynomial": tk.embeddings.poly,
    "legendre": legendre_embedding
}

def get_embedding(name: str):
    """
    Given embedding identifier, return the associated embedding function
    using the `_EMBEDDING_TO_RANGE` dictionary.
    """
    key = name.lower()
    try:
        embedding = _EMBEDDING_MAP[key]
    except KeyError:
        raise ValueError(f"Embedding {name} not recognised. "
                         f"Available: {list(_EMBEDDING_MAP.keys())}")
    return embedding


_EMBEDDING_TO_RANGE = {
    "fourier": (0., 1.),
    "legendre": (-1., 1.)
}

def _embedding_to_range(embedding: str):
    """
    Given embedding identifier, return the associated domain of that embedding
    using the `_EMBEDDING_TO_RANGE` dictionary.
    """
    key = embedding.replace(" ", "").lower()
    try:
        rang = _EMBEDDING_TO_RANGE[key]
    except KeyError:
        raise ValueError(f"Embedding {embedding} not recognised. "
                         f"Available: {list(_EMBEDDING_TO_RANGE.keys())}")
    return rang

def _get_indim_and_ncls(mps: tk.models.MPS)-> Tuple[int, int]:
    """
    Extracts `in_dim`, the input embedding dimension, and `n_cls`, the number of classes assuming same input embedding dimension.

    Returns
    -------
    Tuple[int, int]
        (in_dim, n_cls)
    """
    phys = np.array(mps.phys_dim)
    val, counts = np.unique(ar=phys, return_counts=True)
    if np.max(counts) < phys.size-1:
        raise ValueError("Can only handle same dimensional input embedding dimensions.")
    in_dim = val[np.argmax(counts)]
    if len(val) == 2:
        n_cls = val[val!=in_dim][0]
    else:
        n_cls = in_dim
    return int(in_dim), int(n_cls)


#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#---------------Logger functions MPS-----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
# TODO: Change this function such that it counts the number of times gradients have not been none (not important)
def count_mps_grads(mps: tk.models.MPS, step: int, watch_freq: int, stage: str):
    """Log gradients of tensors in a dict {name: tensor} to W&B."""
    log_grads= {}
    if step % watch_freq == 0:
        for n in mps.mats_env: # also tried mps.tensors with the same error
            if n.grad is not None:
                log_grads[f"{stage}_mps/gradients/{n.name}"] = +1
            else:
                log_grads[f"{stage}_mps/gradients/{n.name}"] = -1
        if log_grads:
            wandb.log(log_grads) 

# TODO: Use this only for debugging
def logged_optimizer_step(optimizer: torch.optim.Optimizer, mps: tk.models.MPS):
    p_before = {}
    t_before = {}
    for name, param in mps.named_parameters():
        p_before[name] = param.clone().detach()
    for node in mps.mats_env:
        t_before[f"{node.name}"] = node.tensor.clone().detach()
    
    optimizer.step()

    for name, param in mps.named_parameters():
        after = param.clone().detach()
        diff = (after - p_before[name]).abs().max()
        wandb.log({f"gan_mps/optim_diff/{name}": diff})
    for node in mps.mats_env:
        after = node.tensor.clone().detach()
        diff = (after - t_before[f"{node.name}"]).abs().max()
        wandb.log({f"gan_mps/optim_diff/{name}": diff})


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------Born rule. Sequential and parallel code.------------------------------------------------------------------------------------------------------------------
#------Could add this maybe as method in a custom MPS class.--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------torch.optim.Optimizer------------------------------------------------------------

# TODO: batch_size could be a confusing misnomer in the documentation. batch_size could be the product of num_bins and num_dist one want so compute.
# TODO: Add tensor shapes

def born_sequential(mps: tk.models.MPS, 
                    embs: Dict[int, torch.Tensor] | torch.Tensor)-> torch.Tensor: # very flexible, not parallizable
    """ 
    Sequential contraction of MPS with embedded input and computation of probabilities using the Born rule. Mainly used for sampling.

    Cases
    -----
    - Not all variables (including class index) appear in p({x_i}_I|{x_j}_J), i.e. in embs. 
        In that case, one has to marginalize over the missing variables resulting in
        a density matrix whose diagonal contrains the unnormalized distribution(s) (batch_size x num_bins values). 
    - All variables appear in embs, one has to square the output of the mps to obtain the unnormalized distribution(s). 
    
    Parameters
    ----------
    mps:    tk.models.MPS
        instance defining the prob amplitude
    embs:   dict
        embedded input as dict of batch_size x phys_dim tensors  with keys indicating input position.
    
    Returns
    -------
    tensor
        probability distribution(s)
    """
    
    # TODO: Maybe remove this check, as it might slow the code down.
    if not isinstance(embs, dict):
        raise TypeError("embs input needs to be dictionary with keys indicating input position and values embedded input feature.")
    
    # Advantage of dictionaries is that one can use the keys to save which legs are input and which are not. 
    mps.in_features = [i for i in embs.keys()]
    # tensorkrowch processes lists of inputs sequentially.
    in_tensors = [embs[i] for i in mps.in_features]
    
    # Case 1: Not all variables appear, thus marginalize_output=True and one has to take the diagonal.
    if len(in_tensors) < mps.n_features:
        rho = mps.forward(in_tensors, marginalize_output=True, inline_input=True, inline_mats=True) # density matrix
        p = torch.diagonal(rho) 
    # Case 2: All variables appear.
    else:
        amplitude = mps.forward(data=in_tensors, inline_input=True, inline_mats=True) # prob. amplitude
        p = torch.square(amplitude)
        
    # logger.debug(f"born_sequential output {p.shape=}")
    return p
    
def born_parallel(mps: tk.models.MPSLayer | tk.models.MPS, # could use MPSLayer class for this one actually
                  embs: torch.Tensor)-> torch.Tensor:
    """ 
    Parallel contraction of MPS with embedded input and computation of probabilities using the Born rule. Mainy used for classification.

    Cases
    -----
    - If mps has n_features = D (only input sites):
        Returns joint Born probabilities p(x₁,…,x_D) ∝ |ψ(x)|².
        Shape: (batch_size, 1)

    - If mps has n_features = D+1 (input sites + output site):
        Returns conditional class probabilities p(c|x₁,…,x_D).
        Shape: (batch_size, num_cls)

    Parameters
    ----------
    mps : tk.models.MPSLayer | tk.models.MPS
        MPS model defining the probability amplitude.
    embs : torch.Tensor
        Embedded input of shape (batch_size, D, phys_dim).

    Returns
    -------
    torch.Tensor
        Probabilities with shape (batch_size, num_cls).
    """
    
    # TODO: Think about removing this check for efficiency. 
    if not isinstance(embs, torch.Tensor): # embs=tensor (parallizable), assume mps.out_feature = [cls_pos] globally??
        raise TypeError("embs input needs to be tensor of shape: (batch_size, D, phys_dim)")
    
    is_joint = isinstance(mps, tk.models.MPS) and (mps.n_features == embs.shape[1])
    # Case: p(c|x_1,..., x_D), since n_features = D+1 (with output site)
    if not is_joint:
        p = torch.square(mps.forward(data=embs)) 
        p = p / p.sum(dim=-1, keepdim=True) 
    # Case: p(x_1,...,x_D), since n_features = D (no central tensor).
    else:
        p = torch.square(mps.forward(data=embs)).unsqueeze(1) # Z x p(x_1,..., x_D) # shape (batch_size, 1)
    # logger.debug(f"born_parallel output: {p.shape=}")
    return p

# TODO: Think of deleting this function as it may never be used.
def batch_normalize(p: torch.Tensor,
                    num_bins: int,
                    batch_size: int):
    """
    Normalize a batch of unnormalized discretized (univariate) probability distributions stored in a single tensor.
    
    Parameters
    ----------
    p: tensor
        batch of unnormalized probability distributions stored in a single tensor of shape (batch_size, num_bins) or (batch_size x num_bins).
    batch_size: intsamples.append(diff_sampling.os_secant(
            p, input_space))  # shape (num_spc,)
        number of unnormalized probability distributions stored in the tensor p
    num_bins: int
        number of discretization points for every single unnormalized probability distribution p[i, :].

    Returns
    -------
    tensor
        single tensor storing batch_size many normalized univariate discrete probability distributions of num_bins bins.
    """
    if p.shape == (batch_size*num_bins,):
        p = p.reshape(batch_size, num_bins)

    return p / p.sum(dim=-1, keepdim=True)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------CONDITIONING VIA NEW CLASS---------------------------------------------------------------------------------------------
#--------I MAY SWITCH TO THIS METHOD IF IT IS BETTER---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


# TODO: Add documentation

# The only case where the class below runs with gradients flowing to R_tensor and
# Cls_tensor is if parametrize=False and inline_input=True for every forward call

class CC_MPS(tk.models.MPS):
    def __init__(self, mps: tk.models.MPS, 
                 cls_pos: int,
                 parametrize: bool):
        # Number of features of the new MPS is reduced by one
        n_feat_new = mps.n_features - 1
        phys_dim = mps.phys_dim[0]
        bond_dim = mps.bond_dim[0]
        self.num_cls = mps.phys_dim[cls_pos]

        # Initialize model with right properties
        super().__init__(n_features=n_feat_new, 
                         phys_dim=phys_dim, 
                         bond_dim=bond_dim, 
                         boundary='obc')

        # Extract and save special tensors as parameters
        assert all(isinstance(mps.mats_env[i].tensor, torch.Tensor) 
                   for i in [cls_pos, cls_pos + 1])

        cls_tensor = mps.mats_env[cls_pos].tensor
        R_tensor = mps.mats_env[cls_pos + 1].tensor

        assert cls_tensor is not None
        assert R_tensor is not None
        self.cls_tensor = nn.Parameter(cls_tensor.clone())
        self.R_tensor = nn.Parameter(R_tensor.clone())

        # Copy all tensors into new mps except special tensor. 
        # Empy node is initialised randomly which is not important
        for old_idx in range(mps.n_features):
            if old_idx == cls_pos or old_idx == cls_pos + 1:
                continue  # skip special cls and R tensors

            # Compute correct new index in the reduced model
            new_idx = old_idx if (old_idx < cls_pos) else (old_idx - 1)
            old_ts = mps.mats_env[old_idx].tensor
            assert old_ts is not None
            self.mats_env[new_idx].set_tensor(old_ts.clone())

        # Store cls_pos for convenience
        self.cls_pos = cls_pos

        # TODO: Add possibility for different embeddings for multi-label
        basis_embs = torch.stack([
            tk.embeddings.basis(torch.tensor(i), dim=self.num_cls).to(dtype=torch.float)
            for i in range(self.num_cls)
        ])

        # Register as buffer (not a Parameter, not trainable)
        self.register_buffer("cls_embs", basis_embs)

        # ParamNode -> Node, doesn't work with stacking
        self.mats_env[self.cls_pos] = self.mats_env[self.cls_pos].parameterize(parametrize)
        

    def forward(self, data, cls, *args, **kwargs):
        assert cls in range(self.num_cls), "not that many classes"
        assert data.shape[1] == self.n_features, "Input feature count mismatch"

        cls_emb = self.cls_embs[cls] # type: ignore
        aux_tensor = torch.einsum("c, lcr, rim -> lim", 
                                  cls_emb, 
                                  self.cls_tensor, 
                                  self.R_tensor)

             
        self.mats_env[self.cls_pos].tensor = aux_tensor

        return super().forward(data=data, *args, **kwargs)
    

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------ENSEMBLE MPS METHOD---------------------------------------------------------------------------------------------
#--------I MAY SWITCH TO THIS METHOD IF IT IS BETTER---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# Instead of a single MPS with a central cls tensor, one could train an ensemble of tensors
# interacting with each other only through the (classification) loss
# Conditioning implies what?


# TODO: Implement ensemble method