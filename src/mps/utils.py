import torch
import tensorkrowch as tk
from typing import Union, Sequence, Callable, Dict, Tuple, List
import sampling as sampling
import torch.nn as nn
from dataclasses import dataclass

import numpy as np


import logging
logger = logging.getLogger(__name__)

# Input to batch_sampling_mps
@dataclass
class PretrainSamplingConfig:
    # mps
    embedding: str
    cls_pos: int
    num_spc: int
    num_bins: int
    batch_spc: int
    # device

@dataclass 
class MPSSamplingConfig:
    # mps
    embedding: str
    cls_pos: int
    num_bins: int
    num_spc: int
    # device 

# cc_mps_sampling input
@dataclass
class ClassSamplingConfig:
    # mps
    embedding: str
    cls_pos: int
    num_bins:int
    num_spc: int
    cls: int
    # device

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
        p = torch.diagonal(mps(in_tensors, marginalize_output=True))
    # Case 2: All variables appear.
    else:
        p = torch.square(mps.forward(data=in_tensors))
        
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
    batch_size: int
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


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----Sampling routines using mps------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----Currenty on purely sequential implementation-------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Add other sampling methods and add it as configuration for experiments
# TODO: Add num_bins as a hyperparameter in ad_train.



def cc_mps_sampling(mps: tk.models.MPS,
                    embedding: str,
                    cls_pos: int, # extract this once for the mps, not at every sampling step
                    num_bins: int,
                    num_spc: int, # per class                    
                    cls: int,
                    device: torch.device
                    )-> torch.Tensor:
    """
    Class-conditional sampling from an MPS (single class).

    This function samples `num_spc` independent samples conditioned on a single
    class label `cls`. Sampling is sequential over the feature sites of the MPS
    (skipping the site `cls_pos` which is fixed to the class embedding), using
    the Born rule computed by `born_sequential`.

    Important shapes / notes
    -----------------------
    - n_sites = mps.n_features
    - input_space: 1D grid of length `num_bins` (torch.linspace)
    - embedding(input_space, in_dim) -> (num_bins, in_dim)
    - For born_sequential, each `embs[site]` must be shaped (num_spc * num_bins, phys_dim)
      where phys_dim == in_dim for your embedding function.
    - The probability vector returned by born_sequential is reshaped to (num_spc, num_bins),
      then sampling.sss_sampling(p, input_space) is used to draw one value per sample.

    Parameters
    ----------
    mps : tk.models.MPS
        The trained MPS model object used to compute Born probabilities.
    embedding : str
        Identifier for the embedding; converted to a callable via get_embedding(embedding).
    cls_pos : int
        Index of the MPS site reserved for the class label (this site is fixed to a class embedding).
    num_bins : int
        Number of candidate discrete values per site (resolution of `input_space`).
    num_spc : int
        Number of independent samples to draw for this class.
    cls : int
        Integer class index (used to form a one-hot / basis embedding).
    device : torch.device
        Device to put tensors on (cpu/cuda).

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_spc, data_dim) where data_dim == n_sites - 1 (class site excluded).
        Each row is a sampled vector in the original input space (not embedded).
    """
    embs = {}
    samples = []
    mps.to(device)

    # Create the 1D grid (input_space) and instanciate the embedding callable
    rang = _embedding_to_range(embedding)
    embedding = get_embedding(embedding)
    input_space = torch.linspace(rang[0], rang[1], num_bins)

    in_dim, num_cls = _get_indim_and_ncls(mps)

    # Class basis embedding (one-hot) -> expand to (num_spc * num_bins, num_cls)
    cls_emb = tk.embeddings.basis(torch.tensor(cls), num_cls).float()
    embs[cls_pos] = cls_emb[None, :].expand(num_spc * num_bins, -1).to(device)

    # Input embedding -
    in_emb = embedding(input_space, in_dim)  # [num_bins, in_dim]
    in_emb = in_emb[None, :, :].expand(num_spc, -1, -1)  # [num_samples, num_bins, in_dim]
    in_emb = in_emb.reshape(num_spc * num_bins, in_dim).to(device) # [num_samples * num_bins, phys_dim]

    # Sequentially sample each non-class site.
    for site in range(mps.n_features):
        if site == cls_pos:
            continue   

        embs[site] = in_emb 

        # This reset has to be performed as born_sequential assigns new in features everytime new embedding is added
        mps.unset_data_nodes()
        mps.reset()
        
        # Compute marginal probability over the bins for the current site for each sample.
        p = born_sequential(mps=mps, embs=embs).view(num_spc, num_bins) # shape (num_spc, num_bins)
        
        # Draw one bin value per sample and site from p, TODO: Make configable
        samples.append(sampling.sss_sampling(p, input_space)) # shape (num_spc,)
        
        # Embed drawn samples and use for conditioning for the next site.
        embs[site] = embedding(samples[-1], in_dim)[:, None, :].expand(-1, num_bins, -1).reshape(num_spc*num_bins, -1)

    # Stack sampled per-site arrays across sites 
    samples = torch.stack(tensors=samples, dim=1) # shape (num_spc, data_dim)
    return samples

def _cc_mps_sampling(mps: tk.models.MPS,
                     embedding: Callable[[torch.Tensor, int], torch.Tensor],
                     cls_pos: int, # extract this once for the mps, not at every sampling step
                     cls_emb: torch.Tensor, # perform embedding once, torch.Size=[num_cls]  
                     in_dim: int,
                     num_bins: int,
                     input_space: torch.Tensor, # need to have the same number of bins                                   
                     num_spc: int, # samples per class             
                     device: torch.device
                    )-> torch.Tensor:
    """
    Class-conditional sampling helper that expects a callable embedding and prebuilt input_space.

    This is functionally the same as cc_mps_sampling but receives already-prepared
    embedding callable, input_space and cls_emb tensor to avoid recomputation.

    Parameters
    ----------
    mps : tk.models.MPS
    embedding : Callable[[torch.Tensor, int], torch.Tensor]
        Callable that maps a 1D tensor of real inputs of length B to a (B, in_dim) tensor.
    cls_pos : int
        Index of the class site in the MPS.
    in_dim : int
        Physical dimension (embedding size) returned by `embedding`.
    num_bins : int
        Size of the discretization for each site.
    input_space : torch.Tensor
        1D tensor of length `num_bins` giving the candidate real-values for each site.
    cls_emb : torch.Tensor
        Precomputed class embedding (one-hot vector sized to match MPS physical dim at cls_pos).
        Expected shape: (num_cls,)
    num_spc : int
        Number of samples to produce per class.
    device : torch.device

    Returns
    -------
    torch.Tensor
        (num_spc, data_dim) sampled values in real input space (not embedded).
    """

    samples = []
    embs = {} # best way to save embeddings and their position

    embs[cls_pos] = cls_emb[None, :].expand(num_spc * num_bins, -1).to(device)

    # Input embedding TODO: Could move outside, as class and batch independent
    in_emb = embedding(input_space, in_dim)  # [num_bins, in_dim]
    in_emb = in_emb[None, :, :].expand(num_spc, -1, -1)  # [num_samples, num_bins, in_dim]
    in_emb = in_emb.reshape(num_spc * num_bins, in_dim).to(device) # [num_samples * num_bins, phys_dim]

    for site in range(mps.n_features):
        if site == cls_pos:
            continue   

        # This reset has to be performed as born_sequential assigns new in features everytime new embedding is added
        mps.unset_data_nodes()
        mps.reset()
    
        # Compute marginal probability over the bins for the current site for each sample.
        p = born_sequential(mps=mps, embs=embs).view(num_spc, num_bins) # shape (num_spc, num_bins)
        
        # Draw one bin value per sample and site from p, TODO: Make configable
        samples.append(sampling.sss_sampling(p, input_space)) # shape (num_spc,)
        
        # Embed drawn samples and use for conditioning for the next site.
        embs[site] = embedding(samples[-1], in_dim)[:, None, :].expand(-1, num_bins, -1).reshape(num_spc*num_bins, -1)

    # Stack sampled per-site arrays across sites 
    samples = torch.stack(tensors=samples, dim=1) # shape (num_spc, data_dim)
    return samples


# TODO: Vectorize across classes ? 
def mps_sampling(   mps: tk.models.MPS,
                    embedding: str,
                    cls_pos: int, # need to provide this if num_cls == in_dim.
                    num_bins: int,
                    num_spc: int, # per class                    
                    device: torch.device) -> torch.Tensor:
    """
    Top-level class-conditional sampler using string-identified embedding.
    Sample multiple classes by looping over class-conditional sampler `_cc_mps_sampling`.

    This prepares `input_space`, converts `embedding` (string -> callable) and runs
    `_cc_mps_sampling` (one call per class).

    Returns a tensor shaped (num_spc, num_classes, data_dim) where data_dim = n_sites - 1.
    

    Returns
    -------
    torch.Tensor
        (num_spc, num_classes, data_dim)
    """
    
    # Initilization code
    samples = []
    mps.to(device)
    rang = _embedding_to_range(embedding)
    embedding = get_embedding(embedding)
    input_space = torch.linspace(rang[0], rang[1], num_bins)
    in_dim, num_cls = _get_indim_and_ncls(mps)

    for cls in range(num_cls):
        cls_emb = tk.embeddings.basis(torch.tensor(cls), num_cls).float()
        cls_samples = _cc_mps_sampling(
            mps=mps,embedding=embedding, cls_pos=cls_pos, in_dim=in_dim,
            cls_emb=cls_emb, num_bins=num_bins, input_space=input_space,
            num_spc=num_spc, device=device
        ) # shape (num_spc, data_dim)
        samples.append(cls_samples)

    samples = torch.stack(tensors=samples, dim=1) # shape: (n_spc, num_cls, data_dim)
    return samples 

# TODO: Vectorize across classes ? 
def _mps_sampling(mps: tk.models.MPS,
                  embedding: Callable[[torch.Tensor, int], torch.Tensor],
                  cls_embs: Sequence[torch.Tensor], # one embedding per class,
                  cls_pos: int,
                  in_dim: int,  
                  num_bins: int,
                  input_space: torch.Tensor,                    
                  num_spc: int, # per class,                                        
                  device: torch.device) -> torch.Tensor:
    """
    Helper version of `mps_sampling` with some computation already assumed to be have been done before.

    Parameters
    ----------
    mps : tk.models.MPS
    embedding : Callable
    cls_embs : sequence of torch.Tensor
        Precomputed embeddings (one per class) — each should match the MPS's class-embedding size.
    cls_pos : int
        Position of the class site in the MPS.
    in_dim : int
        Physical embedding dimension.
    num_bins : int
    input_space : torch.Tensor
    num_spc : int
        Number of samples per class.
    device : torch.device

    Returns
    -------
    torch.Tensor
        (num_spc, num_classes, data_dim)
    """
    
    # Initialization code
    samples = []
    for cls_emb in cls_embs:
        cls_samples = _cc_mps_sampling(
            mps=mps,embedding=embedding, cls_pos=cls_pos, in_dim=in_dim,
            cls_emb=cls_emb, num_bins=num_bins, input_space=input_space,
            num_spc=num_spc, device=device
        ) # shape (num_spc, data_dim)
        samples.append(cls_samples)

    samples = torch.stack(tensors=samples, dim=1) # shape: (n_spc, num_cls, data_dim)
    return samples


# TODO: Vectorize across batches/classes 
def batch_sampling_mps( mps: tk.models.MPS,
                        embedding: str,
                        cls_pos: int,
                        num_spc: int, # total number of samples per class, ought to be divisible by batch_size? hard coding is bad
                        num_bins: int,
                        batch_spc: int, # will be num_spc in mps_sampling # only needed for visualisation
                        device: torch.device
):
    """
    Draw many samples per class in batches to limit peak memory use.

    This function splits the total requested `num_spc` into batches of size `batch_spc`
    (the last batch may be smaller). Each batch is produced by `_mps_sampling` and
    concatenated to build the final dataset.

    Parameters
    ----------
    mps : tk.models.MPS
    embedding : str
        Embedding identifier -> converted via get_embedding.
    cls_pos : int
    num_spc : int
        Total number of samples to draw per class (will be split into batches).
    num_bins : int
    batch_spc : int
        Number of per-class samples produced per sub-batch (used to call _mps_sampling).
    device : torch.device

    Returns
    -------
    samples: tensor
        shape: (num_spc, num_cls, data.dim)
    """

    # Initialization code
    cls_embs = []
    mps.to(device)
    rang = _embedding_to_range(embedding)
    embedding = get_embedding(embedding)
    input_space = torch.linspace(rang[0], rang[1], num_bins)
    in_dim, num_cls = _get_indim_and_ncls(mps)
    for cls in range(num_cls):
        cls_embs.append(tk.embeddings.basis(torch.tensor(cls), num_cls).float())

    num_batches = (num_spc + batch_spc - 1) // batch_spc
    samples = []
    for _ in range(num_batches): # could compute these batches in parallel, not to bad
        batch = _mps_sampling(
            mps=mps, embedding=embedding, cls_embs=cls_embs,
            cls_pos=cls_pos, in_dim=in_dim,
            num_bins=num_bins, input_space=input_space,
            num_spc=batch_spc, device=device
        ) # (batch_size, num_cls, data.dim) samples
        samples.append(batch)
    
    samples = torch.concat(tensors=samples, dim=0) # (batch_size x num_batches, num_cls, data.dim)
    samples = samples[:num_spc, :, :] # slice to (num_spc, num_cls, data.dim)

    return samples


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