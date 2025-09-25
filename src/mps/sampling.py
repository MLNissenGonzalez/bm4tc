from dataclasses import dataclass
from mps.utils import get_embedding, _embedding_to_range, _get_indim_and_ncls, born_sequential
from typing import Sequence, Callable
import src.diff_sampling as diff_sampling
import tensorkrowch as tk
import torch

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
    num_bins: int
    num_spc: int
    cls: int
    # device

# TODO: Add other sampling methods and add it as configuration for experiments
# TODO: Add num_bins as a hyperparameter in ad_train.


def single_class(mps: tk.models.MPS,
                 embedding: str,
                 cls_pos: int,  # extract this once for the mps, not at every sampling step
                 num_bins: int,
                 num_spc: int,  # per class
                 cls: int,
                 device: torch.device
                 ) -> torch.Tensor:
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
    # [num_samples, num_bins, in_dim]
    in_emb = in_emb[None, :, :].expand(num_spc, -1, -1)
    # [num_samples * num_bins, phys_dim]
    in_emb = in_emb.reshape(num_spc * num_bins, in_dim).to(device)

    # Sequentially sample each non-class site.
    for site in range(mps.n_features):
        if site == cls_pos:
            continue

        embs[site] = in_emb

        # This reset has to be performed as born_sequential assigns new in features everytime new embedding is added
        mps.unset_data_nodes()
        mps.reset()

        # Compute marginal probability over the bins for the current site for each sample.
        p = born_sequential(mps=mps, embs=embs).view(
            num_spc, num_bins)  # shape (num_spc, num_bins)

        # Draw one bin value per sample and site from p, TODO: Make configable
        samples.append(diff_sampling.os_secant(
            p, input_space))  # shape (num_spc,)

        # Embed drawn samples and use for conditioning for the next site.
        embs[site] = embedding(samples[-1], in_dim)[:, None,
                                                    :].expand(-1, num_bins, -1).reshape(num_spc*num_bins, -1)

    # Stack sampled per-site arrays across sites
    samples = torch.stack(tensors=samples, dim=1)  # shape (num_spc, data_dim)
    return samples


def _single_class(mps: tk.models.MPS,
                  embedding: Callable[[torch.Tensor, int], torch.Tensor],
                  cls_pos: int,  # extract this once for the mps, not at every sampling step
                  # perform embedding once, torch.Size=[num_cls]
                  cls_emb: torch.Tensor,
                  in_dim: int,
                  num_bins: int,
                  input_space: torch.Tensor,  # need to have the same number of bins
                  num_spc: int,  # samples per class
                  device: torch.device
                  ) -> torch.Tensor:
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
    if num_spc == 0:
    # Return empty tensor with correct feature dimension
        logger.info(f"No real examples of class {cls_emb.argmax()} in this batch.")
        return torch.empty((0, mps.n_features - 1), device=device)
    
    samples = []
    embs = {}  # best way to save embeddings and their position

    embs[cls_pos] = cls_emb[None, :].expand(num_spc * num_bins, -1).to(device)

    # Input embedding TODO: Could move outside, as class and batch independent
    in_emb = embedding(input_space, in_dim)  # [num_bins, in_dim]
    # [num_samples, num_bins, in_dim]
    in_emb = in_emb[None, :, :].expand(num_spc, -1, -1)
    # [num_samples * num_bins, phys_dim]
    in_emb = in_emb.reshape(num_spc * num_bins, in_dim).to(device)

    for site in range(mps.n_features):
        if site == cls_pos:
            continue

        embs[site] = in_emb
        
        # This reset has to be performed as born_sequential assigns new in features everytime new embedding is added
        mps.unset_data_nodes()
        mps.reset()

        # Compute marginal probability over the bins for the current site for each sample.
        p = born_sequential(mps=mps, embs=embs).view(
            num_spc, num_bins)  # shape (num_spc, num_bins)

        # Draw one bin value per sample and site from p, TODO: Make configable
        samples.append(diff_sampling.os_secant(
            p, input_space))  # shape (num_spc,)

        # Embed drawn samples and use for conditioning for the next site.
        embs[site] = embedding(samples[-1], in_dim
                               )[:, None,:].expand(-1, num_bins, -1
                                                   ).reshape(num_spc*num_bins, -1)

    # Stack sampled per-site arrays across sites
    samples = torch.stack(tensors=samples, dim=1)  # shape (num_spc, data_dim)
    return samples


# TODO: Vectorize across classes ?
def batch(mps: tk.models.MPS,
          embedding: str,
          cls_pos: int,  # need to provide this if num_cls == in_dim.
          num_bins: int,
          num_spc: int,  # per class
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
        cls_samples = _single_class(
            mps=mps, embedding=embedding, cls_pos=cls_pos, in_dim=in_dim,
            cls_emb=cls_emb, num_bins=num_bins, input_space=input_space,
            num_spc=num_spc, device=device
        )  # shape (num_spc, data_dim)
        samples.append(cls_samples)

    # shape: (n_spc, num_cls, data_dim)
    samples = torch.stack(tensors=samples, dim=1)
    return samples

# TODO: Vectorize across classes ?


def _batch(mps: tk.models.MPS,
           embedding: Callable[[torch.Tensor, int], torch.Tensor],
           cls_embs: Sequence[torch.Tensor],  # one embedding per class,
           cls_pos: int,
           in_dim: int,
           num_bins: int,
           input_space: torch.Tensor,
           num_spc: int,  # per class,
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
        cls_samples = _single_class(
            mps=mps, embedding=embedding, cls_pos=cls_pos, in_dim=in_dim,
            cls_emb=cls_emb, num_bins=num_bins, input_space=input_space,
            num_spc=num_spc, device=device
        )  # shape (num_spc, data_dim)
        samples.append(cls_samples)

    # shape: (n_spc, num_cls, data_dim)
    samples = torch.stack(tensors=samples, dim=1)
    return samples


# TODO: Bug could come from here.
# TODO: Vectorize across batches/classes
# TODO: After batch of samples is generated, move it to CPU.
def batched(mps: tk.models.MPS,
            embedding: str,
            cls_pos: int,
            num_spc: int,  # total number of samples per class, ought to be divisible by batch_size? hard coding is bad
            num_bins: int,
            batch_spc: int,  # will be num_spc in mps_sampling # only needed for visualisation
            device: torch.device
            ) -> torch.FloatTensor:
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
    input_space = torch.linspace(rang[0], rang[1], num_bins).to(device)
    in_dim, num_cls = _get_indim_and_ncls(mps)
    for cls in range(num_cls):
        cls_emb = tk.embeddings.basis(torch.tensor(cls), 
                                      num_cls).to(device=device, 
                                                  dtype=torch.float32)
        cls_embs.append(cls_emb)

    num_batches = (num_spc + batch_spc - 1) // batch_spc
    samples = []
    for _ in range(num_batches):  # could compute these batches in parallel, not to bad
        batch = _batch(
            mps=mps, embedding=embedding, cls_embs=cls_embs,
            cls_pos=cls_pos, in_dim=in_dim,
            num_bins=num_bins, input_space=input_space,
            num_spc=batch_spc, device=device
        )  # (batch_size, num_cls, data.dim) samples
        samples.append(batch)

    # (batch_size x num_batches, num_cls, data.dim)
    samples = torch.concat(tensors=samples, dim=0)
    samples = samples[:num_spc, :, :]  # slice to (num_spc, num_cls, data.dim)

    return samples
