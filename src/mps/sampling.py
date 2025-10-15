from mps.utils import get_embedding, _embedding_to_range, _get_indim_and_ncls, born_sequential
from typing import Sequence, Callable
from src.differential_sampling import main as diff_sampling
import tensorkrowch as tk
import torch

import logging
logger = logging.getLogger(__name__)


def _single_class(mps: tk.models.MPS,
                  embedding: Callable[[torch.FloatTensor, int], torch.FloatTensor],
                  method: str,
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
    Core class-conditional sampling routine for a single class embedding.

    This function sequentially samples each non-class site of an MPS according to
    its Born probabilities, conditioned on a fixed class embedding at position
    `cls_pos`. Sampling at each step is handled by `diff_sampling(p, input_space, method)`,
    which allows configurable sampling strategies (e.g., secant).

    Parameters
    ----------
    mps : tk.models.MPS
        The MPS model used to compute conditional Born probabilities.
    embedding : Callable[[torch.FloatTensor, int], torch.FloatTensor]
        Embedding function mapping real scalars → feature vectors of dimension `in_dim`.
    method : str
        Sampling method identifier passed to `diff_sampling` (e.g., "secant").
    cls_pos : int
        Index of the class site within the MPS (held fixed during sampling).
    cls_emb : torch.Tensor
        Precomputed embedding of the target class; shape `(num_cls,)`.
    in_dim : int
        Input embedding dimension.
    num_bins : int
        Number of candidate discrete values for each feature (resolution of `input_space`).
    input_space : torch.Tensor
        1D tensor of shape `(num_bins,)` defining the sampling grid of real values.
    num_spc : int
        Number of samples to generate for this class.
    device : torch.device
        Device on which to perform sampling.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(num_spc, data_dim)` where `data_dim = mps.n_features - 1`.
        Each row is a sampled point in the original real-valued input space.
    """
    if num_spc == 0:
        # Return empty tensor with correct feature dimension
        logger.info(
            f"No real examples of class {cls_emb.argmax()} in this batch.")
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

        # This missing caused a silent bug.
        embs[site] = in_emb

        # This reset has to be performed as born_sequential assigns new in features everytime new embedding is added
        mps.unset_data_nodes()
        mps.reset()

        # Compute marginal probability over the bins for the current site for each sample.
        p = born_sequential(mps=mps, embs=embs).view(
            num_spc, num_bins)  # shape (num_spc, num_bins)

        # Draw one bin value per sample and site from p, TODO: Make configable
        feature = diff_sampling(p, input_space, method)
        samples.append(feature)  # shape (num_spc,)

        # Embed drawn samples and use for conditioning for the next site.
        embs[site] = embedding(samples[-1], in_dim
                               )[:, None, :].expand(-1, num_bins, -1
                                                    ).reshape(num_spc*num_bins, -1)

    # Stack sampled per-site arrays across sites
    samples = torch.stack(tensors=samples, dim=1)  # shape (num_spc, data_dim)
    return samples


def single_class(mps: tk.models.MPS,
                 embedding: str,
                 method: str,
                 cls_pos: int,  # extract this once for the mps, not at every sampling step
                 num_bins: int,
                 num_spc: int,  # per class
                 cls: int,
                 device: torch.device
                 ) -> torch.Tensor:
    """
    Sample data conditioned on a single class label from an MPS model.

    This is a user-facing wrapper around `_single_class` that prepares
    the embedding callable, input grid, and class embedding, then performs
    class-conditional sampling using the specified `method` from `diff_sampling`.

    Parameters
    ----------
    mps : tk.models.MPS
        Trained MPS used for sampling.
    embedding : str
        String identifier of the embedding (resolved via `get_embedding`).
    method : str
        Sampling method name, passed to `diff_sampling`.
    cls_pos : int
        Position of the class site in the MPS.
    num_bins : int
        Number of discretization points for the real-valued input grid.
    num_spc : int
        Number of samples to draw for this class.
    cls : int
        Integer class index (converted to one-hot / basis embedding).
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    torch.Tensor
        Samples of shape `(num_spc, data_dim)` where `data_dim = mps.n_features - 1`.
    """
    mps.to(device)

    # Create the 1D grid (input_space) and instanciate the embedding callable
    rang = _embedding_to_range(embedding)
    embedding = get_embedding(embedding)
    input_space = torch.linspace(rang[0], rang[1], num_bins, device=device)

    in_dim, num_cls = _get_indim_and_ncls(mps)

    # Class basis embedding (one-hot) -> expand to (num_spc * num_bins, num_cls)
    cls_emb = tk.embeddings.basis(torch.tensor(cls), num_cls).float()
    samples = _single_class(mps=mps, embedding=embedding, method=method,
                            cls_pos=cls_pos, cls_emb=cls_emb,
                            in_dim=in_dim, input_space=input_space,
                            num_spc=num_spc, device=device)
    return samples


def _batch(mps: tk.models.MPS,
           embedding: Callable[[torch.Tensor, int], torch.Tensor],
           method: str,
           cls_embs: Sequence[torch.Tensor],  # one embedding per class,
           cls_pos: int,
           in_dim: int,
           num_bins: int,
           input_space: torch.Tensor,
           num_spc: int,  # per class,
           device: torch.device) -> torch.Tensor:
    """
    Internal helper to sample multiple classes in a single call, assuming
    precomputed embeddings and input grid.

    Loops over a list of class embeddings, invoking `_single_class` for each,
    and stacks the results along the class dimension.

    Parameters
    ----------
    mps : tk.models.MPS
        The trained MPS used to compute Born probabilities.
    embedding : Callable
        Callable embedding function.
    method : str
        Sampling method passed to `diff_sampling`.
    cls_embs : Sequence[torch.Tensor]
        Precomputed embeddings for all classes.
    cls_pos : int
        Index of the class site in the MPS.
    in_dim : int
        Input embedding dimension.
    num_bins : int
        Number of candidate values per feature site.
    input_space : torch.Tensor
        Sampling grid over the input variable.
    num_spc : int
        Number of samples to generate per class.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(num_spc, num_cls, data_dim)`.
    """

    samples = []
    for cls_emb in cls_embs:
        cls_samples = _single_class(
            mps=mps, embedding=embedding, cls_pos=cls_pos, in_dim=in_dim,
            cls_emb=cls_emb, num_bins=num_bins, input_space=input_space,
            num_spc=num_spc, device=device, method=method
        )  # shape (num_spc, data_dim)
        samples.append(cls_samples)

    # shape: (n_spc, num_cls, data_dim)
    samples = torch.stack(tensors=samples, dim=1)
    return samples


def batch(mps: tk.models.MPS,
          embedding: str,
          method: str,
          cls_pos: int,  # need to provide this if num_cls == in_dim.
          num_bins: int,
          num_spc: int,  # per class
          device: torch.device) -> torch.Tensor:
    """
    Sample data from all classes using class-conditional MPS sampling.

    This function converts the string embedding identifier into a callable,
    builds the input grid and class embeddings, and calls `_batch`.
    Sampling for each site uses the chosen `diff_sampling` method.

    Parameters
    ----------
    mps : tk.models.MPS
        Trained MPS to sample from.
    embedding : str
        Embedding identifier string.
    method : str
        Sampling method name used by `diff_sampling`.
    cls_pos : int
        Position of the class site in the MPS.
    num_bins : int
        Number of discrete candidate values for the input grid.
    num_spc : int
        Number of samples per class.
    device : torch.device
        Device to run sampling on.

    Returns
    -------
    torch.Tensor
        Sample tensor of shape `(num_spc, num_cls, data_dim)`.
    """
    # Initilization code
    mps.to(device)
    rang = _embedding_to_range(embedding)
    embedding = get_embedding(embedding)
    input_space = torch.linspace(rang[0], rang[1], num_bins)
    in_dim, num_cls = _get_indim_and_ncls(mps)
    cls_embs = []
    for cls in range(num_cls):
        cls_embs.append(tk.embeddings.basis(
            torch.tensor(cls), num_cls).float())

    samples = _batch(
        mps=mps, embedding=embedding, cls_embs=cls_embs,
        cls_pos=cls_pos, in_dim=in_dim, num_bins=num_bins,
        input_space=input_space, num_spc=num_spc,
        method=method, device=device
    )
    return samples


def batched(mps: tk.models.MPS,
            embedding: str,
            method: str,
            cls_pos: int,
            num_spc: int,  # total number of samples per class, ought to be divisible by batch_size? hard coding is bad
            num_bins: int,
            batch_spc: int,  # will be num_spc in mps_sampling # only needed for visualisation
            device: torch.device
            ) -> torch.FloatTensor:
    """
    Memory-efficient sampling wrapper for generating many samples per class.

    Splits the total number of requested samples `num_spc` into smaller batches
    of size `batch_spc`, sampling each sub-batch sequentially via `_batch`.
    Results are concatenated and truncated to the exact target size.

    Parameters
    ----------
    mps : tk.models.MPS
        Trained MPS model.
    embedding : str
        Embedding identifier (converted to callable).
    method : str
        Sampling method name for `diff_sampling`.
    cls_pos : int
        Index of the class site in the MPS.
    num_spc : int
        Total number of samples per class.
    num_bins : int
        Number of discrete grid points for each feature.
    batch_spc : int
        Number of per-class samples per sub-batch.
    device : torch.device
        Device to perform sampling on.

    Returns
    -------
    torch.FloatTensor
        Tensor of shape `(num_spc, num_cls, data_dim)` containing non-embedded samples.
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
    batches = []
    for _ in range(num_batches):  # could compute these batches in parallel, not to bad
        batch = _batch(
            mps=mps, embedding=embedding, method=method,
            cls_embs=cls_embs, cls_pos=cls_pos, in_dim=in_dim,
            num_bins=num_bins, input_space=input_space,
            num_spc=batch_spc, device=device
        )  # (batch_size, num_cls, data.dim) samples
        batch = batch.cpu()
        batches.append(batch)

    # (batch_size x num_batches, num_cls, data.dim)
    samples = torch.concat(tensors=batches, dim=0)
    batches.clear()
    samples = samples[:num_spc, :, :]  # slice to (num_spc, num_cls, data.dim)

    return samples
