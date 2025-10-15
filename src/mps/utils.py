import torch
import tensorkrowch as tk
from typing import Dict, Tuple, Callable
import torch.nn as nn
import wandb
import numpy as np


import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------Born rule. Sequential and parallel code.------------------------------------------------------------------------------------------------------------------
# ------Could add this maybe as method in a custom MPS class.--------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------torch.optim.Optimizer------------------------------------------------------------

def born_sequential(mps: tk.models.MPS,
                    embs: Dict[int, torch.Tensor] | torch.Tensor) -> torch.Tensor:  # very flexible, not parallizable
    """
    Sequential contraction of an MPS with embedded input and computation of probabilities
    using the Born rule. Used primarily for **sampling**, since it supports marginalization
    over unobserved variables.

    Behavior
    --------
    - **Partial inputs (marginalized case):**
      When not all sites are included in `embs`, i.e. p({x_i}_I | {x_j}_J),
      the function marginalizes over missing sites by computing a reduced density
      matrix ρ. The diagonal entries of ρ represent unnormalized marginal probabilities.

    - **Complete inputs (full contraction):**
      When all sites are present, the function computes the amplitude ψ(x₁,…,x_D)
      and returns its squared magnitude |ψ|² as the unnormalized joint distribution.

    Parameters
    ----------
    mps : tk.models.MPS
        The tensor network representing the probability amplitude ψ(x).
    embs : dict[int, torch.Tensor]
        A mapping from site index → embedded tensor of shape (batch_size, phys_dim).
        The keys specify which input sites are provided. Must be a dict, not a tensor.

    Returns
    -------
    torch.Tensor
        The unnormalized probability distribution(s):
        - (batch_size, num_bins) if marginalized,
        - (batch_size, 1) if all variables included.

    Raises
    ------
    TypeError
        If `embs` is not a dictionary.
    """

    if not isinstance(embs, dict):
        raise TypeError(
            "embs input must be a dictionary with site indices as keys and "
            "embedded tensors as values."
        )

    # Identify which features are fed into the MPS.
    mps.in_features = [i for i in embs.keys()]
    in_tensors = [embs[i] for i in mps.in_features]

    # Case 1: Not all variables appear, thus marginalize_output=True and one has to take the diagonal.
    if len(in_tensors) < mps.n_features:
        rho = mps.forward(in_tensors, marginalize_output=True,
                          inline_input=True, inline_mats=True)  # density matrix
        p = torch.diagonal(rho)
    # Case 2: All variables appear.
    else:
        amplitude = mps.forward(
            data=in_tensors, inline_input=True, inline_mats=True)  # prob. amplitude
        p = torch.square(amplitude)

    # logger.debug(f"born_sequential output {p.shape=}")
    return p


def born_parallel(mps: tk.models.MPSLayer | tk.models.MPS,  # could use MPSLayer class for this one actually
                  embs: torch.Tensor) -> torch.Tensor:
    """
    Parallel contraction of an MPS with embedded input and computation of Born probabilities.
    Used primarily for **classification** or **joint probability estimation** over all inputs.

    Behavior
    --------
    - **Classification:**  
      When `mps.n_features = D + 1`, the last site corresponds to the class index.
      The function computes unnormalized conditional probabilities p(c | x₁,…,x_D) and normalizes
      them across classes.

    - **Joint distribution:**  
      When `mps.n_features = D`, the network has no output site, and the function
      returns the unnormalized joint Born probability p(x₁,…,x_D) ∝ |ψ(x)|².

    Parameters
    ----------
    mps : tk.models.MPSLayer | tk.models.MPS
        The MPS model defining the quantum-like probability amplitude ψ(x).
    embs : torch.Tensor
        Embedded inputs with shape (batch_size, D, phys_dim).

    Returns
    -------
    torch.Tensor
        - (batch_size, num_cls): Unnormalized conditional probabilities p(c | x)
        - (batch_size, 1): Unnormalized joint probabilities p(x)

    Raises
    ------
    TypeError
        If `embs` is not a `torch.Tensor`.
    """

    # embs=tensor (parallizable), assume mps.out_feature = [cls_pos] globally??
    if not isinstance(embs, torch.Tensor):
        raise TypeError(
            "embs input must be a tensor of shape (batch_size, D, phys_dim).")

    is_joint = isinstance(mps, tk.models.MPS) and (
        mps.n_features == embs.shape[1])
    # Case 1: p(c | x₁,…,x_D), since n_features = D+1 (with output site)
    if not is_joint:
        p = torch.square(mps.forward(data=embs))
        p = p / p.sum(dim=-1, keepdim=True)
    # Case 2: p(x₁,…,x_D)
    else:
        p = torch.square(mps.forward(data=embs)).unsqueeze(
            1)  # Z x p(x_1,..., x_D) # shape (batch_size, 1)
    # logger.debug(f"born_parallel output: {p.shape=}")
    return p


# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ---------------Getter functions for MPS-----------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

def legendre_embedding(data: torch.Tensor, degree: int = 2, axis: int = -1):
    """
    Compute Legendre polynomial embedding of input data.

    Generates Legendre polynomials up to the specified degree for each
    element along the given axis, stacking them along the same axis.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of arbitrary shape, e.g., (batch_size, n_features).
        Each element is a scalar value in [-1, 1] (typical for Legendre polynomials).
    degree : int, default=2
        Maximum degree of Legendre polynomials.
    axis : int, default=-1
        Axis along which to stack polynomial values. Can be negative.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(degree+1, ...)` where polynomials are stacked along
        `axis`. Output dtype matches input `data.dtype`.

    Notes
    -----
    Uses the standard recursive formula:
        P_0(x) = 1
        P_1(x) = x
        P_n(x) = ((2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n
    """

    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')

    polies = [1, data]
    for i in range(2, degree + 1):
        p_i = ((2*i-1) * data * polies[-1] - (i-1)*polies[-2]) / i
        polies.append(p_i)
    return torch.stack(polies, dim=axis)


_EMBEDDING_MAP = {
    "fourier": tk.embeddings.fourier,
    "poly": tk.embeddings.poly,
    "polynomial": tk.embeddings.poly,
    "legendre": legendre_embedding
}


def get_embedding(name: str) -> Callable[[torch.FloatTensor, int], torch.FloatTensor]:
    """
    Retrieve embedding function by name.

    Parameters
    ----------
    name : str
        Embedding identifier (case-insensitive). Valid options:
        - "fourier"
        - "poly" or "polynomial"
        - "legendre"

    Returns
    -------
    callable
        Embedding function that maps a tensor to its embedding.
        Signature: `tensor -> tensor`. E.g., `legendre_embedding`.

    Raises
    ------
    ValueError
        If the embedding name is not recognized.

    Notes
    -----
    The returned function may expect inputs within specific ranges,
    defined in `_EMBEDDING_TO_RANGE`.
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


def _get_indim_and_ncls(mps: tk.models.MPS) -> Tuple[int, int]:
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
        raise ValueError(
            "Can only handle same dimensional input embedding dimensions.")
    in_dim = val[np.argmax(counts)]
    if len(val) == 2:
        n_cls = val[val != in_dim][0]
    else:
        n_cls = in_dim
    return int(in_dim), int(n_cls)


# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ---------------Logger functions MPS-----------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

def log_grads(mps: tk.models.MPS, step: int, watch_freq: int, stage: str):
    """
    Logs gradient statistics of all MPS parameters to Weights & Biases (wandb).

    The function periodically records absolute gradient histograms for each trainable
    tensor in the MPS. This is useful for monitoring training stability, detecting
    vanishing or exploding gradients, and diagnosing optimizer behavior.

    Logging occurs only every `watch_freq` steps.

    Parameters
    ----------
    mps : tk.models.MPS
        The MPS model whose gradients are logged.
    step : int
        Current global training step.
    watch_freq : int
        Frequency (in steps) at which gradient statistics are recorded.
    stage : str
        Label for the current training phase (e.g., "pre", "gan").
        Used as a prefix in the wandb metric names.

    Notes
    -----
    - Gradients are detached and moved to CPU before histogram computation.
    - Parameters without gradients at the current step are marked with
      `{stage}_mps_abs_grad/{name}/has_grad = 0`.
    - To reduce logging overhead, scalar summaries (mean/std/max) are currently disabled
      but can be re-enabled by uncommenting the respective lines.
    """
    if step % watch_freq != 0:
        return

    log_grads = {}
    for name, tensor in mps.named_parameters():
        if tensor.grad is not None:
            g = tensor.grad.detach().abs().cpu().numpy()
            log_grads[f"{stage}_mps_abs_grad/{name}/hist"] = wandb.Histogram(
                np_histogram=np.histogram(a=g, bins=64)
            )
            # log_grads[f"{stage}_mps_abs_grad/{name}/mean"] = g.mean()
            # log_grads[f"{stage}_mps_abs_grad/{name}/std"] = g.std()
            # log_grads[f"{stage}_mps_abs_grad/{name}/max"] = g.max()
            # log_grads[f"{stage}_mps_abs_grad/{name}/has_grad"] = 1
        else:
            log_grads[f"{stage}_mps_abs_grad/{name}/has_grad"] = 0

    wandb.log(log_grads)


# Use this only for debugging
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


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# --------ENSEMBLE MPS METHOD---------------------------------------------------------------------------------------------
# --------I MAY SWITCH TO THIS METHOD IF IT IS BETTER---------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

# Instead of a single MPS with a central cls tensor, one could train an ensemble of tensors
# interacting with each other only through the (classification) loss
# Conditioning implies what?


# TODO: Implement ensemble method (ADDED AS ISSUE)
