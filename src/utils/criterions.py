"""
Classes for training criterions suitable for BornMachines which
output amplitudes/probabilities, not logits. 
"""
import torch
from torch import nn
from typing import TYPE_CHECKING
from . import schemas

if TYPE_CHECKING:
    from src.models import BornMachine

# TODO: Think about making both NLL losses share more code or call structures..


class ClassificationNLL(nn.Module):
    """
    Negative Log-Likelihood Loss (custom implementation for MPS models).

    This loss computes the mean negative log-likelihood of the true class probabilities
    predicted by an MPS classifier. It is equivalent to the categorical cross-entropy
    loss for one-hot targets, but implemented explicitly to ensure numerical stability
    and control over small-value clamping.

    Parameters
    ----------
    eps : float, optional
        Small positive constant used to clamp probabilities from below before
        applying the logarithm to prevent numerical underflow. Default is 1e-12.

    Forward Pass
    -------------
    Given predicted probabilities `p` of shape (batch_size, num_classes) and integer
    targets `t` of shape (batch_size,), the loss is computed as:

        L = -mean( log( p[i, t[i]] ) )

    Returns a scalar tensor representing the average NLL across the batch.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = p.clamp(min=self.eps)
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()

# This uses logs to improve numerical stability which is not necessary for class probabilities.


class GenerativeNLL(nn.Module):
    """
    Abstract base for generative NLL losses.

    Computes: -log(p(x|c)) = -log(|psi(x,c)|^2) + log(Z_c)

    where:
    - |psi(x,c)|^2 is the unnormalized probability from the MPS
    - Z_c is the partition function (normalization constant) for class c

    User must subclass and implement both abstract methods with their
    chosen normalization approach.

    Args:
        eps: Small constant for numerical stability in log computations.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(
            self,
            bornmachine: "BornMachine",
            data: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean NLL: -mean(log(p(x|c))).

        Args:
            bornmachine: BornMachine instance.
            data: Input tensor of shape (batch_size, data_dim).
            labels: Class labels of shape (batch_size,).

        Returns:
            Scalar tensor with mean negative log-likelihood.
        """
        born = bornmachine.generator
        log_unnorm = torch.log(
            born.unnormalized_prob(data, labels)
            .clamp(min=self.eps)
        )
        log_Z: torch.Tensor = born.log_partition_function(labels)
        log_prob = log_unnorm - log_Z
        return -log_prob.mean()


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# ---------API-----------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
_CLASSIFICATION_LOSSES = {
    "nll": ClassificationNLL,
    "nlll": ClassificationNLL,
    "negativeloglikelihood": ClassificationNLL,
    "negloglikelihood": ClassificationNLL,
}
_GENERATIVE_LOSSES = {
    "nll": GenerativeNLL,
    "nlll": GenerativeNLL,
    "negativeloglikelihood": GenerativeNLL,
    "negloglikelihood": GenerativeNLL
}

_LOSS_MAP = {
    "classification": _CLASSIFICATION_LOSSES,
    "classifiy": _CLASSIFICATION_LOSSES,
    "generate": _GENERATIVE_LOSSES,
    "generative": _GENERATIVE_LOSSES
}


def criterion(mode: str, cfg: schemas.CriterionConfig) -> nn.Module:
    """
    Instantiates a loss function based on the configuration specification.

    The function looks up a registered loss in `_LOSS_MAP` using a normalized
    version of the name provided in `cfg.name`. It supports flexible name
    variants (case- and delimiter-insensitive) and automatically handles
    optional keyword arguments.

    Parameters
    ----------
    cfg : schemas.CriterionConfig
        Configuration object specifying:
        - `name`: name or alias of the desired loss function (e.g. "nll", "bce").
        - `kwargs`: optional dictionary of keyword arguments for initialization.

    Returns
    -------
    nn.Module
        A PyTorch loss module ready to be used for training.

    Raises
    ------
    ValueError
        If the specified loss name does not match any entry in `_LOSS_MAP`.

    Notes
    -----
    - The `MPSNLLL` loss is tailored for MPS-based classifiers with discrete labels.
    - The `BCELoss` is included for binary tasks with probabilistic outputs.
    - Unrecognized loss names will raise an explicit error.
    """
    mode_key = mode.lower().replace(" ", "").replace("-", "")
    if mode_key not in _LOSS_MAP:
        raise KeyError(f"Training mode {mode} not recognised.")
    OPTIONS = _LOSS_MAP[mode_key]
    loss_key = cfg.name.replace(" ", "").replace("-", "").lower()
    if loss_key not in OPTIONS:
        raise ValueError(f"Loss '{cfg.name}' not recognised")
    # use empty dict if kwargs is None
    return OPTIONS[loss_key](**(cfg.kwargs or {}))
