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

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------Classification------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

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

class ClassificationBrier(nn.Module):
    """
    Brier Score Loss for MPS classification using Born rule probabilities.

    Computes the mean squared error between Born-rule class probabilities and
    one-hot targets — a bounded proper scoring rule ∈ [0, 2]:

        BS = mean_i  Σ_c (p(c|x_i) - y_{i,c})²

    where y_{i,c} = 1 if c == t_i else 0.

    Compared to the spherical NLL, gradients ∂BS/∂p_c = 2(p_c - y_c) are bounded,
    avoiding instability when probabilities are near zero.
    """

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(p)
        y.scatter_(1, t.unsqueeze(1), 1.0)
        return ((p - y) ** 2).sum(dim=1).mean()


class ClassificationSoftmaxNLL(nn.Module):
    """
    Thin wrapper around nn.CrossEntropyLoss for softmax-based MPS classifiers.

    Expects raw MPS amplitudes ψ (signed, unnormalized) as logits — NOT
    Born-rule probabilities. Delegates to nn.CrossEntropyLoss, which applies
    log-softmax internally (numerically stable fused kernel).

    IMPORTANT: Must be used with experiments/softmax_sanity.py, which calls
    bm.classifier.amplitudes() instead of bm.class_probabilities().
    """
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, amplitudes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._loss(amplitudes, t)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------Generative------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------


# This uses logs to improve numerical stability which is not necessary for class probabilities.
class GenerativeNLL(nn.Module):
    """
    Generative NLL loss for joint distribution p(x,c).

    Computes: -log(p(x,c)) = -log(|psi(x,c)|^2) + log(Z)

    where:
    - |psi(x,c)|^2 is the unnormalized probability from the MPS
    - Z is the global partition function (normalization constant)

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
        Compute mean NLL: -mean(log(p(x,c))).

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
        log_Z: torch.Tensor = born.log_partition_function()
        if not torch.isfinite(log_Z):
            raise RuntimeError(
                f"log_partition_function returned non-finite value: {log_Z.item():.4g}. "
                "MPS has collapsed or exploded."
            )
        log_prob = log_unnorm - log_Z
        return -log_prob.mean()


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------Mixed------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

class MixedNLL(nn.Module):
    """
    Compute (1-alpha)*ClassificationNLL + alpha*GenerativeNLL

    Alpha convention:
        alpha=0  →  pure classification loss
        alpha=1  →  pure generative (NLL) loss

    Derivation:
        = ClassificationNLL - alpha * ln(p(x)).mean()
        = Sum_(x,c) -ln(bm(x,c)) + ln(sum_c' bm(x,c'))
          + alpha * (bm.log_partition() - ln(sum_c' bm(x,c')))
        = Sum_(x,c) -ln(bm(x,c)) + (1-alpha)*ln(sum_c' bm(x,c')) + alpha*bm.log_partition()

    where bm(x,c) = tn(x,c).abs_square() is the unnormalized joint probability.
    """

    def __init__(self, eps: float = 1e-12, alpha: float = 0.1):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, bm: "BornMachine", data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gen = bm.generator
        log_Z: torch.Tensor = gen.log_partition_function()
        if not torch.isfinite(log_Z):
            raise RuntimeError(
                f"log_partition_function returned non-finite value: {log_Z.item():.4g}. "
                "MPS has collapsed or exploded."
            )

        # Compute |ψ(x,c)|² for all C classes via C generator forward passes
        # unnorm shape: (batch, num_classes)
        unnorm = torch.stack(
            [gen.unnormalized_prob(data, torch.full_like(labels, c))
             for c in range(bm.out_dim)],
            dim=1
        )

        # Term 1: -log|ψ(x,c_true)|²
        log_unnorm_joint = torch.log(
            unnorm[torch.arange(len(labels)), labels].clamp(min=self.eps)
        )

        # Term 2: log Σ_c |ψ(x,c)|²  (marginal, needed for classification component)
        log_unnorm_marginal = torch.log(unnorm.sum(dim=1).clamp(min=self.eps))

        loss = -log_unnorm_joint + (1 - self.alpha) * log_unnorm_marginal + self.alpha * log_Z
        return loss.mean()



class NormRegularizer(nn.Module):
    """
    Partition-function norm regularization penalty (trainer-level, not a criterion).

    Computes  strength * (Z - target)²  where Z = exp(log_partition_function()).

    Not in _GENERATIVE_LOSSES — instantiated directly by GenerativeTrainer, so that
    PerformanceEvaluator remains unaffected by the regularization term.

    Parameters
    ----------
    strength : float
        Regularization coefficient.
    target : float
        Target value for the partition function Z (norm² of the MPS).
    """

    def __init__(self, strength: float, target: float):
        super().__init__()
        self.strength = strength
        self.target = target

    def forward(self, bornmachine: "BornMachine") -> torch.Tensor:
        """
        Compute the norm regularization penalty.

        Args:
            bornmachine: BornMachine instance.

        Returns:
            Scalar tensor with the penalty value.
        """
        log_Z: torch.Tensor = bornmachine.generator.log_partition_function()
        Z = torch.exp(log_Z)
        return self.strength * (Z - self.target) ** 2


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
    "brier": ClassificationBrier,
    "brierscore": ClassificationBrier,
    "bs": ClassificationBrier,
    "softmaxnll": ClassificationSoftmaxNLL,
    "softmax_nll": ClassificationSoftmaxNLL,
    "softmax": ClassificationSoftmaxNLL,
}
_GENERATIVE_LOSSES = {
    "nll": GenerativeNLL,
    "nlll": GenerativeNLL,
    "negativeloglikelihood": GenerativeNLL,
    "negloglikelihood": GenerativeNLL,
    "mixednll": MixedNLL,
    "mixed": MixedNLL,
    "mixed_nll": MixedNLL,
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
