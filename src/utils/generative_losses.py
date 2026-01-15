"""
Abstract base classes for generative NLL losses.

Users must subclass GenerativeNLL and implement both abstract methods
to define their normalization approach for computing p(x|c).
"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import BornMachine


class GenerativeNLL(nn.Module, ABC):
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

    @abstractmethod
    def compute_unnormalized_log_prob(
            self,
            bornmachine: "BornMachine",
            data: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log(|psi(x,c)|^2) for each sample.

        This should use the generator's sequential method with full contraction
        to compute the squared amplitude for each (data, label) pair.

        Args:
            bornmachine: BornMachine instance with generator view.
            data: Input tensor of shape (batch_size, data_dim).
            labels: Class labels of shape (batch_size,).

        Returns:
            Tensor of shape (batch_size,) with log unnormalized probabilities.
        """
        pass

    @abstractmethod
    def compute_log_partition(
            self,
            bornmachine: "BornMachine",
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log(Z_c) for each class in the batch.

        The partition function Z_c = sum_x |psi(x,c)|^2 normalizes the
        Born distribution. Implement using your chosen approach:
        - Exact enumeration (tractable for MPS via tensor contraction)
        - Importance sampling
        - Variational bounds

        Args:
            bornmachine: BornMachine instance.
            labels: Class labels of shape (batch_size,).

        Returns:
            Tensor of shape (batch_size,) with log partition function values.
        """
        pass

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
        log_unnorm = self.compute_unnormalized_log_prob(bornmachine, data, labels)
        log_Z = self.compute_log_partition(bornmachine, labels)
        log_prob = log_unnorm - log_Z
        return -log_prob.mean()
