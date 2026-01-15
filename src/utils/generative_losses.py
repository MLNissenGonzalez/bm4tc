"""
Abstract base classes for generative NLL losses.

Users must subclass GenerativeNLL and implement both abstract methods
to define their normalization approach for computing p(x|c).
"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import TYPE_CHECKING
import tensorkrowch as tk

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
        (because on internal tensorkrowch stuff) to compute the squared amplitude for each (data, label) pair.

        Args:
            bornmachine: BornMachine instance with generator view.
            data: Input tensor of shape (batch_size, data_dim).
            labels: Class labels of shape (batch_size,).

        Returns:
            Tensor of shape (batch_size,) with log unnormalized probabilities.
        """
        data_embs = bornmachine.embedding(data, bornmachine.in_dim) # (batch_size, data_dim, physical_dim)
        class_embs = tk.embeddings.basis(labels, bornmachine.out_dim) # (batch_size, num_classes)
        embs = {}
        # TODO: Can this be vectorized?
        for site in range(bornmachine.num_sites):
            if site == bornmachine.out_position:
                embs[site] = class_embs # (batch_size, num_classes)
            else:
                offset = int(site > bornmachine.out_position)
                embs[site] = data_embs[:, site - offset, :] # (batch_size, physical_dim)
        # TODO: Check with José if I have to use sequential method here because num_classes neq physical_dim
        unnormalized_probs = bornmachine.generator.sequential(embs) # batch_size
        return torch.log(unnormalized_probs + self.eps)

    @abstractmethod
    def compute_log_partition(
            self,
            bornmachine: "BornMachine"
    ) -> torch.Tensor:
        """
        Compute log(Z).

        The partition function Z = sum_x,c |psi(x,c)|^2 normalizes the joint Born distribution. 
        Basically just the norm of the MPS squared (Full contraction of all tensors with themselves).

        Args:
            bornmachine: BornMachine instance

        Returns:
            Tensor of shape (1,) with log partition function values.
        """
        # TODO: Problem: reset changes parameters of bornmachine.generator! 
        #       For gradient descent, gradients need to pass through here. Talk with José about this.
        #       1. I could take two optimization steps, one after log_unnorm and one after log_Z, but could be inefficient.
        #       2. I could write the contraction myself without renaming of the nn.Parameters, but could be tedious. 

        bornmachine.generator.reset()
        Z = bornmachine.generator.norm() # this takes the square root of the full contraction (need to  be squared)
        bornmachine.generator.reset()

        return torch.log(Z + self.eps)

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
