# Likelihood-based purification for Born Machines

import torch
from typing import Tuple


def normalizing(x: torch.FloatTensor, norm: int | str):
    """
    Normalize a tensor of shape (batch size, data dim)
    along the data dim (flattened).
    """
    if norm == "inf":
        normalized = x.sign()

    elif isinstance(norm, int):
        if norm < 1:
            raise ValueError("Only accept p >= 1.")
        x_norm = x.norm(p=norm, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-12)
        normalized = x / x_norm

    else:
        raise ValueError(f"{norm=}, but expected to be int or 'inf'.")

    return normalized


class LikelihoodPurification:
    """
    Purify adversarial examples by maximizing marginal log-likelihood
    within a perturbation ball around the input.

    Uses gradient descent on the negative marginal log-probability
    (i.e., ascent on log p(x)) with projection back onto the Lp ball,
    analogous to PGD but in reverse direction.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            num_steps: int = 20,
            step_size: float | None = None,
            random_start: bool = False,
            eps: float = 1e-12,
    ):
        """
        Initialize purification.

        Args:
            norm: Lp norm for perturbation ball ("inf" or int >= 1).
            num_steps: Number of gradient descent iterations.
            step_size: Step size per iteration. If None, defaults to
                2.5 * radius / num_steps.
            random_start: Whether to start from random point within
                the radius ball.
            eps: Clamping floor for numerical stability in log p(x).
        """
        self.norm = norm
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.eps = eps

    def _project(self, perturbation: torch.Tensor, radius: float) -> torch.Tensor:
        """Project perturbation back into the Lp ball."""
        if self.norm == "inf":
            return perturbation.clamp(-radius, radius)
        elif isinstance(self.norm, int):
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / radius, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, radius: float, device: torch.device) -> torch.Tensor:
        """Initialize random perturbation within the Lp ball."""
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * radius
        elif isinstance(self.norm, int):
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * radius * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def purify(
            self,
            born,
            data: torch.Tensor,
            radius: float,
            device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purify inputs by gradient descent on marginal NLL within an Lp ball.

        Moves the input towards higher likelihood regions of the learned
        distribution, staying within a radius of the original input.

        Args:
            born: BornMachine instance (must have marginal_log_probability).
            data: Input tensor of shape (batch_size, data_dim).
            radius: Maximum perturbation radius.
            device: Torch device.

        Returns:
            Tuple of:
                - purified: Purified inputs, shape (batch_size, data_dim).
                - log_px: Marginal log-probabilities of purified inputs,
                  shape (batch_size,).
        """
        born.to(device)
        data = data.to(device).detach()
        input_range = born.input_range

        step_size = self.step_size if self.step_size is not None else 2.5 * radius / self.num_steps

        # Initialize perturbation
        if self.random_start:
            delta = self._random_init(data.shape, radius, device)
        else:
            delta = torch.zeros_like(data)

        # Iterative gradient descent on NLL (= gradient ascent on log p(x))
        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            x_tilde = (data + delta).clamp(input_range[0], input_range[1])

            nll = -born.marginal_log_probability(x_tilde, eps=self.eps).mean()

            born.classifier.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()

            nll.backward()

            grad = delta.grad.detach()
            normalized_gradient = normalizing(grad, norm=self.norm)

            # Gradient descent on NLL (subtract, not add)
            delta = delta.detach() - step_size * normalized_gradient
            # Project back into Lp ball
            delta = self._project(delta, radius)

        # Final purified samples, clamped to input range
        purified = (data + delta).clamp(input_range[0], input_range[1]).detach()

        # Compute final log p(x) for the purified samples
        with torch.no_grad():
            log_px = born.marginal_log_probability(purified, eps=self.eps)

        return purified, log_px
