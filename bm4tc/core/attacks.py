"""Evasion (adversarial) attacks against any GenerativeClassifier (bm4tc.core.interface).

Budget convention (see "Budget vocabulary" in CLAUDE.md):
    ``eps_rel``  authored fraction of the embedding domain width ``hi - lo``. This is
                 what configs carry, and it equals the budget in the data's own units.
    ``eps_abs``  model-domain value, ``eps_rel * (hi - lo)``. Every attack method in
                 this module takes ``eps_abs`` — conversion happens in the caller
                 (``Trainer``, ``bm4tc/pipeline/analyse.py``) via
                 ``rel_to_abs``.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from bm4tc.core.interface import best_wrong_log_joint, nll

@dataclass
class EvasionConfig:
    method: str = "PGD"
    norm: int | str = "inf"
    eps_rel: list = field(default_factory=lambda: [0.1, 0.3])
    num_steps: int = 10
    step_size: Optional[float] = None
    random_start: bool = True


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


def project(perturbation: torch.Tensor, norm: int | str, radius: float) -> torch.Tensor:
    """Project a (batch, dim) perturbation onto the Lp ball of ``radius``."""
    if norm == "inf":
        return perturbation.clamp(-radius, radius)
    elif isinstance(norm, int):
        norms = perturbation.norm(p=norm, dim=1, keepdim=True)
        scale = torch.clamp(norms / radius, min=1.0)
        return perturbation / scale
    else:
        raise ValueError(f"{norm=}, but expected int or 'inf'.")


def random_in_ball(shape: torch.Size, norm: int | str, radius: float,
                   device: torch.device) -> torch.Tensor:
    """A random perturbation inside the Lp ball of ``radius`` (uniform for L∞;
    for Lp a random direction times a uniform radius, not uniform in volume)."""
    if norm == "inf":
        return (2 * torch.rand(shape, device=device) - 1) * radius
    elif isinstance(norm, int):
        delta = torch.randn(shape, device=device)
        return normalizing(delta, norm) * radius * torch.rand(shape[0], 1, device=device)
    else:
        raise ValueError(f"{norm=}, but expected int or 'inf'.")


class _PGD:
    """Projected gradient ascent on ``_loss`` within the eps ball and the input
    domain. Subclasses define only the loss."""

    def __init__(
            self,
            norm: int | str = "inf",
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True,
    ):
        self.norm = norm
        self.num_steps = num_steps if num_steps is not None else 10
        self.step_size = step_size
        self.random_start = random_start

    def _loss(self, model, x: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    def _bounded_delta(
            self,
            perturbation: torch.Tensor,
            naturals: torch.Tensor,
            eps_abs: float,
            input_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Project onto both the valid input domain and the epsilon ball."""
        lo, hi = input_range
        in_domain = (naturals + perturbation).clamp(lo, hi) - naturals
        return project(in_domain, self.norm, eps_abs)

    def generate(
            self,
            model,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            eps_abs: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples.

        ``eps_abs`` is an absolute model-domain budget, not a fraction.
        """
        model.to(device)
        naturals = naturals.to(device).detach()
        labels = labels.to(device)

        step_size = self.step_size if self.step_size is not None else 2.5 * eps_abs / self.num_steps

        if self.random_start:
            delta = random_in_ball(naturals.shape, self.norm, eps_abs, device)
            delta = self._bounded_delta(delta, naturals, eps_abs, model.input_range)
        else:
            delta = torch.zeros_like(naturals)

        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            loss = self._loss(model, naturals + delta, labels)
            # The input gradient only: parameter gradients are left untouched, so an
            # attack inside a training step does not disturb the gradient a
            # micro-batched step accumulates (D79).
            (grad,) = torch.autograd.grad(loss, delta)
            delta = delta.detach() + step_size * normalizing(grad, norm=self.norm)
            delta = self._bounded_delta(delta, naturals, eps_abs, model.input_range)

        lo, hi = model.input_range
        return (naturals + delta).clamp(lo, hi).detach()


class ProjectedGradientDescent(_PGD):
    """PGD maximising the discriminative NLL -log p(c|x)."""

    def _loss(self, model, x, labels):
        return nll(model, x, labels)


class JointProjectedGradientDescent(_PGD):
    """PGD maximising max_{c'≠c} log p(x̃, c')  (joint generative attack).

    The worst-case wrong class is re-selected dynamically at every gradient step.
    """

    def _loss(self, model, x, labels):
        return best_wrong_log_joint(model, x, labels)


_METHOD_MAP = {
    "PGD":       ProjectedGradientDescent,
    "JOINT_PGD": JointProjectedGradientDescent,
}


def build_attack(
    evasion_cfg: EvasionConfig,
) -> ProjectedGradientDescent | JointProjectedGradientDescent:
    """Construct an attack object from an EvasionConfig."""
    if evasion_cfg.method not in _METHOD_MAP:
        raise ValueError(f"Unknown attack method: {evasion_cfg.method!r}. "
                         "Expected 'PGD' or 'JOINT_PGD'.")
    return _METHOD_MAP[evasion_cfg.method](
        norm=evasion_cfg.norm,
        num_steps=evasion_cfg.num_steps,
        step_size=evasion_cfg.step_size,
        random_start=evasion_cfg.random_start,
    )
