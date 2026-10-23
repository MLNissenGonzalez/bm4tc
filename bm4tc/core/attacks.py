"""Evasion (adversarial) attacks against any GenerativeClassifier (bm4tc.core.interface).

Budget convention (see "Vocabulary" in GUIDE.md):
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

from bm4tc.core.graphs import Graphs
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

    def _start(self, model, naturals: torch.Tensor, eps_abs) -> torch.Tensor:
        """The first perturbation: random in the ball and the domain, or zero."""
        if not self.random_start:
            return torch.zeros_like(naturals)
        delta = random_in_ball(naturals.shape, self.norm, eps_abs, naturals.device)
        return self._bounded_delta(delta, naturals, eps_abs, model.input_range)

    def _step(self, model, naturals: torch.Tensor, labels: torch.LongTensor,
              delta: torch.Tensor, eps_abs) -> torch.Tensor:
        """One ascent step on the loss from ``delta``, projected."""
        step_size = self.step_size if self.step_size is not None else 2.5 * eps_abs / self.num_steps
        delta = delta.detach().requires_grad_(True)
        loss = self._loss(model, naturals + delta, labels)
        # The input gradient only: parameter gradients are left untouched, so an
        # attack inside a training step does not disturb the gradient a
        # micro-batched step accumulates (D79).
        (grad,) = torch.autograd.grad(loss, delta)
        delta = delta.detach() + step_size * normalizing(grad, norm=self.norm)
        return self._bounded_delta(delta, naturals, eps_abs, model.input_range)

    def graphed(self, model, graphs: Graphs) -> "GraphedPGD":
        """The attack on ``model`` with each step replayed from a CUDA graph of
        ``graphs`` (eager when they are disabled); see :class:`GraphedPGD`."""
        return GraphedPGD(self, model, graphs)

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
        return self.graphed(model, Graphs(enabled=False))(
            naturals.to(device).detach(), labels.to(device), eps_abs)


class GraphedPGD:
    """A :class:`_PGD` attack on one model, its step replayed from a CUDA graph.

    The loop over steps stays in Python and the graph holds one step: a graph of
    the whole attack holds steps × sites contractions, forward and backward, and
    at 784 sites and 40 steps capturing it crashed the process (segmentation
    fault) or ran out of memory (D105). The start (a random draw or zeros) and
    the final clamp are a few eager kernels. Disabled graphs make this the eager
    attack, number for number. Keep one per model and :class:`Graphs`, so each
    batch shape captures once.
    """

    def __init__(self, attack: _PGD, model, graphs: Graphs):
        self.attack = attack
        self.model = model
        self.step = graphs.wrap(
            lambda naturals, labels, delta, eps_abs:
                attack._step(model, naturals, labels, delta, eps_abs))

    def __call__(self, naturals: torch.Tensor, labels: torch.LongTensor,
                 eps_abs) -> torch.Tensor:
        """Adversarial examples for ``naturals`` (on the model's device);
        ``eps_abs`` a float, or a device scalar when the graphs are enabled."""
        delta = self.attack._start(self.model, naturals, eps_abs)
        for _ in range(self.attack.num_steps):
            delta = self.step(naturals, labels, delta, eps_abs)
        lo, hi = self.model.input_range
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
