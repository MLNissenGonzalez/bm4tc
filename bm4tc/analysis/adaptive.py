"""Adaptive attacks: PGD against the whole defence, purifier and detector included (D96).

The theory is in docs/adaptive.md §1 and §3. A :class:`Defense` is a purifier P, the
model's classifier f and a detector that accepts iff log p(·) ≥ τ, read on the input or
on P(x). Its output is ⊥ (reject) or f(P(x)). :class:`AdaptivePGD` attacks it:

- through the purifier with BPDA: the real P forward, the identity backward (exact almost
  everywhere for L∞ sign-step likelihood purification; the pathwise gradient for Gibbs and
  SGLD), averaged over ``eot`` draws when P is random;
- through the detector with selective PGD: per sample, a step up log p while rejected and
  down the class margin once accepted.

It keeps the best iterate per sample over every step and restart. Given a ``start``
(the pipeline passes the PGD example on the classifier), the first restart begins
there, so the attack never does worse than that transfer attack (D98). A random
defence's examples are scored on fresh draws (:meth:`Defense.rates`), not on the
attack's own.

Budgets are absolute (``eps_abs``), as in :mod:`bm4tc.core.attacks`.
"""
from typing import Callable, Optional, Tuple

import torch

from bm4tc.core.attacks import normalizing, project, random_in_ball
from bm4tc.core.interface import log_px


class Defense:
    """h(x) = ⊥ if the detector rejects, else f(P(x)).

    Args:
        model: a GenerativeClassifier (bm4tc.core.interface).
        purify: P, a function of a batch returning the purified batch (no gradient
            needed); None is the identity.
        threshold: the detector's τ (accept iff log p ≥ τ); None is no detector.
        detect_on: where the detector reads log p: ``"input"`` or ``"purified"``.
        random: P draws random numbers, so outcomes are probabilities over draws.
    """

    def __init__(self, model, purify: Optional[Callable] = None,
                 threshold: Optional[float] = None, detect_on: str = "input",
                 random: bool = False):
        if detect_on not in ("input", "purified"):
            raise ValueError(f"detect_on must be 'input' or 'purified', got {detect_on!r}")
        self.model = model
        self.purify = purify
        self.threshold = threshold
        self.detect_on = detect_on
        self.random = random

    def _draw(self, x: torch.Tensor, labels: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One draw of the defence: (loss to descend, error, accepted) per sample.

        The loss is differentiable in ``x`` (BPDA through P). Error means the defence
        accepts and classifies wrong.
        """
        z = x
        if self.purify is not None:
            purified = self.purify(x.detach()).to(x.device).detach()
            z = x + (purified - x).detach()      # P forward, identity backward
        log_joint = self.model.log_joint(z)
        true = log_joint.gather(1, labels[:, None]).squeeze(1)
        others = log_joint.scatter(1, labels[:, None], float("-inf"))
        margin = true - others.max(dim=1).values  # log p(y|z) - max_{c≠y} log p(c|z)
        wrong = log_joint.argmax(dim=1) != labels
        if self.threshold is None:
            return margin, wrong, torch.ones_like(wrong)
        lp = log_px(self.model, x if self.detect_on == "input" else z)
        accepted = lp >= self.threshold
        loss = torch.where(accepted, margin, self.threshold - lp)
        return loss, wrong & accepted, accepted

    def attack_step(self, x: torch.Tensor, labels: torch.Tensor, draws: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(loss averaged over ``draws``, differentiable in x; error rate over them)."""
        losses, errors = [], []
        for _ in range(draws):
            loss, error, _ = self._draw(x, labels)
            losses.append(loss)
            errors.append(error.float())
        return torch.stack(losses).mean(0), torch.stack(errors).mean(0).detach()

    def rates(self, x: torch.Tensor, labels: torch.Tensor, draws: int = 1
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per sample, the probability that h(x) = y and that h(x) is a wrong class,
        estimated over ``draws`` draws (one if P is deterministic)."""
        draws = draws if self.random else 1
        right, error = torch.zeros(len(x), device=x.device), torch.zeros(len(x), device=x.device)
        for _ in range(draws):
            with torch.enable_grad():   # a likelihood purifier needs input gradients
                _, wrong_accepted, accepted = self._draw(x, labels)
            right += (accepted & ~wrong_accepted).float()
            error += wrong_accepted.float()
        return right / draws, error / draws


class AdaptivePGD:
    """L∞ PGD on :meth:`Defense.attack_step` within the ε ball and the input box, from
    ``restarts`` starts (the given ``start`` first, if any, then random ones), keeping
    per sample the iterate with the highest error rate (the first one reaching it)."""

    def __init__(self, steps: int = 100, restarts: int = 5, eot: int = 16,
                 step_size: Optional[float] = None):
        self.steps, self.restarts, self.eot, self.step_size = steps, restarts, eot, step_size

    def generate(self, defense: Defense, x: torch.Tensor, labels: torch.Tensor,
                 eps_abs: float, start: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.detach()
        lo, hi = defense.model.input_range
        draws = self.eot if defense.random else 1
        step = self.step_size if self.step_size is not None else 2.5 * eps_abs / self.steps

        def bounded(delta):
            return project((x + delta).clamp(lo, hi) - x, "inf", eps_abs)

        best_x = x.clone()
        best_rate = torch.full((len(x),), -1.0, device=x.device)
        for restart in range(self.restarts):
            if restart == 0 and start is not None:
                delta = bounded(start.to(x.device).detach() - x)
            else:
                delta = bounded(random_in_ball(x.shape, "inf", eps_abs, x.device))
            for t in range(self.steps + 1):
                delta.requires_grad_(True)
                loss, rate = defense.attack_step(x + delta, labels, draws)
                better = rate > best_rate
                best_x[better] = (x + delta).detach()[better]
                best_rate[better] = rate[better]
                if t == self.steps:
                    break
                (grad,) = torch.autograd.grad(loss.sum(), delta)
                delta = bounded(delta.detach() - step * normalizing(grad, "inf"))
        return best_x.clamp(lo, hi)
