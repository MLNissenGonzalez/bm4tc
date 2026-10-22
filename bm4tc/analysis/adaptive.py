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
there, so the attack never does worse than that transfer attack (D98). On a defence
with no purifier and no detector it is plain PGD on the classifier: :class:`BarePGD`,
the attack behind ``rob`` and every transfer number (D149). A random
defence's examples are scored on fresh draws (:meth:`Defense.rates`), not on the
attack's own.

Budgets are absolute (``eps_abs``), as in :mod:`bm4tc.core.attacks`.
"""
from typing import Callable, Optional, Tuple

import torch

from bm4tc.core.attacks import normalizing, project, random_in_ball
from bm4tc.core.graphs import Graphs
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

    def purified(self, x: torch.Tensor, draws: int) -> Optional[torch.Tensor]:
        """``draws`` draws of P(x), stacked (draws, *x.shape), without gradient;
        None without a purifier."""
        if self.purify is None:
            return None
        return torch.stack([self.purify(x.detach()).to(x.device).detach()
                            for _ in range(draws)])

    def _judge(self, x: torch.Tensor, labels: torch.Tensor, purified: Optional[torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For one draw P(x) = ``purified``: (loss to descend, error, accepted) per
        sample. The loss is differentiable in ``x`` (BPDA through P). Error means
        the defence accepts and classifies wrong."""
        z = x
        if purified is not None:
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

    def attack_step(self, x: torch.Tensor, labels: torch.Tensor,
                    purified: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(loss averaged over the draws of P(x) in ``purified``, differentiable in
        x; error rate over them). Without a purifier, one draw. Capturable: P runs
        outside (:meth:`purified`)."""
        draws = [None] if purified is None else purified.unbind(0)
        losses, errors = [], []
        for p in draws:
            loss, error, _ = self._judge(x, labels, p)
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
            with torch.enable_grad():   # an eager likelihood purifier needs input gradients
                p = self.purified(x, 1)
                _, wrong_accepted, accepted = self._judge(x, labels, None if p is None else p[0])
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

    def graphed(self, defense: Defense, graphs: Graphs) -> "GraphedAdaptivePGD":
        """The attack on ``defense`` with each step replayed from a CUDA graph of
        ``graphs`` (eager when they are disabled); see :class:`GraphedAdaptivePGD`."""
        return GraphedAdaptivePGD(self, defense, graphs)

    def generate(self, defense: Defense, x: torch.Tensor, labels: torch.Tensor,
                 eps_abs: float, start: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.graphed(defense, Graphs(enabled=False))(x, labels, eps_abs, start)

    def _bounded(self, defense: Defense, x: torch.Tensor, delta: torch.Tensor, eps_abs):
        lo, hi = defense.model.input_range
        return project((x + delta).clamp(lo, hi) - x, "inf", eps_abs)

    def _step(self, defense: Defense, x, labels, delta, best_x, best_rate, eps_abs,
              purified=None):
        """Score ``x + delta``, keep it where it beats the best so far, and take
        one sign step: (next delta, best_x, best_rate)."""
        step = self.step_size if self.step_size is not None else 2.5 * eps_abs / self.steps
        delta = delta.detach().requires_grad_(True)
        loss, rate = defense.attack_step(x + delta, labels, purified)
        better = rate > best_rate
        best_x = torch.where(better[:, None], (x + delta).detach(), best_x)
        best_rate = torch.where(better, rate, best_rate)
        (grad,) = torch.autograd.grad(loss.sum(), delta)
        delta = delta.detach() - step * normalizing(grad, "inf")
        return self._bounded(defense, x, delta, eps_abs), best_x, best_rate


class BarePGD:
    """PGD on the bare classifier, with restarts (D149): :class:`AdaptivePGD` on a
    :class:`Defense` with no purifier and no detector (the margin loss, ``restarts``
    starts, the best iterate per sample). ``graphed(model, graphs)`` gives the
    attack ``(x, labels, eps_abs, start=None)``, as :class:`GraphedAdaptivePGD`."""

    def __init__(self, steps: int, restarts: int):
        self.attack = AdaptivePGD(steps=steps, restarts=restarts)

    def graphed(self, model, graphs: Graphs) -> "GraphedAdaptivePGD":
        return self.attack.graphed(Defense(model), graphs)


class GraphedAdaptivePGD:
    """:class:`AdaptivePGD` on one defence, its step replayed from a CUDA graph.

    As :class:`bm4tc.core.attacks.GraphedPGD` (D105): the loop stays in Python and
    the graph holds one step (the defence's loss and error rate, the input
    gradient, the sign step, the projection, the best-iterate update). P(x + δ)
    is an input of the step, computed before it (BPDA needs only its value), so
    the graph never holds the purifier's steps; a likelihood purifier replays
    from graphs of its own. The starts and the final clamp are a few eager
    kernels. Disabled graphs make this the eager attack, number for number. Keep
    one per defence and :class:`Graphs`, so each batch shape captures once.
    """

    def __init__(self, attack: AdaptivePGD, defense: Defense, graphs: Graphs):
        self.attack = attack
        self.defense = defense
        self.step = graphs.wrap(
            lambda x, labels, delta, best_x, best_rate, eps_abs, *purified:
                attack._step(defense, x, labels, delta, best_x, best_rate, eps_abs,
                             *purified))

    def __call__(self, x: torch.Tensor, labels: torch.Tensor, eps_abs,
                 start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adversarial examples for ``x`` (on the model's device); ``eps_abs`` a
        float, or a device scalar when the graphs are enabled."""
        a, defense = self.attack, self.defense
        x = x.detach()
        draws = a.eot if defense.random else 1
        best_x = x.clone()
        best_rate = torch.full((len(x),), -1.0, device=x.device)
        for restart in range(a.restarts):
            if restart == 0 and start is not None:
                delta = a._bounded(defense, x, start.to(x.device).detach() - x, eps_abs)
            else:
                delta = a._bounded(defense, x, random_in_ball(x.shape, "inf", eps_abs, x.device),
                                   eps_abs)
            # steps + 1 scorings; the last step's delta is not used
            for _ in range(a.steps + 1):
                purified = defense.purified(x + delta, draws)
                delta, best_x, best_rate = self.step(
                    x, labels, delta, best_x, best_rate, eps_abs,
                    *(() if purified is None else (purified,)))
        lo, hi = defense.model.input_range
        return best_x.clamp(lo, hi)
