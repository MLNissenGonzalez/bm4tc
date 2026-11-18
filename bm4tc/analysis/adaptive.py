"""Adaptive attacks: PGD against the whole defence, purifier and detector included (D96).

The theory is in notes:adaptive §1 and §3. A :class:`Defense` is a purifier P, the
model's classifier f and a detector that accepts iff log p(·) ≥ τ, read on the input or
on P(x). Its output is ⊥ (reject) or f(P(x)). :class:`AdaptivePGD` attacks it:

- through the purifier with BPDA: the real P forward, the identity backward (exact almost
  everywhere for L∞ sign-step likelihood purification; the pathwise gradient for Gibbs and
  SGLD), averaged over ``eot`` draws when P is random;
- through the detector with selective PGD: per sample, a step up log p while rejected and
  down the class margin once accepted.

It keeps the best iterate per sample over every step and restart. Given ``starts``,
the first restarts begin there: the pipeline passes the PGD example on the classifier,
so the attack never does worse than that transfer attack (D98), and against a
detector also a start on the data (:class:`DataStarts`, D156). On a defence
with no purifier and no detector it is plain PGD on the classifier: :class:`BarePGD`,
the attack behind ``rob`` and every transfer number (D149). A random
defence's examples are scored on fresh draws (:meth:`Defense.rates`), not on the
attack's own.

Budgets are absolute (``eps_abs``), as in :mod:`bm4tc.core.attacks`.
"""
import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

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
    ``max(restarts, len(starts))`` starts (the given ``starts`` first, then random
    ones), keeping per sample the iterate with the highest error rate (the first one
    reaching it).

    Args:
        box: the (lo, hi) the iterates stay in; None is the model's ``attack_range``.
        mask: a function of the clean batch x giving 1 where a feature may move and 0
            where it stays exactly at x (broadcastable to x; :func:`image_masks`), so
            every start and step changes only the masked-in features (D193). None
            moves every feature.
    """

    def __init__(self, steps: int = 100, restarts: int = 5, eot: int = 16,
                 step_size: Optional[float] = None,
                 box: Optional[Tuple[float, float]] = None,
                 mask: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.steps, self.restarts, self.eot, self.step_size = steps, restarts, eot, step_size
        self.box, self.mask = box, mask

    def graphed(self, defense: Defense, graphs: Graphs) -> "GraphedAdaptivePGD":
        """The attack on ``defense`` with each step replayed from a CUDA graph of
        ``graphs`` (eager when they are disabled); see :class:`GraphedAdaptivePGD`."""
        return GraphedAdaptivePGD(self, defense, graphs)

    def generate(self, defense: Defense, x: torch.Tensor, labels: torch.Tensor,
                 eps_abs: float, starts: Sequence[torch.Tensor] = ()) -> torch.Tensor:
        return self.graphed(defense, Graphs(enabled=False))(x, labels, eps_abs, starts)

    def _bounded(self, defense: Defense, x: torch.Tensor, delta: torch.Tensor, eps_abs):
        lo, hi = defense.model.attack_range if self.box is None else self.box
        delta = project((x + delta).clamp(lo, hi) - x, "inf", eps_abs)
        # masked last: the clamp and the L∞ projection act per feature, and an inset box
        # (D178) would otherwise move a frozen end pixel by the inset
        return delta if self.mask is None else delta * self.mask(x)

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


def image_masks(specs: Sequence[dict], train: torch.Tensor,
                input_range: Tuple[float, float]) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """The masks of ``specs`` for square images flattened in raster order (D193): each
    spec is a one-key mapping,

    - ``{window: s}``: the centred square whose side is the fraction s of the image's;
    - ``{data: f}``: the pixels off the background in at least the fraction f of
      ``train``;
    - ``{digit: k}``: the pixels of the attacked image itself that are off the
      background, grown by k pixels (k 3×3 max-pools).

    A pixel is background when it lies within 10⁻³ of the range above its lower end
    (MNIST's 0 after rescaling). An attack confined to such a mask leaves the
    background, where a steep density makes every sign step cost log p, untouched:
    against Legendre's detector it found far more than the attack on every pixel
    (notes:evidence, "The detector under an attack confined to the digit").

    Raises ValueError for a spec that is not one known key, or for non-square inputs."""
    n = train.shape[1]
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"image masks need square images; got {n} features")
    lo, hi = input_range
    floor = lo + 1e-3 * (hi - lo)
    masks = []
    for spec in specs:
        items = list(dict(spec).items())
        kind, value = items[0] if len(items) == 1 else (None, None)
        if kind == "window":
            s = min(side, max(1, round(float(value) * side)))
            m = torch.zeros(side, side)
            a = (side - s) // 2
            m[a:a + s, a:a + s] = 1
            masks.append(_FixedMask(m.flatten()))
        elif kind == "data":
            masks.append(_FixedMask(((train > floor).float().mean(0) >= float(value)).float()))
        elif kind == "digit":
            masks.append(partial(_digit_mask, floor=floor, side=side, grow=int(value)))
        else:
            raise ValueError(f"a mask spec is one of {{window: s}}, {{data: f}}, "
                             f"{{digit: k}}; got {dict(spec)!r}")
    return masks


class _FixedMask:
    """The same mask for every image, moved to the batch's device once."""

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask.device != x.device or self.mask.dtype != x.dtype:
            self.mask = self.mask.to(x.device, x.dtype)
        return self.mask


def _digit_mask(x: torch.Tensor, floor: float, side: int, grow: int) -> torch.Tensor:
    m = (x > floor).to(x.dtype).view(-1, 1, side, side)
    for _ in range(grow):
        m = F.max_pool2d(m, 3, stride=1, padding=1)
    return m.view(x.shape)


class BarePGD:
    """PGD on the bare classifier, with restarts (D149): :class:`AdaptivePGD` on a
    :class:`Defense` with no purifier and no detector (the margin loss, ``restarts``
    starts, the best iterate per sample). ``graphed(model, graphs)`` gives the
    attack ``(x, labels, eps_abs, starts=())``, as :class:`GraphedAdaptivePGD`."""

    def __init__(self, steps: int, restarts: int,
                 box: Optional[Tuple[float, float]] = None):
        self.attack = AdaptivePGD(steps=steps, restarts=restarts, box=box)

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
                 starts: Sequence[torch.Tensor] = ()) -> torch.Tensor:
        """Adversarial examples for ``x`` (on the model's device); ``eps_abs`` a
        float, or a device scalar when the graphs are enabled. The first restarts
        begin at ``starts`` (each projected into the ball), the others at random."""
        a, defense = self.attack, self.defense
        x = x.detach()
        draws = a.eot if defense.random else 1
        best_x = x.clone()
        best_rate = torch.full((len(x),), -1.0, device=x.device)
        for restart in range(max(a.restarts, len(starts))):
            if restart < len(starts):
                delta = a._bounded(defense, x, starts[restart].to(x.device).detach() - x,
                                   eps_abs)
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


class DataStarts:
    """Starts on the data for the attack on a detector (D156): for each point, the
    nearest (L∞) training point that the detector accepts and the classifier assigns
    to a class other than the point's label, or the point itself if there is none.

    Inside the ε ball such a start is already an error of the defence; outside it,
    the attack projects it onto the ball, towards the other class's data. Random
    starts and the PGD example can instead lie where log p is flat and every point
    is rejected, so the selective attack never finds an accepted one: on spirals and
    MNIST 12×12 they left the detector at up to 0.06 (spirals) and 0.92 (MNIST) for
    an unbounded ε, where the data start reaches 0, and higher by 0.01–0.12 at ε =
    0.2 (notes:adaptive §12).

    Args:
        defense: a :class:`Defense` with a detector on its input and no purifier.
        data: the training inputs; the accepted ones are kept with their predicted
            class, once.
        batch_size: the batch for scoring ``data``.
    """

    CHUNK = 8192   # training points compared at once: a (batch, CHUNK) distance matrix

    def __init__(self, defense: Defense, data: torch.Tensor, batch_size: int):
        if defense.threshold is None or defense.purify is not None:
            raise ValueError("DataStarts needs a defence with a detector and no purifier")
        device = next(defense.model.parameters()).device
        kept, predicted = [], []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                z = data[i:i + batch_size].to(device)
                z = z[log_px(defense.model, z) >= defense.threshold]
                if len(z):   # the model takes no empty batch
                    kept.append(z)
                    predicted.append(defense.model.log_joint(z).argmax(dim=1))
        self.points = torch.cat(kept) if kept else data[:0].to(device)
        self.predicted = (torch.cat(predicted) if kept
                          else torch.zeros(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The start for each point of ``x`` (same shape and device)."""
        x_flat, points = x.flatten(1), self.points.flatten(1)
        dist = torch.full((len(x),), float("inf"), device=x.device)
        nearest = torch.zeros(len(x), dtype=torch.long, device=x.device)
        for k in range(0, len(points), self.CHUNK):
            d = torch.cdist(x_flat, points[k:k + self.CHUNK], p=float("inf"))
            d = d.masked_fill(self.predicted[None, k:k + self.CHUNK] == labels[:, None],
                              float("inf"))
            d_k, j_k = d.min(dim=1)
            closer = d_k < dist
            dist, nearest = torch.where(closer, d_k, dist), torch.where(closer, j_k + k, nearest)
        if not len(points):
            return x.clone()
        found = torch.isfinite(dist).view(-1, *[1] * (x.dim() - 1))
        return torch.where(found, self.points[nearest].reshape(x.shape), x)
