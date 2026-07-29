"""Minimal bounded attacks for the JEM proof-of-concept."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import JEMMLP


@dataclass
class PGDConfig:
    epsilon: float = 0.2
    num_steps: int = 40
    step_size: float | None = None
    random_start: bool = True
    restarts: int = 1


def _bounded_delta(
    candidate: torch.Tensor,
    natural: torch.Tensor,
    epsilon: float,
    input_range: tuple[float, float],
) -> torch.Tensor:
    lo, hi = input_range
    candidate = candidate.clamp(lo, hi)
    return (candidate - natural).clamp(-epsilon, epsilon)


def pgd_classification(
    model: JEMMLP,
    naturals: torch.Tensor,
    labels: torch.LongTensor,
    cfg: PGDConfig,
) -> torch.Tensor:
    """L-inf PGD maximizing cross-entropy, with domain and epsilon projection."""
    x = naturals.detach()
    step = cfg.step_size or 2.5 * cfg.epsilon / cfg.num_steps
    best_adv = x.clone()
    best_loss = torch.full((len(x),), float("-inf"), device=x.device)
    lo, hi = model.input_range

    for _ in range(cfg.restarts):
        if cfg.random_start:
            delta = torch.empty_like(x).uniform_(-cfg.epsilon, cfg.epsilon)
            delta = _bounded_delta(x + delta, x, cfg.epsilon, model.input_range)
        else:
            delta = torch.zeros_like(x)
        for _ in range(cfg.num_steps):
            delta.requires_grad_(True)
            loss = F.cross_entropy(model(x + delta), labels)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            with torch.no_grad():
                delta = delta + step * grad.sign()
                delta = _bounded_delta(x + delta, x, cfg.epsilon, model.input_range)
        with torch.no_grad():
            per_item = F.cross_entropy(model(x + delta), labels, reduction="none")
            improve = per_item > best_loss
            best_loss[improve] = per_item[improve]
            best_adv[improve] = (x + delta)[improve]
    return best_adv.detach()


def pgd_likelihood_aware(
    model: JEMMLP,
    naturals: torch.Tensor,
    labels: torch.LongTensor,
    cfg: PGDConfig,
    score_weight: float = 1.0,
) -> torch.Tensor:
    """Adaptive PGD: misclassify while keeping the marginal score high.

    Maximizes ``CE(x, y) + score_weight * s(x)``. The second term discourages
    the low-score behavior used by likelihood/energy threshold detectors.
    """
    x = naturals.detach()
    step = cfg.step_size or 2.5 * cfg.epsilon / cfg.num_steps
    if cfg.random_start:
        delta = torch.empty_like(x).uniform_(-cfg.epsilon, cfg.epsilon)
        delta = _bounded_delta(x + delta, x, cfg.epsilon, model.input_range)
    else:
        delta = torch.zeros_like(x)
    for _ in range(cfg.num_steps):
        delta.requires_grad_(True)
        adv = x + delta
        objective = (
            F.cross_entropy(model(adv), labels)
            + score_weight * model.marginal_score(adv).mean()
        )
        grad = torch.autograd.grad(objective, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta = delta + step * grad.sign()
            delta = _bounded_delta(x + delta, x, cfg.epsilon, model.input_range)
    return (x + delta).detach()
