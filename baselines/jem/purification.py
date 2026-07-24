"""Natural JEM purification strategies: score ascent and projected SGLD."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import JEMMLP
from .sampler import SGLDSampler


@dataclass
class PurificationConfig:
    radius: float = 0.2
    num_steps: int = 20
    step_size: float | None = None
    sgld_noise_std: float = 0.005


def gradient_purify(
    model: JEMMLP, data: torch.Tensor, cfg: PurificationConfig
) -> torch.Tensor:
    """Projected ascent on the unnormalised marginal JEM score."""
    center = data.detach()
    delta = torch.zeros_like(center)
    step = cfg.step_size or 2.5 * cfg.radius / cfg.num_steps
    lo, hi = model.input_range
    for _ in range(cfg.num_steps):
        delta.requires_grad_(True)
        x = (center + delta).clamp(lo, hi)
        score = model.marginal_score(x).mean()
        grad = torch.autograd.grad(score, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta = delta + step * grad.sign()
            delta.clamp_(-cfg.radius, cfg.radius)
    return (center + delta).clamp(lo, hi).detach()


def sgld_purify(
    model: JEMMLP,
    sampler: SGLDSampler,
    data: torch.Tensor,
    cfg: PurificationConfig,
) -> torch.Tensor:
    """Projected SGLD around the observed input."""
    return sampler.refine(
        model,
        data,
        center=data,
        radius=cfg.radius,
        num_steps=cfg.num_steps,
        step_size=cfg.step_size,
        noise_std=cfg.sgld_noise_std,
    )
