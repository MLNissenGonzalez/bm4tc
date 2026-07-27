"""Natural JEM purification strategies: score ascent and projected SGLD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    """Run one locally projected SGLD purification sweep."""
    return sgld_purify_snapshots(model, sampler, data, cfg, (1,))[1]


def sgld_purify_snapshots(
    model: JEMMLP,
    sampler: SGLDSampler,
    data: torch.Tensor,
    cfg: PurificationConfig,
    sweep_points: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Run locally projected SGLD sweeps and return requested snapshots.

    One sweep is ``cfg.num_steps`` SGLD transitions projected onto the
    L-infinity ball of radius ``cfg.radius`` around the state at the start of
    that sweep. The next sweep is recentered on the previous result, mirroring
    the local restriction and between-sweep drift of MPS Gibbs purification.
    """
    points = sorted({int(value) for value in sweep_points})
    if not points or points[0] < 1:
        raise ValueError("sweep_points must contain positive integers")

    current = data.detach()
    snapshots = {}
    for sweep in range(1, points[-1] + 1):
        center = current.detach()
        current = sampler.refine(
            model,
            current,
            center=center,
            radius=cfg.radius,
            num_steps=cfg.num_steps,
            step_size=cfg.step_size,
            noise_std=cfg.sgld_noise_std,
        )
        if sweep in points:
            snapshots[sweep] = current.detach().clone()
    return snapshots
