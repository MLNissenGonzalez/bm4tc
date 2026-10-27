"""How steep the classifier is at the data, to first order, and where (D169).

For a point (x, y) the margin is m = log p(y|x) − max_{c≠y} log p(c|x) (the PGD loss,
D149) and g = ∂m/∂x. An L∞ perturbation of size ε that stays in the input range
[lo, hi] lowers m, to first order, by at most ε · G, with

    G = Σ_{lo < x_i < hi} |g_i|  +  Σ_{x_i = lo} max(0, −g_i)  +  Σ_{x_i = hi} max(0, g_i),

since a feature at an end of the range can only move inwards. ε* = max(m, 0) / G is then
the budget that flips the point to first order (0 when it is already wrong), so the
fraction of points with ε* below a budget estimates 1 − rob there. The share of G that
comes from features at the ends says whether the steepness sits where many inputs
saturate (MNIST's background pixels, 77 % of them at lo).

Only the model interface is used (D69), so the MPS and JEM give the same quantities.
Budgets are absolute here, as in :mod:`bm4tc.analysis.uq`; the part in
:mod:`bm4tc.pipeline.analyse` reports them relative to the range.
"""
from dataclasses import dataclass

import torch


@dataclass
class Sensitivity:
    """Per point: ``margin`` m, ``gain`` G (per unit of absolute ε) and ``gain_ends``,
    the part of G from features at an end of the input range."""
    margin: torch.Tensor
    gain: torch.Tensor
    gain_ends: torch.Tensor


def margin_sensitivity(model, x: torch.Tensor, labels: torch.Tensor, batch_size: int,
                       device) -> Sensitivity:
    """:class:`Sensitivity` of ``model`` (a :class:`GenerativeClassifier`) at the points
    ``x`` (in the model's coordinates) with true classes ``labels``, on CPU."""
    lo, hi = model.input_range
    margins, gains, ends = [], [], []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device).detach().requires_grad_(True)
        yb = labels[i:i + batch_size].to(device)
        log_joint = model.log_joint(xb)
        true = log_joint.gather(1, yb[:, None]).squeeze(1)
        wrong = log_joint.scatter(1, yb[:, None], float("-inf")).max(dim=1).values
        margin = true - wrong
        (g,) = torch.autograd.grad(margin.sum(), xb)
        model.reset()
        at_lo, at_hi = xb.detach() <= lo, xb.detach() >= hi
        inward = torch.where(at_lo, (-g).clamp(min=0),
                             torch.where(at_hi, g.clamp(min=0), g.abs()))
        margins.append(margin.detach().cpu())
        gains.append(inward.flatten(1).sum(1).cpu())
        ends.append(torch.where(at_lo | at_hi, inward, 0.0).flatten(1).sum(1).cpu())
    return Sensitivity(torch.cat(margins), torch.cat(gains), torch.cat(ends))
