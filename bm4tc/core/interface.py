"""What the analysis needs from a model (D69): its log joint density, up to a constant.

Both the MPS (:class:`bm4tc.core.model.ConditionalBornMachine`) and JEM
(:mod:`bm4tc.core.jem`) model p(x, c). The analysis (attacks, detection,
purification) reaches them only through :class:`GenerativeClassifier` and the
functions below, so "the same attacks, the same defences" holds by construction.

    log_joint(x)       (B, K)  log p(x, c) + C, differentiable in x
    log_normalizer()   float   C: log Z for the MPS (exact); 0 for JEM, whose
                               log p(x) is therefore unnormalised: fine for
                               thresholds and purification, not comparable across
                               models in absolute value

Everything else is derived here, once.
"""
from typing import Protocol, Tuple

import torch


class GenerativeClassifier(Protocol):
    input_range: Tuple[float, float]
    out_dim: int

    def log_joint(self, data: torch.Tensor) -> torch.Tensor: ...

    def log_normalizer(self) -> float: ...

    def reset(self) -> None:
        """Drop state a failed computation may have left behind (no-op if none)."""

    # and nn.Module's to / eval / zero_grad


def class_probabilities(model: GenerativeClassifier, data: torch.Tensor) -> torch.Tensor:
    """p(c | x) -> (B, K)."""
    log_joint = model.log_joint(data)
    return (log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)).exp()


def log_px(model: GenerativeClassifier, data: torch.Tensor) -> torch.Tensor:
    """log p(x) -> (B,), unnormalised if the model's normalizer is."""
    return torch.logsumexp(model.log_joint(data), dim=-1) - model.log_normalizer()


def nll(model: GenerativeClassifier, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean -log p(c | x): the PGD loss."""
    log_joint = model.log_joint(data)
    true = log_joint[torch.arange(len(labels)), labels]
    return (-true + torch.logsumexp(log_joint, dim=-1)).mean()


def best_wrong_log_joint(model: GenerativeClassifier, data: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
    """Mean max_{c' != c} log p(x, c'): the joint attack's loss. The constant C
    does not move the gradient, so it is left out."""
    log_joint = model.log_joint(data)
    true = torch.zeros_like(log_joint, dtype=torch.bool)
    true[torch.arange(len(labels)), labels] = True
    return log_joint.masked_fill(true, float("-inf")).max(dim=-1).values.mean()
