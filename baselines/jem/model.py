"""Small MLP interpreted as a Joint Energy-based Model (JEM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class JEMMLPConfig:
    input_dim: int = 144
    hidden_dims: tuple[int, ...] = (550, 480)
    num_classes: int = 10
    activation: str = "silu"
    input_range: tuple[float, float] = (-1.0, 1.0)


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "silu":
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation {name!r}; use silu, relu or gelu.")


class JEMMLP(nn.Module):
    """Deterministic MLP whose logits define an unnormalised joint density.

    ``f(x)[y]`` is the unnormalised log-density of ``p(x, y)`` and
    ``logsumexp(f(x), y)`` is the unnormalised marginal score for ``x``.
    The partition function is intractable, so this class deliberately calls it
    a score rather than a normalised log-likelihood.
    """

    def __init__(self, cfg: JEMMLPConfig):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.input_dim, *cfg.hidden_dims, cfg.num_classes]
        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i < len(dims) - 2:
                layers.append(_activation(cfg.activation))
        self.network = nn.Sequential(*layers)
        self.input_range = tuple(cfg.input_range)
        self.out_dim = cfg.num_classes
        self.data_dim = cfg.input_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)

    def joint_scores(self, data: torch.Tensor) -> torch.Tensor:
        return self(data)

    def marginal_score(self, data: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self(data), dim=-1)

    def energy(self, data: torch.Tensor) -> torch.Tensor:
        return -self.marginal_score(data)

    def class_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        return self(data).softmax(dim=-1)

    def discriminative_loss(
        self, data: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        return F.cross_entropy(self(data), labels)

    def joint_contrastive_loss(
        self,
        positives: torch.Tensor,
        labels: torch.LongTensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive estimator of joint NLL, up to the unknown log Z.

        The positive phase is ``-f_y(x+)``. The negative phase estimates the
        gradient of ``log Z`` with samples from ``p(x)`` and is
        ``logsumexp_y f_y(x-)``. Negative samples must already be detached.
        """
        logits_pos = self(positives)
        pos_joint = logits_pos.gather(1, labels[:, None]).squeeze(1)
        neg_marginal = self.marginal_score(negatives.detach())
        return -pos_joint.mean() + neg_marginal.mean()

    def mixed_loss(
        self,
        positives: torch.Tensor,
        labels: torch.LongTensor,
        negatives: torch.Tensor | None,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Convex interpolation matching the MPS alpha semantics.

        ``alpha=0`` is pure cross-entropy and never requires negatives.
        ``alpha=1`` is the approximate joint generative objective.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        logits_pos = self(positives)
        dis = F.cross_entropy(logits_pos, labels)
        pos_marginal = torch.logsumexp(logits_pos, dim=-1).mean()
        if alpha == 0.0:
            px_cd = dis.detach().new_full((), float("nan"))
            joint_cd = dis.detach().new_full((), float("nan"))
            neg_marginal = dis.detach().new_full((), float("nan"))
            mixed = dis
        else:
            if negatives is None:
                raise ValueError("Negative samples are required when alpha > 0.")
            neg_marginal = self.marginal_score(negatives.detach()).mean()
            px_cd = neg_marginal - pos_marginal
            joint_cd = dis + px_cd
            # Equivalent to (1-alpha) * CE + alpha * joint-CD, but this form
            # makes explicit that CE keeps unit weight, as in the original JEM.
            mixed = dis + alpha * px_cd
        return mixed, {
            "dis_loss": dis,
            "px_cd_loss": px_cd,
            "joint_cd_loss": joint_cd,
            "gen_loss": joint_cd,
            "mixed_loss": mixed,
            "positive_marginal_score": pos_marginal,
            "negative_marginal_score": neg_marginal,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path, **extra) -> None:
        payload = {
            "model_config": asdict(self.cfg),
            "model_state": self.state_dict(),
            **extra,
        }
        torch.save(payload, str(path))

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | str = "cpu"
    ) -> tuple["JEMMLP", dict]:
        payload = torch.load(str(path), map_location=device, weights_only=False)
        raw = dict(payload["model_config"])
        raw["hidden_dims"] = tuple(raw["hidden_dims"])
        raw["input_range"] = tuple(raw["input_range"])
        model = cls(JEMMLPConfig(**raw)).to(device)
        model.load_state_dict(payload["model_state"])
        extra = {k: v for k, v in payload.items() if k not in {"model_config", "model_state"}}
        return model, extra


def nearest_uniform_width(
    target_parameters: int, input_dim: int = 144, num_classes: int = 10
) -> tuple[int, int]:
    """Return the two-hidden-layer width nearest to a parameter target."""
    best: tuple[int, int] | None = None
    for width in range(1, 10000):
        count = (
            input_dim * width + width
            + width * width + width
            + width * num_classes + num_classes
        )
        error = abs(count - target_parameters)
        if best is None or error < best[1]:
            best = (width, error)
        if count > target_parameters and error > best[1]:
            break
    assert best is not None
    return best


def mps_parameter_count(
    data_dim: int, physical_dim: int, bond_dim: int, num_classes: int
) -> int:
    """Complex-element count for the OBC MPS used by MNIST experiments."""
    if data_dim < 2:
        raise ValueError("Expected at least two data sites.")
    boundary = 2 * physical_dim * bond_dim
    internal_data = (data_dim - 2) * physical_dim * bond_dim**2
    class_site = num_classes * bond_dim**2
    return boundary + internal_data + class_site
