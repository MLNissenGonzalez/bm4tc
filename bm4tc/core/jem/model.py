"""A small MLP read as a Joint Energy-based Model (JEM).

``f(x)[c]`` is the unnormalised log density of p(x, c); ``logsumexp_c f(x)`` is the
unnormalised marginal score of x. The partition function is intractable, so
``log_normalizer`` is 0 and log p(x) from :mod:`bm4tc.core.interface` is a score:
fine for thresholds, ranking, SGLD and purification, not comparable in absolute
value to the MPS's exact log-likelihood.

The width is chosen to match an MPS (D70): two hidden layers of the uniform width
whose real parameter count is nearest to the MPS's real degrees of freedom (twice
its complex element count).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch import nn


@dataclass
class JEMModelConfig:
    # The MPS this model is sized to (D70): set from the study's arch.
    match_in_dim: int = 3
    match_bond_dim: int = 20
    depth: int = 2                     # hidden layers, all of the matched width
    activation: str = "silu"
    input_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    hidden_dims: Optional[List[int]] = None   # None: computed from the match


_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


def mps_parameter_count(data_dim: int, in_dim: int, bond_dim: int, num_classes: int) -> int:
    """Complex elements of the OBC ConditionalBornMachine with the class site inside."""
    if data_dim < 2:
        raise ValueError("Expected at least two data sites.")
    boundary = 2 * in_dim * bond_dim
    internal = (data_dim - 2) * in_dim * bond_dim ** 2
    return boundary + internal + num_classes * bond_dim ** 2


def mlp_parameter_count(data_dim: int, hidden_dims: List[int], num_classes: int) -> int:
    dims = [data_dim, *hidden_dims, num_classes]
    return sum(i * o + o for i, o in zip(dims[:-1], dims[1:]))


def matched_width(target: int, data_dim: int, num_classes: int, depth: int = 2) -> int:
    """The uniform hidden width whose parameter count is nearest to ``target``."""
    best, width = None, 1
    while True:
        error = abs(mlp_parameter_count(data_dim, [width] * depth, num_classes) - target)
        if best is not None and error > best[1]:
            return best[0]
        if best is None or error < best[1]:
            best = (width, error)
        width += 1


def hidden_dims(cfg: JEMModelConfig, data_dim: int, num_classes: int) -> List[int]:
    if cfg.hidden_dims is not None:
        return list(cfg.hidden_dims)
    target = 2 * mps_parameter_count(data_dim, cfg.match_in_dim, cfg.match_bond_dim, num_classes)
    return [matched_width(target, data_dim, num_classes, cfg.depth)] * cfg.depth


class JEMMLP(nn.Module):
    def __init__(self, cfg: JEMModelConfig, data_dim: int, num_classes: int):
        super().__init__()
        cfg = OmegaConf.structured(cfg) if not OmegaConf.is_config(cfg) else cfg
        self.cfg = cfg
        dims = [data_dim, *hidden_dims(cfg, data_dim, num_classes), num_classes]
        layers: List[nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i < len(dims) - 2:
                layers.append(_ACTIVATIONS[cfg.activation]())
        self.network = nn.Sequential(*layers)
        self.input_range: Tuple[float, float] = tuple(float(v) for v in cfg.input_range)
        self.data_dim = data_dim
        self.out_dim = num_classes
        self.hidden_dims = dims[1:-1]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)

    # ── The analysis interface (bm4tc.core.interface, D69) ──────────────────

    def log_joint(self, data: torch.Tensor) -> torch.Tensor:
        """f(x) = log p(x, c) + log Z -> (B, K)."""
        return self(data)

    def log_normalizer(self) -> float:
        return 0.0   # intractable: log p(x) is unnormalised

    def reset(self) -> None:
        pass

    def marginal_score(self, data: torch.Tensor) -> torch.Tensor:
        """logsumexp_c f(x) -> (B,): the unnormalised log p(x)."""
        return torch.logsumexp(self(data), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def save(self, path: str, **extra) -> None:
        torch.save({"config": OmegaConf.to_container(self.cfg, resolve=True),
                    "data_dim": self.data_dim, "num_classes": self.out_dim,
                    "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, device="cpu") -> Tuple["JEMMLP", dict]:
        """The model and the checkpoint's extra entries (e.g. the replay buffer)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = OmegaConf.merge(OmegaConf.structured(JEMModelConfig), ckpt.pop("config"))
        model = cls(cfg, ckpt.pop("data_dim"), ckpt.pop("num_classes")).to(device)
        model.load_state_dict(ckpt.pop("state"))
        return model, ckpt
