"""Evasion (adversarial) attacks against a ConditionalBornMachine.

Budget convention (see "Budget vocabulary" in CLAUDE.md):
    ``eps_rel``  authored fraction of the embedding domain width ``hi - lo``. This is
                 what configs carry, and it equals the budget in the data's own units.
    ``eps_abs``  model-domain value, ``eps_rel * (hi - lo)``. Every attack method in
                 this module takes ``eps_abs`` — conversion happens in the caller
                 (``AdversarialTrainer._init_attack``, ``analysis/run.py``) via
                 ``rel_to_abs``.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.utils.train import CriterionConfig

_LOG_PROB_EPS: float = float(torch.finfo(torch.float32).tiny)


@dataclass
class EvasionConfig:
    method: str = "FGM"
    norm: int | str = "inf"
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    eps_rel: list = field(default_factory=lambda: [0.1, 0.3])
    num_steps: int = 10
    step_size: Optional[float] = None
    random_start: bool = True


def _dis_loss(born, data: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    """Discriminative NLL loss via CBM mixed_nll with alpha=0."""
    return born.mixed_nll(data, labels, alpha=0.0)


def _zero_grad(born) -> None:
    born.zero_grad()


def normalizing(x: torch.FloatTensor, norm: int | str):
    """
    Normalize a tensor of shape (batch size, data dim)
    along the data dim (flattened).
    """
    if norm == "inf":
        normalized = x.sign()

    elif isinstance(norm, int):
        if norm < 1:
            raise ValueError("Only accept p >= 1.")
        x_norm = x.norm(p=norm, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-12)
        normalized = x / x_norm

    else:
        raise ValueError(f"{norm=}, but expected to be int or 'inf'.")
    
    return normalized

class FastGradientMethod:
    """
    Fast Gradient Method (FGM) adversarial attack.

    Single-step attack that perturbs inputs in the direction of the loss gradient,
    normalized according to the specified Lp norm.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None)
    ):
        self.norm = norm
        # criterion parameter retained for API compatibility; loss is now
        # computed via born.mixed_nll(alpha=0) in generate().

    def generate(
            self,
            born,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            eps_abs: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using a single gradient step.

        ``eps_abs`` is an absolute model-domain budget, not a fraction.
        """
        born.to(device)
        naturals = naturals.to(device).detach().clone().requires_grad_(True)
        labels = labels.to(device)

        loss = _dis_loss(born, naturals, labels)

        _zero_grad(born)
        if naturals.grad is not None:
            naturals.grad.zero_()

        loss.backward()

        grad = naturals.grad.detach()
        normalized_gradient = normalizing(grad, norm=self.norm)

        ad_examples = (naturals + eps_abs * normalized_gradient).detach()
        return ad_examples


class ProjectedGradientDescent:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Iterative attack that performs multiple gradient ascent steps with projection
    back onto the epsilon ball. Stronger than FGM but more expensive.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None),
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True
    ):
        self.norm = norm
        # criterion parameter retained for API compatibility; loss is computed
        # via born.mixed_nll(alpha=0) in generate().
        self.num_steps = num_steps if num_steps is not None else 10
        self.step_size = step_size
        self.random_start = random_start

    def _project(self, perturbation: torch.Tensor, eps_abs: float) -> torch.Tensor:
        """Project perturbation back into the epsilon ball."""
        if self.norm == "inf":
            return perturbation.clamp(-eps_abs, eps_abs)
        elif isinstance(self.norm, int):
            # Project onto Lp ball
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / eps_abs, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, eps_abs: float, device: torch.device) -> torch.Tensor:
        """Initialize random perturbation within epsilon ball."""
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * eps_abs
        elif isinstance(self.norm, int):
            # Sample uniformly from Lp ball (approximate via normalize + scale)
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * eps_abs * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _bounded_delta(
            self,
            perturbation: torch.Tensor,
            naturals: torch.Tensor,
            eps_abs: float,
            input_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Project onto both the valid input domain and the epsilon ball."""
        lo, hi = input_range
        in_domain = (naturals + perturbation).clamp(lo, hi) - naturals
        return self._project(in_domain, eps_abs)

    def generate(
            self,
            born,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            eps_abs: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using iterative PGD.

        ``eps_abs`` is an absolute model-domain budget, not a fraction.
        """
        born.to(device)
        naturals = naturals.to(device).detach()
        labels = labels.to(device)

        step_size = self.step_size if self.step_size is not None else 2.5 * eps_abs / self.num_steps

        if self.random_start:
            delta = self._random_init(naturals.shape, eps_abs, device)
            delta = self._bounded_delta(delta, naturals, eps_abs, born.input_range)
        else:
            delta = torch.zeros_like(naturals)

        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            loss = _dis_loss(born, naturals + delta, labels)

            _zero_grad(born)
            if delta.grad is not None:
                delta.grad.zero_()

            loss.backward()

            grad = delta.grad.detach()
            normalized_gradient = normalizing(grad, norm=self.norm)

            delta = delta.detach() + step_size * normalized_gradient
            delta = self._bounded_delta(delta, naturals, eps_abs, born.input_range)

        lo, hi = born.input_range
        return (naturals + delta).clamp(lo, hi).detach()


class JointProjectedGradientDescent:
    """PGD maximising max_{c'≠c} ln|ψ(x̃, c')|²  (joint generative attack).

    Loss per step: +mean( max_{c'≠c}  2·log|ψ(x̃, c')| )  — gradient ascent.
    The worst-case wrong class is re-selected dynamically at every gradient step.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True,
    ):
        self.norm = norm
        self.num_steps = num_steps if num_steps is not None else 10
        self.step_size = step_size
        self.random_start = random_start

    def _project(self, perturbation: torch.Tensor, eps_abs: float) -> torch.Tensor:
        if self.norm == "inf":
            return perturbation.clamp(-eps_abs, eps_abs)
        elif isinstance(self.norm, int):
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / eps_abs, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, eps_abs: float, device: torch.device) -> torch.Tensor:
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * eps_abs
        elif isinstance(self.norm, int):
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * eps_abs * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _bounded_delta(
            self,
            perturbation: torch.Tensor,
            naturals: torch.Tensor,
            eps_abs: float,
            input_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Project onto both the valid input domain and the epsilon ball."""
        lo, hi = input_range
        in_domain = (naturals + perturbation).clamp(lo, hi) - naturals
        return self._project(in_domain, eps_abs)

    def generate(
            self,
            born,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            eps_abs: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using the joint generative attack.

        ``eps_abs`` is an absolute model-domain budget, not a fraction.
        """
        born.to(device)
        naturals = naturals.to(device).detach()
        labels   = labels.to(device)

        step_size = self.step_size if self.step_size is not None \
                    else 2.5 * eps_abs / self.num_steps

        delta = (self._random_init(naturals.shape, eps_abs, device)
                 if self.random_start else torch.zeros_like(naturals))
        delta = self._bounded_delta(delta, naturals, eps_abs, born.input_range)

        batch = len(labels)
        K = born.out_dim
        true_class_mask = torch.zeros(batch, K, dtype=torch.bool, device=device)
        true_class_mask[torch.arange(batch), labels] = True

        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            _amps = born.amplitudes if hasattr(born, "amplitudes") else born.classifier.amplitudes
            amplitudes  = _amps(naturals + delta)                                # (B, K)
            log_joint   = 2 * torch.log(amplitudes.abs().clamp(min=_LOG_PROB_EPS))  # (B, K)
            log_joint_w = log_joint.masked_fill(true_class_mask, float('-inf'))
            loss        = log_joint_w.max(dim=-1).values.mean()

            _zero_grad(born)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad  = delta.grad.detach()
            delta = delta.detach() + step_size * normalizing(grad, norm=self.norm)
            delta = self._bounded_delta(delta, naturals, eps_abs, born.input_range)

        lo, hi = born.input_range
        return (naturals + delta).clamp(lo, hi).detach()


_METHOD_MAP = {
    "FGM":       FastGradientMethod,
    "PGD":       ProjectedGradientDescent,
    "JOINT_PGD": JointProjectedGradientDescent,
}


def build_attack(
    evasion_cfg: EvasionConfig,
) -> FastGradientMethod | ProjectedGradientDescent | JointProjectedGradientDescent:
    """Construct an attack object from an EvasionConfig."""
    method = evasion_cfg.method
    if method == "PGD":
        return ProjectedGradientDescent(
            norm=evasion_cfg.norm,
            criterion=evasion_cfg.criterion,
            num_steps=evasion_cfg.num_steps,
            step_size=evasion_cfg.step_size,
            random_start=evasion_cfg.random_start,
        )
    if method == "FGM":
        return FastGradientMethod(
            norm=evasion_cfg.norm,
            criterion=evasion_cfg.criterion,
        )
    if method == "JOINT_PGD":
        return JointProjectedGradientDescent(
            norm=evasion_cfg.norm,
            num_steps=evasion_cfg.num_steps,
            step_size=evasion_cfg.step_size,
            random_start=evasion_cfg.random_start,
        )
    raise ValueError(f"Unknown attack method: {method!r}. Expected 'FGM', 'PGD', or 'JOINT_PGD'.")


class RobustnessEvaluation:
    """
    Dispatching wrapper around the attack methods.

    Builds an FGM / PGD / JOINT_PGD attack from a method name and forwards
    :meth:`generate` to it.
    """

    def __init__(
            self,
            method: str = "FGM",
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None),
            eps_rel: List[float] = [0.1, 0.3],
            # PGD-specific parameters (ignored for FGM)
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True
    ):
        """
        Initialize robustness evaluator.

        Args:
            method: Attack method - "FGM", "PGD" or "JOINT_PGD".
            norm: Lp norm for perturbation ball.
            criterion: Loss function configuration.
            eps_rel: Relative budgets (fractions of the input domain) this evaluator
                was configured with. Carried for provenance only — :meth:`generate`
                takes an absolute ``eps_abs``.
            num_steps: PGD iterations (ignored for FGM).
            step_size: PGD step size (ignored for FGM).
            random_start: PGD random initialization (ignored for FGM).
        """
        self.eps_rel = eps_rel
        method_cls = _METHOD_MAP[method]
        if method == "PGD":
            self.method = method_cls(
                norm=norm,
                criterion=criterion,
                num_steps=num_steps,
                step_size=step_size,
                random_start=random_start
            )
        elif method == "JOINT_PGD":
            self.method = method_cls(
                norm=norm,
                num_steps=num_steps,
                step_size=step_size,
                random_start=random_start
            )
        else:
            self.method = method_cls(
                norm=norm,
                criterion=criterion
            )

    def generate(
            self,
            born,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            eps_abs: float,
            device: torch.device | str = "cpu"
    ):
        return self.method.generate(
            born, naturals, labels, eps_abs, device
        )


if __name__ == "__main__":
    import sys
    import torch
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    from src.utils.embeddings import range_size_of, rel_to_abs

    device = torch.device("cpu")
    cbm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    cbm.prepare(device=device)

    x = torch.linspace(-1.0, 1.0, 8).unsqueeze(1).expand(8, 2).clone()
    y = torch.randint(0, 2, (8,))
    # Authored relative; converted once, as every caller must.
    eps_rel = 0.05
    eps_abs = rel_to_abs(eps_rel, range_size_of(cbm))  # legendre: 0.05 * 2.0 = 0.1

    for name, ec in [
        ("FGM", EvasionConfig(method="FGM", eps_rel=[eps_rel])),
        ("PGD", EvasionConfig(method="PGD", num_steps=3, eps_rel=[eps_rel])),
    ]:
        attack = build_attack(ec)
        adv = attack.generate(born=cbm, naturals=x, labels=y, eps_abs=eps_abs, device=device)
        assert adv.shape == x.shape, f"{name}: shape mismatch"
        delta = (adv - x).abs().max().item()
        print(f"  {name:5s}  max_delta={delta:.4f}  (eps_rel={eps_rel}, eps_abs={eps_abs})")

    print("evasion.py smoke test passed.")
