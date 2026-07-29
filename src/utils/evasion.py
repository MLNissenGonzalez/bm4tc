import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.utils.train import CriterionConfig

_LOG_PROB_EPS: float = float(torch.finfo(torch.float32).tiny)


@dataclass
class EvasionConfig:
    method: str = "FGM"
    norm: int | str = "inf"
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    strengths: list = field(default_factory=lambda: [0.1, 0.3])
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
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using a single gradient step."""
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

        ad_examples = (naturals + strength * normalized_gradient).detach()
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

    def _project(self, perturbation: torch.Tensor, strength: float) -> torch.Tensor:
        """Project perturbation back into the epsilon ball."""
        if self.norm == "inf":
            return perturbation.clamp(-strength, strength)
        elif isinstance(self.norm, int):
            # Project onto Lp ball
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / strength, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, strength: float, device: torch.device) -> torch.Tensor:
        """Initialize random perturbation within epsilon ball."""
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * strength
        elif isinstance(self.norm, int):
            # Sample uniformly from Lp ball (approximate via normalize + scale)
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * strength * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _bounded_delta(
            self,
            perturbation: torch.Tensor,
            naturals: torch.Tensor,
            strength: float,
            input_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Project onto both the valid input domain and the epsilon ball."""
        lo, hi = input_range
        in_domain = (naturals + perturbation).clamp(lo, hi) - naturals
        return self._project(in_domain, strength)

    def generate(
            self,
            born,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using iterative PGD."""
        born.to(device)
        naturals = naturals.to(device).detach()
        labels = labels.to(device)

        step_size = self.step_size if self.step_size is not None else 2.5 * strength / self.num_steps

        if self.random_start:
            delta = self._random_init(naturals.shape, strength, device)
            delta = self._bounded_delta(delta, naturals, strength, born.input_range)
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
            delta = self._bounded_delta(delta, naturals, strength, born.input_range)

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

    def _project(self, perturbation: torch.Tensor, strength: float) -> torch.Tensor:
        if self.norm == "inf":
            return perturbation.clamp(-strength, strength)
        elif isinstance(self.norm, int):
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / strength, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, strength: float, device: torch.device) -> torch.Tensor:
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * strength
        elif isinstance(self.norm, int):
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * strength * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _bounded_delta(
            self,
            perturbation: torch.Tensor,
            naturals: torch.Tensor,
            strength: float,
            input_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Project onto both the valid input domain and the epsilon ball."""
        lo, hi = input_range
        in_domain = (naturals + perturbation).clamp(lo, hi) - naturals
        return self._project(in_domain, strength)

    def generate(
            self,
            born,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using the joint generative attack."""
        born.to(device)
        naturals = naturals.to(device).detach()
        labels   = labels.to(device)

        step_size = self.step_size if self.step_size is not None \
                    else 2.5 * strength / self.num_steps

        delta = (self._random_init(naturals.shape, strength, device)
                 if self.random_start else torch.zeros_like(naturals))
        delta = self._bounded_delta(delta, naturals, strength, born.input_range)

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
            delta = self._bounded_delta(delta, naturals, strength, born.input_range)

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
    Evaluate adversarial robustness of a ConditionalBornMachine.

    Generates adversarial examples using FGM or PGD and computes accuracy
    under attack at multiple perturbation strengths.
    """

    def __init__(
            self,
            method: str = "FGM",
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None),
            strengths: List[float] = [0.1, 0.3],
            # PGD-specific parameters (ignored for FGM)
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True
    ):
        """
        Initialize robustness evaluator.

        Args:
            method: Attack method - "FGM" or "PGD".
            norm: Lp norm for perturbation ball.
            criterion: Loss function configuration.
            strengths: List of epsilon values to evaluate.
            num_steps: PGD iterations (ignored for FGM).
            step_size: PGD step size (ignored for FGM).
            random_start: PGD random initialization (ignored for FGM).
        """
        self.strengths = strengths
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
            strength: float,
            device: torch.device | str = "cpu"
    ):
        return self.method.generate(
            born, naturals, labels, strength, device
        )

    def evaluate(
            self,
            born,
            loader: DataLoader,
            device: torch.device | str = "cpu"
    ):
        """
        Evaluate robustness at each relative strength; returns list of accuracies.

        strengths values are relative fractions of the embedding range size:
        abs_eps = strength * (input_range[1] - input_range[0]).
        """
        born.to(device)
        born.eval()

        range_size = born.input_range[1] - born.input_range[0]
        strength_acc = []

        for strength in self.strengths:
            abs_strength = strength * range_size
            batch_acc = []

            for naturals, labels in loader:
                ad_examples = self.generate(
                    born, naturals, labels, abs_strength, device
                )

                with torch.no_grad():
                    ad_probs = born.class_probabilities(ad_examples)
                    ad_pred = torch.argmax(ad_probs, dim=1)
                    acc = (ad_pred == labels.to(device)).float().mean().item()
                    batch_acc.append(acc)

            mean_acc = sum(batch_acc) / len(batch_acc)
            strength_acc.append(mean_acc)

        return strength_acc


if __name__ == "__main__":
    import sys
    import torch
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig

    device = torch.device("cpu")
    cbm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    cbm.prepare(device=device)

    x = torch.linspace(-1.0, 1.0, 8).unsqueeze(1).expand(8, 2).clone()
    y = torch.randint(0, 2, (8,))
    # strength=0.05 is a relative fraction; abs_eps = 0.05 * range_size(legendre=2.0) = 0.1
    strength_frac = 0.05
    abs_eps = strength_frac * 2.0

    for name, ec in [
        ("FGM", EvasionConfig(method="FGM", strengths=[abs_eps])),
        ("PGD", EvasionConfig(method="PGD", num_steps=3, strengths=[abs_eps])),
    ]:
        attack = build_attack(ec)
        adv = attack.generate(born=cbm, naturals=x, labels=y, strength=abs_eps, device=device)
        assert adv.shape == x.shape, f"{name}: shape mismatch"
        delta = (adv - x).abs().max().item()
        print(f"  {name:5s}  max_delta={delta:.4f}  (eps_abs={abs_eps})")

    print("evasion.py smoke test passed.")
