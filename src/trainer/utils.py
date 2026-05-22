import math
import logging
from dataclasses import dataclass
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_epoch: int
    best_metrics: dict


class NormRegularizer(nn.Module):
    """
    Partition-function norm regularization penalty (trainer-level).

    Computes  strength * (Z - target)²  where Z = exp(log_partition_function()).

    Parameters
    ----------
    strength : float
        Regularization coefficient.
    target : float
        Target value for the partition function Z (norm² of the MPS).
    """

    def __init__(self, strength: float, target: float):
        super().__init__()
        self.strength = strength
        self.target = target

    def forward(self, cbm) -> torch.Tensor:
        log_Z: torch.Tensor = cbm.log_partition_function()
        Z = torch.exp(log_Z)
        return self.strength * (Z - self.target) ** 2


def eval_metrics(cbm, loader, device) -> tuple[float, float, float]:
    """Single forward pass using CBM interface; returns (dis_loss, acc, gen_loss)."""
    cbm.eval()
    with torch.no_grad():
        log_Z = cbm.log_partition_function()
    gen_finite = math.isfinite(log_Z.item())
    if not gen_finite:
        logger.warning(f"log_Z is non-finite ({log_Z.item()}); gen_loss will be nan.")
    losses_dis, losses_gen, correct, total = [], [], 0, 0
    eps = 1e-8
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            abs_sq = cbm.abs_square(cbm.amplitudes(data))
            log_sq_obs = abs_sq[range(len(labels)), labels].clamp(min=eps).log()
            log_class_marg = abs_sq.sum(dim=1).clamp(min=eps).log()
            losses_dis.append((log_class_marg - log_sq_obs).mean().item())
            correct += (abs_sq.argmax(dim=1) == labels).sum().item()
            total += len(labels)
            if gen_finite:
                gen_batch = (log_Z - log_sq_obs).mean().item()
                if math.isfinite(gen_batch):
                    losses_gen.append(gen_batch)
                else:
                    logger.warning(f"Non-finite gen_loss ({gen_batch}), skipping batch.")
    dis_loss = sum(losses_dis) / len(losses_dis) if losses_dis else float("nan")
    acc = correct / total if total > 0 else float("nan")
    gen_loss = sum(losses_gen) / len(losses_gen) if losses_gen else float("nan")
    return dis_loss, acc, gen_loss


def eval_rob(cbm, loader, attack, abs_strength: float, device) -> float:
    """Evaluates robustness at a single perturbation strength; returns mean robust acc."""
    cbm.eval()
    correct, total = 0, 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        adv = attack.generate(born=cbm, naturals=data, labels=labels,
                              strength=abs_strength, device=device)
        with torch.no_grad():
            probs = cbm.class_probabilities(adv)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else float("nan")


def eval_softmax(cbm, loader, device) -> tuple[float, float]:
    """(softmax_loss, softmax_acc) using raw CBM amplitudes."""
    from src.trainer.softmax import ClassificationSoftmaxNLL
    cbm.eval()
    criterion = ClassificationSoftmaxNLL()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            amps = cbm.amplitudes(data)
            losses.append(criterion(amps, labels).item())
            correct += (amps.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    softmax_loss = sum(losses) / len(losses) if losses else float("nan")
    softmax_acc = correct / total if total > 0 else float("nan")
    return softmax_loss, softmax_acc


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig

    device = torch.device("cpu")
    cfg = CBMConfig(
        embedding="legendre",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
    )
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2, device=device)
    cbm.prepare(device)

    ds = TensorDataset(torch.zeros(8, 2), torch.randint(0, 2, (8,)))
    loader = DataLoader(ds, batch_size=4)

    dis_loss, acc, gen_loss = eval_metrics(cbm, loader, device)
    assert math.isfinite(dis_loss), f"dis_loss non-finite: {dis_loss}"
    assert math.isfinite(acc), f"acc non-finite: {acc}"
    assert math.isfinite(gen_loss), f"gen_loss non-finite: {gen_loss}"
    print(f"  eval_metrics  dis_loss={dis_loss:.4f}  acc={acc:.4f}  gen_loss={gen_loss:.4f}")

    softmax_loss, softmax_acc = eval_softmax(cbm, loader, device)
    assert math.isfinite(softmax_loss), f"softmax_loss non-finite: {softmax_loss}"
    assert math.isfinite(softmax_acc), f"softmax_acc non-finite: {softmax_acc}"
    print(f"  eval_softmax  loss={softmax_loss:.4f}  acc={softmax_acc:.4f}")

    reg = NormRegularizer(strength=1.0, target=1.0)
    cbm.reset()
    penalty = reg(cbm)
    assert penalty.isfinite(), f"NormRegularizer penalty non-finite: {penalty}"
    print(f"  NormRegularizer penalty={penalty.item():.4f}")

    print("\nutils.py smoke test passed.")
