import math
import logging
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_epoch: int
    best_metrics: dict


def eval_dis(bm, loader, device) -> tuple[float, float]:
    """Single forward pass over loader; returns (dis_loss, acc)."""
    bm.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            embs = bm.classifier.embed(data)
            amps = bm.classifier.forward(embs)
            sq = bm.classifier.abs_square(amps)
            eps = 1e-8
            log_sq_obs = sq[range(len(labels)), labels].clamp(min=eps).log()
            log_class_marg = sq.sum(dim=1).clamp(min=eps).log()
            losses.append((log_class_marg - log_sq_obs).mean().item())
            correct += (sq.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    dis_loss = sum(losses) / len(losses) if losses else float("nan")
    acc = correct / total if total > 0 else float("nan")
    return dis_loss, acc


def eval_gen(bm, loader, device) -> float:
    """Returns mean NLL of the joint distribution p(x, c) over the loader."""
    bm.eval()
    with torch.no_grad():
        log_Z = bm.generator.log_partition_function()
    if not math.isfinite(log_Z.item()):
        logger.warning(f"log_Z is non-finite ({log_Z.item()}); gen_loss will be nan.")
        return float("nan")
    losses = []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            embs = bm.classifier.embed(data)
            sq = bm.classifier.abs_square(bm.classifier.forward(embs))
            eps = 1e-8
            log_sq_obs = sq[range(len(labels)), labels].clamp(min=eps).log()
            loss = (log_Z - log_sq_obs).mean().item()
            if not math.isfinite(loss):
                logger.warning(f"Non-finite gen_loss ({loss}), skipping batch.")
                continue
            losses.append(loss)
    return sum(losses) / len(losses) if losses else float("nan")


def eval_rob(bm, loader, attack, abs_strength: float, device) -> float:
    """Evaluates robustness at a single perturbation strength; returns mean robust acc."""
    bm.eval()
    correct, total = 0, 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        adv = attack.generate(born=bm, naturals=data, labels=labels,
                              strength=abs_strength, device=device)
        with torch.no_grad():
            probs = bm.class_probabilities(adv)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else float("nan")


def eval_softmax(bm, loader, device) -> tuple[float, float]:
    """Like eval_dis but uses raw signed amplitudes for ClassificationSoftmaxNLL."""
    from src.utils.criterions import ClassificationSoftmaxNLL
    bm.eval()
    criterion = ClassificationSoftmaxNLL()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            amps = bm.classifier.amplitudes(data)
            losses.append(criterion(amps, labels).item())
            correct += (amps.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    softmax_loss = sum(losses) / len(losses) if losses else float("nan")
    softmax_acc = correct / total if total > 0 else float("nan")
    return softmax_loss, softmax_acc
