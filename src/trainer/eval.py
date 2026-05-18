import math
import logging
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_epoch: int
    best_metrics: dict


def eval_metrics(bm, loader, device) -> tuple[float, float, float]:
    """Single forward pass; returns (dis_loss, acc, gen_loss). Caller must sync tensors first."""
    bm.eval()
    with torch.no_grad():
        log_Z = bm.generator.log_partition_function()
    gen_finite = math.isfinite(log_Z.item())
    if not gen_finite:
        logger.warning(f"log_Z is non-finite ({log_Z.item()}); gen_loss will be nan.")
    losses_dis, losses_gen, correct, total = [], [], 0, 0
    eps = 1e-8
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            embs = bm.classifier.embed(data)
            amps = bm.classifier.forward(embs)
            sq = bm.classifier.abs_square(amps)
            log_sq_obs = sq[range(len(labels)), labels].clamp(min=eps).log()
            log_class_marg = sq.sum(dim=1).clamp(min=eps).log()
            losses_dis.append((log_class_marg - log_sq_obs).mean().item())
            correct += (sq.argmax(dim=1) == labels).sum().item()
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.born import BornMachine, BornMachineConfig, MPSInitConfig

    device = torch.device("cpu")
    bm = BornMachine(
        cfg=BornMachineConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    bm.sync_tensors(after="classification")
    bm.cache_log_Z()

    ds = TensorDataset(torch.zeros(8, 2), torch.randint(0, 2, (8,)))
    loader = DataLoader(ds, batch_size=4)

    dis_loss, acc, gen_loss = eval_metrics(bm, loader, device)
    assert math.isfinite(dis_loss), f"dis_loss non-finite: {dis_loss}"
    assert math.isfinite(acc), f"acc non-finite: {acc}"
    assert math.isfinite(gen_loss), f"gen_loss non-finite: {gen_loss}"
    print(f"  eval_metrics  dis_loss={dis_loss:.4f}  acc={acc:.4f}  gen_loss={gen_loss:.4f}")
    print("eval.py smoke test passed.")
