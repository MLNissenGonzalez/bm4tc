import math
import logging
import random
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_LOG_PROB_EPS: float = float(torch.finfo(torch.float32).tiny)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducible
    *training* randomness (model init, DataLoader shuffling, PGD, sampling).

    Must be called **after** data loading and **before** model creation.
    Data pipeline seeds (``gen_dow_kwargs.seed``, ``dataset.split_seed``)
    are handled independently by their respective functions.

    Parameters
    ----------
    seed : int
        Integer value used to seed all random number generators.
        Corresponds to ``tracking.seed`` in the Hydra config.

    Notes
    -----
    - Seeds Python's `random` module, NumPy, and PyTorch (CPU and GPU).
    - For PyTorch, also sets `torch.backends.cudnn.deterministic=True` to
        enforce deterministic algorithms in cuDNN.
    - Disables `torch.backends.cudnn.benchmark` to avoid non-deterministic
        optimizations.
    - Sets the `PYTHONHASHSEED` environment variable for hash-based operations.
    - May reduce performance due to disabling some GPU optimizations.

    Examples
    --------
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior in cuDNN (can slow things down!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Optimizer config + factory
# ---------------------------------------------------------------------------

@dataclass
class CriterionConfig:
    name: str = "nll"
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class OptimizerConfig:
    name: str = "adam"
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


_OPTIMIZER_MAP = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adamax": optim.Adamax,
    "nadam": optim.NAdam,
}


def optimizer(params, config: OptimizerConfig) -> optim.Optimizer:
    """
    Select and instantiate a PyTorch optimizer.

    Parameters
    ----------
    params : iterable
        Parameters to optimize, e.g. model.parameters()
    config.name : str
        Name of the optimizer, e.g. "adam"
    config.kwargs : dict, optional
        Extra arguments passed to the optimizer, e.g. {"lr": 1e-3}

    Returns
    -------
    optim.Optimizer
        Instantiated optimizer.
    """
    key = config.name.replace("-", "").replace("_", "").lower()
    try:
        optimizer_cls = _OPTIMIZER_MAP[key]
    except KeyError:
        raise ValueError(f"Optimizer {config.name} not recognised. "
                         f"Available: {list(_OPTIMIZER_MAP.keys())}")

    return optimizer_cls(params, **config.kwargs)


# ---------------------------------------------------------------------------
# Trainer utilities
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    best_epoch: int
    best_metrics: dict


@dataclass
class NormControlConfig:
    log_target: Optional[Union[float, str]] = 0.0
    hard_every: int = 1
    soft_strength: float = 0.0
    debug: bool = False


class NormRegularizer(nn.Module):
    """
    Partition-function norm regularization penalty (trainer-level).
    Computes  strength * (log Z - log_target)²  where log Z = cbm.log_Z().

    Parameters
    ----------
    strength : float
        Regularization coefficient.
    log_target : float
        Target value for log Z (must be finite).
    """

    def __init__(self, strength: float, log_target: float):
        if not math.isfinite(log_target):
            raise ValueError(f"NormRegularizer: log_target must be finite, got {log_target}")
        super().__init__()
        self.strength = strength
        self.log_target: float = log_target

    def forward(self, cbm) -> torch.Tensor:
        # recompute=False reuses the with-gradient log Z from the same step's
        # mixed_nll forward (one norm contraction/step instead of two). Falls
        # back to a fresh contraction if nothing is cached (e.g. alpha=0).
        log_Z: torch.Tensor = cbm.log_Z(recompute=False)
        return self.strength * (log_Z - self.log_target) ** 2


def resolve_log_target(cbm, datahandler, nc: NormControlConfig) -> float:
    """Resolve ``NormControlConfig.log_target`` to a finite float.

    - ``None`` → the pretrained model's current ``log Z`` (a no-op target that
      pins the norm wherever the (already normalized) start model sits).
    - ``str``  → a Python expression in terms of ``n_features``, ``data_dim``,
      ``in_dim``, ``out_dim``, ``bond_dim`` and ``sqrt``/``log``/``exp``.
    - ``float`` → used directly.

    Shared by NLLTrainer and AdversarialTrainer.
    """
    raw = nc.log_target

    if raw is None:
        with torch.no_grad():
            log_Z0 = cbm.log_partition_function()
        log_target = log_Z0.item()
        logger.info(f"NormControl: log_target (pretrained) = {log_target:.6g}")
        return log_target

    if isinstance(raw, str):
        n_features = cbm.n_features
        data_dim = datahandler.data_dim
        in_dim = cbm.in_dim
        out_dim = cbm.out_dim
        bond_dim = cbm.bond_dim
        _ns = {
            "__builtins__": {},
            "n_features": n_features,
            "data_dim": data_dim,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "bond_dim": bond_dim,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
        }
        try:
            result = eval(raw, _ns)  # noqa: S307
        except Exception as exc:
            raise ValueError(
                f"NormControl: could not evaluate log_target expression "
                f"{raw!r} (n_features={n_features}, data_dim={data_dim}, "
                f"in_dim={in_dim}, out_dim={out_dim}, bond_dim={bond_dim}): {exc}"
            ) from exc
        log_target = float(result)
        if not math.isfinite(log_target):
            raise ValueError(
                f"NormControl: log_target expression {raw!r} evaluated to "
                f"{log_target}, but log_target must be finite."
            )
        logger.info(
            f"NormControl: log_target (expression {raw!r}) = {log_target:.6g} "
            f"[n_features={n_features}, data_dim={data_dim}, "
            f"in_dim={in_dim}, out_dim={out_dim}, bond_dim={bond_dim}]"
        )
        return log_target

    return float(raw)


def eval_metrics(cbm, loader, device, progress: bool = False) -> tuple[float, float, float]:
    """Single forward pass using CBM interface; returns (dis_loss, acc, gen_loss).

    Set ``progress=True`` to show a transient per-batch tqdm bar (used by post-hoc
    analysis); the default keeps training-time validation output clean.
    """
    cbm.eval()
    with torch.no_grad():
        log_Z = cbm.log_partition_function()
    gen_finite = math.isfinite(log_Z.item())
    if not gen_finite:
        logger.warning(f"log_Z is non-finite ({log_Z.item()}); gen_loss will be nan.")
    losses_dis, losses_gen, correct, total = [], [], 0, 0
    with torch.no_grad():
        for data, labels in tqdm(
            loader, desc="eval", unit="batch", leave=False,
            dynamic_ncols=True, disable=not progress,
        ):
            data, labels = data.to(device), labels.to(device)
            amp     = cbm.amplitudes(data)
            log_abs = torch.log(amp.abs().clamp(min=_LOG_PROB_EPS))
            log_sq_obs     = 2.0 * log_abs[range(len(labels)), labels]
            log_class_marg = torch.logsumexp(2.0 * log_abs, dim=1)
            losses_dis.append((log_class_marg - log_sq_obs).mean().item())
            correct += (log_abs.argmax(dim=1) == labels).sum().item()
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


def eval_rob(cbm, loader, attack, abs_strength: float, device, progress: bool = False) -> float:
    """Evaluates robustness at a single perturbation strength; returns mean robust acc.

    Set ``progress=True`` to show a transient per-batch tqdm bar (used by post-hoc
    analysis); the default keeps training-time validation output clean.
    """
    cbm.eval()
    correct, total = 0, 0
    for data, labels in tqdm(
        loader, desc=f"rob eps={abs_strength:.3g}", unit="batch", leave=False,
        dynamic_ncols=True, disable=not progress,
    ):
        data, labels = data.to(device), labels.to(device)
        adv = attack.generate(born=cbm, naturals=data, labels=labels,
                              strength=abs_strength, device=device)
        with torch.no_grad():
            probs = cbm.class_probabilities(adv)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else float("nan")
