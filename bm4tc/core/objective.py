import math
import logging
import random
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm.auto import tqdm

from bm4tc.core.interface import class_probabilities

logger = logging.getLogger(__name__)


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
class OptimizerConfig:
    name: str = "adam"
    # weight_decay is 0 and not a knob of these experiments: norm control (the
    # soft log Z penalty) already constrains the parameter scale.
    kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"weight_decay": 0.0})


_OPTIMIZER_MAP = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adamax": optim.Adamax,
    "nadam": optim.NAdam,
}


def optimizer(params, config: OptimizerConfig, **extra) -> optim.Optimizer:
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
    extra :
        Keyword arguments set by the caller, not the config, e.g.
        ``capturable=True`` for a CUDA-graph step (D90).

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

    return optimizer_cls(params, **config.kwargs, **extra)


# ---------------------------------------------------------------------------
# Trainer utilities
# ---------------------------------------------------------------------------

@dataclass
class NormControlConfig:
    log_target: Optional[Union[float, str]] = 0.0
    soft_strength: float = 0.1
    debug: bool = False


class NormRegularizer(nn.Module):
    """
    Partition-function norm regularization penalty (trainer-level).
    Computes  strength * (log Z - log_target)² / N  where log Z = cbm.log_Z() and
    N = cbm.n_features. The /N matches the per-variable objective (D86): it gives
    each site the same restoring force on every dataset, and at β=1 the loss is
    exactly the α=1 loss of before (pilot A) divided by N.

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
        # back to a fresh contraction if nothing is cached (e.g. beta=0).
        log_Z: torch.Tensor = cbm.log_Z(recompute=False)
        return self.strength * (log_Z - self.log_target) ** 2 / cbm.n_features


def resolve_log_target(cbm, nc: NormControlConfig) -> float:
    """Resolve ``NormControlConfig.log_target`` to a finite float.

    - ``None`` → the pretrained model's current ``log Z`` (a no-op target that
      pins the norm wherever the (already normalized) start model sits).
    - ``str``  → a Python expression in terms of ``n_features``, ``data_dim``,
      ``in_dim``, ``out_dim``, ``bond_dim`` and ``sqrt``/``log``/``exp``.
    - ``float`` → used directly.

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
        data_dim = n_features - 1  # every site but the class site
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


# The order of norm_statistics' entries; NormTracker and the collapse report
# read them by these names.
NORM_STATISTICS = ("log_amp_sq_mean", "log_amp_sq_min", "log_amp_sq_max",
                   "amp_nonfinite_count", "amp_nan_count", "log_Z")


def norm_statistics(log_amp_sq: torch.Tensor, log_Z: Optional[torch.Tensor]) -> torch.Tensor:
    """One training forward's norm statistics, as one tensor in the order of
    ``NORM_STATISTICS``, on the device and without a host sync (D90).

    From the batch's log|ψ|² (B, C): the mean, min and max of its finite entries
    (nan, +inf, -inf when none is), its non-finite and NaN counts (+inf is an
    amplitude overflow, NaN a degenerate contraction); then log Z, nan when the
    step did not form it.
    """
    log_amp_sq = log_amp_sq.detach()
    finite = torch.isfinite(log_amp_sq)
    zero = torch.zeros((), dtype=log_amp_sq.dtype, device=log_amp_sq.device)
    mean = torch.where(finite, log_amp_sq, zero).sum() / finite.sum()
    low = torch.where(finite, log_amp_sq, zero + math.inf).min()
    high = torch.where(finite, log_amp_sq, zero - math.inf).max()
    nonfinite = (~finite).sum().to(log_amp_sq.dtype)
    nans = torch.isnan(log_amp_sq).sum().to(log_amp_sq.dtype)
    log_Z = zero + math.nan if log_Z is None else log_Z.detach().to(log_amp_sq.dtype)
    return torch.stack([mean, low, high, nonfinite, nans, log_Z])


class NormTracker:
    """Accumulate the per-step norm statistics (:func:`norm_statistics`) over one
    epoch, then emit one ``norm/*`` metric dict: per site (divided by N =
    ``n_features``, D86), except ``norm/log_Z_headroom``, which is absolute
    because overflow is.

    :meth:`add` keeps a copy of each step's statistics on the device;
    :meth:`finalize` reads them all with one host sync. Running max/min are kept
    alongside the mean so an intra-epoch explosion (a spike) survives
    aggregation instead of being smeared by the mean. Only finite values count.
    If no step formed log Z (``beta=0`` with no soft norm control),
    :meth:`finalize` takes one post-epoch ``log_partition_function()`` snapshot.
    """

    def __init__(self):
        self._steps: List[torch.Tensor] = []

    def add(self, statistics: torch.Tensor) -> None:
        """Fold in one step's statistics (copied: a captured step overwrites its
        outputs on the next replay)."""
        self._steps.append(statistics.detach().clone())

    def finalize(self, cbm) -> Dict[str, float]:
        rows = torch.stack(self._steps).tolist() if self._steps else []
        columns = {name: [row[i] for row in rows if math.isfinite(row[i])]
                   for i, name in enumerate(NORM_STATISTICS)}
        log_Z = columns["log_Z"]
        if not log_Z:
            # beta=0 without soft norm control never forms log Z during the
            # step; take one post-epoch snapshot so norm/log_Z is still reported.
            with torch.no_grad():
                try:
                    snapshot = cbm.log_partition_function().item()
                except Exception:
                    snapshot = float("nan")
            if math.isfinite(snapshot):
                log_Z = [snapshot]

        out: Dict[str, float] = {}
        n = cbm.n_features
        if log_Z:
            out["norm/log_Z_mean"] = sum(log_Z) / len(log_Z) / n
            out["norm/log_Z_max"] = max(log_Z) / n
            out["norm/log_Z_min"] = min(log_Z) / n
            # Amplitudes overflow once ‖ψ‖ = exp(log_Z/2) crosses the dtype max,
            # i.e. log_Z > 2·log(finfo.max) (≈177.45 for float32/complex64).
            ceiling = 2.0 * math.log(torch.finfo(cbm.dtype).max)
            out["norm/log_Z_headroom"] = ceiling - max(log_Z)
        # Each amp stat on its own guard: a step can contribute a finite max/min
        # even if its mean was non-finite (and vice versa).
        means = columns["log_amp_sq_mean"]
        if means:
            out["norm/log_amp_sq_mean"] = sum(means) / len(means) / n
        if columns["log_amp_sq_max"]:
            out["norm/log_amp_sq_max"] = max(columns["log_amp_sq_max"]) / n
        if columns["log_amp_sq_min"]:
            out["norm/log_amp_sq_min"] = min(columns["log_amp_sq_min"]) / n
        return out


def n_vars(data: torch.Tensor) -> int:
    """N, the number of modelled variables of a batch: its features and the class
    (D86). The MPS's ``n_features`` is the same number."""
    return data[0].numel() + 1


def mix(dis: float, gen: float, beta: float, n: int) -> float:
    """``(1-β)·dis + β·gen/N`` for N = ``n`` modelled variables (D86): the
    objective in nats per variable, gated exactly as in :meth:`CBM.mixed_nll`.

    Each term is dropped rather than multiplied by a zero weight, so an endpoint
    beta never turns a non-finite half (a nan ``gen`` from a diverged ``log_Z``)
    into a nan mix.
    """
    out = (1.0 - beta) * dis if beta < 1.0 else 0.0
    if beta > 0.0:
        out += beta * gen / n
    return out


def evaluate(
    cbm, loader, device, *, log_Z=None,
    beta: float = 0.0, attack=None, eps_abs: float = 0.0, progress: bool = False,
) -> dict:
    """Validation (or test) metrics, and the training objective mirrored on them.

    Without an attack, ``objective = mix(L_dis, L_gen, β, N)``. With one, it
    mirrors the AT objective of :class:`bm4tc.core.train.Trainer` (D91), every
    sample attacked:

        objective = mix(mean L_dis(x_adv), mean L_gen(x), β, N)

    Every mean is over samples. ``loss_dis``, ``loss_x`` and ``acc`` are clean;
    ``loss_x = -log p(x) / n`` is in nats per feature (D86). With an attack,
    ``loss_adv`` (mean L_dis on x_adv) and ``rob`` are added.

    Any model of :mod:`bm4tc.core.interface` (``cbm`` is its ``log_joint``):
    ``L_gen = log Z - log_joint(x)[y]``. ``log_Z`` None computes the MPS's exact
    one; JEM passes its SGLD estimate, or nan when there is no generative term
    (then ``loss_x`` is nan and the objective drops the generative term).
    """
    cbm.eval()
    if log_Z is None:
        with torch.no_grad():
            log_Z = cbm.log_partition_function()
        if not math.isfinite(log_Z.item()):
            logger.warning(f"log_Z is non-finite ({log_Z.item()}); loss_x will be nan.")
    gen_finite = math.isfinite(float(log_Z))

    dis_sum = gen_sum = marg_sum = 0.0      # clean
    dis_adv_sum = 0.0                       # adversarial
    correct = rob_correct = total = 0

    for data, labels in tqdm(
        loader, desc="eval", unit="batch", leave=False,
        dynamic_ncols=True, disable=not progress,
    ):
        data, labels = data.to(device), labels.to(device)
        B = len(labels)
        with torch.no_grad():
            # MPS: log|ψ|², the loss's entry point, so evaluation matches
            # training's numerics.
            las = cbm.log_joint(data)                        # (B, C)
            log_sq_obs = las[range(B), labels]
            dis = torch.logsumexp(las, dim=1) - log_sq_obs    # (B,)
            correct += (las.argmax(dim=1) == labels).sum().item()
            total += B
            dis_sum += dis.sum().item()
            if gen_finite:
                gen_sum += (log_Z - log_sq_obs).sum().item()
                marg_sum += (log_Z - torch.logsumexp(las, dim=1)).sum().item()

        if attack is not None:
            adv = attack.generate(model=cbm, naturals=data, labels=labels,
                                  eps_abs=eps_abs, device=device)
            with torch.no_grad():
                las_adv = cbm.log_joint(adv)
                log_sq_adv = las_adv[range(B), labels]
                dis_adv_sum += (torch.logsumexp(las_adv, dim=1) - log_sq_adv).sum().item()
                rob_correct += (las_adv.argmax(dim=1) == labels).sum().item()

    def _mean(s, n):
        return s / n if n else float("nan")

    dis_loss = _mean(dis_sum, total)
    n = n_vars(data) if total else 1
    gen_loss = _mean(gen_sum, total) if gen_finite else float("nan")
    loss_x = _mean(marg_sum, total) / (n - 1) if gen_finite else float("nan")
    out = {"loss_dis": dis_loss, "loss_x": loss_x, "acc": _mean(correct, total)}

    if attack is None:
        out["objective"] = mix(dis_loss, gen_loss, beta, n)
        return out

    adv_loss = _mean(dis_adv_sum, total)
    out["objective"] = mix(adv_loss, gen_loss, beta, n)
    out["loss_adv"] = adv_loss
    out["rob"] = _mean(rob_correct, total)
    return out


def eval_rob(cbm, loader, attack, eps_abs: float, device, progress: bool = False) -> float:
    """Evaluates robustness at a single absolute epsilon; returns mean robust acc.

    ``eps_abs`` is a model-domain budget, not a fraction — callers convert via
    ``rel_to_abs(eps_rel, range_size_of(cbm))``.

    Set ``progress=True`` to show a transient per-batch tqdm bar (used by post-hoc
    analysis); the default keeps training-time validation output clean.
    """
    cbm.eval()
    correct, total = 0, 0
    for data, labels in tqdm(
        loader, desc=f"rob eps_abs={eps_abs:.3g}", unit="batch", leave=False,
        dynamic_ncols=True, disable=not progress,
    ):
        data, labels = data.to(device), labels.to(device)
        adv = attack.generate(model=cbm, naturals=data, labels=labels,
                              eps_abs=eps_abs, device=device)
        with torch.no_grad():
            probs = class_probabilities(cbm, adv)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else float("nan")
