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
    hard_every: int = 0
    soft_strength: float = 0.1
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


class NormTracker:
    """Accumulate per-step training-side norm (log Z) and mean-amplitude
    (log|ψ|²) statistics over one epoch, then emit one ``norm/*`` metric dict.

    Reads the caches ``mixed_nll`` (``_amp_diag_cache``, always) and the forward
    / ``NormRegularizer`` (``_log_Z_cache``, when ``alpha>0`` or
    ``soft_strength>0``) already populate, so it adds no contraction in the
    common cases. Call :meth:`record_amp` / :meth:`record_logZ` per step *before*
    ``optimizer.step()`` invalidates the caches, then :meth:`finalize` once.

    Running max/min are kept alongside the mean so an intra-epoch explosion (a
    spike) survives aggregation instead of being smeared by the mean. ``log Z``
    is taken only from finite cache values; if it is never cached during the
    epoch (``alpha=0`` with no soft norm control), :meth:`finalize` falls back to
    a single post-epoch ``log_partition_function()`` snapshot.
    """

    def __init__(self):
        self._logZ_sum, self._logZ_n = 0.0, 0
        self._logZ_max, self._logZ_min = -math.inf, math.inf
        self._amp_sum, self._amp_n = 0.0, 0
        self._amp_max, self._amp_min = -math.inf, math.inf

    def record_amp(self, cbm) -> None:
        """Fold in this step's log|ψ|² batch stats (cached by ``mixed_nll``)."""
        d = getattr(cbm, "_amp_diag_cache", None)
        if not d:
            return
        mean = d.get("log_amp_sq_mean", float("nan"))
        if math.isfinite(mean):
            self._amp_sum += mean
            self._amp_n += 1
        mx = d.get("log_amp_sq_max", float("nan"))
        if math.isfinite(mx):
            self._amp_max = max(self._amp_max, mx)
        mn = d.get("log_amp_sq_min", float("nan"))
        if math.isfinite(mn):
            self._amp_min = min(self._amp_min, mn)

    def record_logZ(self, cbm) -> None:
        """Fold in this step's log Z if it is cached and finite."""
        cached = getattr(cbm, "_log_Z_cache", None)
        if cached is None:
            return
        v = cached.detach().item()
        if math.isfinite(v):
            self._logZ_sum += v
            self._logZ_n += 1
            self._logZ_max = max(self._logZ_max, v)
            self._logZ_min = min(self._logZ_min, v)

    def finalize(self, cbm) -> Dict[str, float]:
        if self._logZ_n == 0:
            # alpha=0 without soft norm control never forms log Z during the
            # step; take one post-epoch snapshot so norm/log_Z is still reported.
            with torch.no_grad():
                try:
                    v = cbm.log_partition_function().item()
                except Exception:
                    v = float("nan")
            if math.isfinite(v):
                self._logZ_sum, self._logZ_n = v, 1
                self._logZ_max = self._logZ_min = v

        out: Dict[str, float] = {}
        if self._logZ_n:
            out["norm/log_Z_mean"] = self._logZ_sum / self._logZ_n
            out["norm/log_Z_max"] = self._logZ_max
            out["norm/log_Z_min"] = self._logZ_min
            # Amplitudes overflow once ‖ψ‖ = exp(log_Z/2) crosses the dtype max,
            # i.e. log_Z > 2·log(finfo.max) (≈177.45 for float32/complex64).
            ceiling = 2.0 * math.log(torch.finfo(cbm.dtype).max)
            out["norm/log_Z_headroom"] = ceiling - self._logZ_max
        # Emit each amp stat on its own guard: a step can contribute a finite
        # max/min even if its mean was non-finite (and vice versa).
        if self._amp_n:
            out["norm/log_amp_sq_mean"] = self._amp_sum / self._amp_n
        if math.isfinite(self._amp_max):
            out["norm/log_amp_sq_max"] = self._amp_max
        if math.isfinite(self._amp_min):
            out["norm/log_amp_sq_min"] = self._amp_min
        return out


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
            # Shared entry point with the loss: dispatches on cbm.accumulate, so
            # eval uses the same log|ψ|² path as training (overflow-safe when on).
            las = cbm._log_amp_sq(data)                       # (B, C) = log|ψ|²
            log_sq_obs     = las[range(len(labels)), labels]
            log_class_marg = torch.logsumexp(las, dim=1)
            losses_dis.append((log_class_marg - log_sq_obs).mean().item())
            correct += (las.argmax(dim=1) == labels).sum().item()
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


def _mix(dis: float, gen: float, alpha: float) -> float:
    """``(1-α)·dis + α·gen``, gated exactly as in :meth:`CBM.mixed_nll`.

    Each term is dropped rather than multiplied by a zero weight, so an endpoint
    alpha never turns a non-finite half (a nan ``gen`` from a diverged ``log_Z``)
    into a nan mix.
    """
    out = (1.0 - alpha) * dis if alpha < 1.0 else 0.0
    if alpha > 0.0:
        out += alpha * gen
    return out


def eval_split(
    cbm, loader, attack, eps_abs: float, device, *,
    alpha: float, clean_weight: float, adv_indices, progress: bool = False,
) -> dict:
    """Combined clean + robust validation for split-objective adversarial training.

    Mirrors the ``gen_on_clean`` training objective on the validation set:

        at_loss = (1-α)·[ (1-cw)·mean_{S_adv} L_dis(x_adv)
                        +    cw ·mean_{S_cln} L_dis(x)     ]
                +   α ·mean_{all} L_gen(x)

    ``S_adv`` is the fixed sample subset given by ``adv_indices`` (positions in the
    loader's iteration order — non-train splits are built with ``shuffle=False``,
    so they are stable across epochs); ``S_cln`` is its complement. Sizing
    ``|S_adv| = (1-cw)·n`` makes the two weighted means reconstruct a single pass
    over the validation set while attacking only a ``(1-cw)`` fraction of it.

    ``dis_loss``/``gen_loss``/``acc`` are clean and over the *full* set, so they
    stay directly comparable to :func:`eval_metrics`, and so is ``mixed_loss``,
    their clean α-mix — ``at_loss`` is the only key that sees adversarial data.
    ``rob`` is over ``S_adv`` only, and is omitted when that subset is empty
    (``clean_weight == 1``); ``n_rob`` reports its size so the estimator is
    recoverable from the run.
    """
    cbm.eval()
    with torch.no_grad():
        log_Z = cbm.log_partition_function()
    gen_finite = math.isfinite(log_Z.item())
    if not gen_finite:
        logger.warning(f"log_Z is non-finite ({log_Z.item()}); gen_loss will be nan.")

    adv_indices = set(adv_indices)
    offset = 0
    dis_sum = gen_sum = 0.0      # clean, full set
    dis_adv_sum = 0.0            # adversarial, S_adv
    dis_cln_sum = 0.0            # clean, S_cln
    correct = total = 0
    rob_correct = n_adv = 0

    for data, labels in tqdm(
        loader, desc="eval split", unit="batch", leave=False,
        dynamic_ncols=True, disable=not progress,
    ):
        data, labels = data.to(device), labels.to(device)
        B = len(labels)
        mask = torch.tensor(
            [(offset + i) in adv_indices for i in range(B)],
            dtype=torch.bool, device=device,
        )
        offset += B

        with torch.no_grad():
            # Same log|ψ|² entry point as eval_metrics / the loss: dispatches on
            # cbm.accumulate, so validation matches training's numerics.
            las = cbm._log_amp_sq(data)                       # (B, C)
            log_sq_obs = las[range(B), labels]
            dis = torch.logsumexp(las, dim=1) - log_sq_obs    # (B,)
            correct += (las.argmax(dim=1) == labels).sum().item()
            total += B
            dis_sum += dis.sum().item()
            dis_cln_sum += dis[~mask].sum().item()
            if gen_finite:
                gen_sum += (log_Z - log_sq_obs).sum().item()

        if bool(mask.any()):
            sub_data, sub_labels = data[mask], labels[mask]
            adv = attack.generate(born=cbm, naturals=sub_data, labels=sub_labels,
                                  eps_abs=eps_abs, device=device)
            with torch.no_grad():
                las_adv = cbm._log_amp_sq(adv)
                n_sub = len(sub_labels)
                log_sq_adv = las_adv[range(n_sub), sub_labels]
                dis_adv_sum += (torch.logsumexp(las_adv, dim=1) - log_sq_adv).sum().item()
                rob_correct += (las_adv.argmax(dim=1) == sub_labels).sum().item()
            n_adv += n_sub

    n_cln = total - n_adv

    def _mean(s, n):
        return s / n if n else float("nan")

    dis_loss = _mean(dis_sum, total)
    gen_loss = _mean(gen_sum, total) if gen_finite else float("nan")

    # Weighted means use the realised subset sizes, so a rounded |S_adv| stays
    # consistent with the weight it is combined under.
    dis_term = 0.0
    if n_adv:
        dis_term += (1.0 - clean_weight) * _mean(dis_adv_sum, n_adv)
    if n_cln:
        dis_term += clean_weight * _mean(dis_cln_sum, n_cln)

    out = {
        "dis_loss": dis_loss,
        "gen_loss": gen_loss,
        "acc": _mean(correct, total),
        "mixed_loss": _mix(dis_loss, gen_loss, alpha),
        "at_loss": _mix(dis_term, gen_loss, alpha),
        "n_rob": n_adv,
    }
    if n_adv:
        out["rob"] = rob_correct / n_adv
    return out


def eval_at(
    cbm, loader, attack, eps_abs: float, device, *,
    alpha: float, clean_weight: float, progress: bool = False,
) -> dict:
    """Combined clean + robust validation for the default adversarial objective.

    The non-split counterpart of :func:`eval_split`: it mirrors

        at_loss = (1-cw)·mixed_nll(x_adv, α) + cw·mixed_nll(x, α)

    on the validation set, where ``mixed_nll(·, α) = (1-α)·L_dis + α·L_gen``. Both
    terms are over the *whole* set — attacking a subset is the split path's device,
    and is what makes its ``rob`` a subset estimator; here ``rob`` keeps exactly the
    meaning it has in :func:`eval_rob`.

    Note the generative half of the adversarial term is ``L_gen(x_adv)``, not
    ``L_gen(x)``: this is a mirror of the objective actually minimized, and the
    default objective does put the generative term on adversarial examples. (The
    ``gen_on_clean`` objective does not — that is what ``eval_split`` is for.)

    ``dis_loss``/``gen_loss``/``acc``/``mixed_loss`` are clean, so they stay directly
    comparable to :func:`eval_metrics`; ``at_loss`` is the only key that sees
    adversarial data. Cost is one clean forward plus one attack over the loader,
    i.e. an ``eval_metrics`` and an ``eval_rob`` folded into a single pass.
    """
    cbm.eval()
    with torch.no_grad():
        log_Z = cbm.log_partition_function()
    gen_finite = math.isfinite(log_Z.item())
    if not gen_finite:
        logger.warning(f"log_Z is non-finite ({log_Z.item()}); gen_loss will be nan.")

    dis_sum = gen_sum = 0.0              # clean
    dis_adv_sum = gen_adv_sum = 0.0      # adversarial
    correct = rob_correct = total = 0

    for data, labels in tqdm(
        loader, desc=f"eval at eps_abs={eps_abs:.3g}", unit="batch", leave=False,
        dynamic_ncols=True, disable=not progress,
    ):
        data, labels = data.to(device), labels.to(device)
        B = len(labels)

        with torch.no_grad():
            # Same log|ψ|² entry point as eval_metrics / the loss: dispatches on
            # cbm.accumulate, so validation matches training's numerics.
            las = cbm._log_amp_sq(data)                       # (B, C)
            log_sq_obs = las[range(B), labels]
            dis_sum += (torch.logsumexp(las, dim=1) - log_sq_obs).sum().item()
            correct += (las.argmax(dim=1) == labels).sum().item()
            total += B
            if gen_finite:
                gen_sum += (log_Z - log_sq_obs).sum().item()

        adv = attack.generate(born=cbm, naturals=data, labels=labels,
                              eps_abs=eps_abs, device=device)
        with torch.no_grad():
            las_adv = cbm._log_amp_sq(adv)
            log_sq_adv = las_adv[range(B), labels]
            dis_adv_sum += (torch.logsumexp(las_adv, dim=1) - log_sq_adv).sum().item()
            rob_correct += (las_adv.argmax(dim=1) == labels).sum().item()
            if gen_finite:
                gen_adv_sum += (log_Z - log_sq_adv).sum().item()

    def _mean(s, n):
        return s / n if n else float("nan")

    dis_loss = _mean(dis_sum, total)
    gen_loss = _mean(gen_sum, total) if gen_finite else float("nan")
    mixed_loss = _mix(dis_loss, gen_loss, alpha)
    mixed_adv = _mix(
        _mean(dis_adv_sum, total),
        _mean(gen_adv_sum, total) if gen_finite else float("nan"),
        alpha,
    )

    # Gated like the training objective: at cw=0 the clean term is absent, not
    # weighted by zero, so a nan clean half cannot leak into the criterion.
    at_loss = (1.0 - clean_weight) * mixed_adv if clean_weight < 1.0 else 0.0
    if clean_weight > 0.0:
        at_loss += clean_weight * mixed_loss

    return {
        "dis_loss": dis_loss,
        "gen_loss": gen_loss,
        "acc": _mean(correct, total),
        "mixed_loss": mixed_loss,
        "at_loss": at_loss,
        "rob": _mean(rob_correct, total),
    }


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
        adv = attack.generate(born=cbm, naturals=data, labels=labels,
                              eps_abs=eps_abs, device=device)
        with torch.no_grad():
            probs = cbm.class_probabilities(adv)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else float("nan")
