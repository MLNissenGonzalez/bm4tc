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
class NormControlConfig:
    log_target: Optional[Union[float, str]] = 0.0
    hard_every: int = 0
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


class NormTracker:
    """Accumulate per-step training-side norm (log Z) and mean-amplitude
    (log|ψ|²) statistics over one epoch, then emit one ``norm/*`` metric dict:
    per site (divided by N = ``n_features``, D86), except ``norm/log_Z_headroom``,
    which is absolute because overflow is.

    Reads ``cbm.forward_stats()``: the amplitude stats ``mixed_nll`` always
    forms, and the log Z the forward or the ``NormRegularizer`` forms when
    ``beta>0`` or ``soft_strength>0``, so it adds no contraction in the common
    cases. Call :meth:`record_amp` / :meth:`record_logZ` per step *before*
    ``optimizer.step()`` changes the parameters, then :meth:`finalize` once.

    Running max/min are kept alongside the mean so an intra-epoch explosion (a
    spike) survives aggregation instead of being smeared by the mean. ``log Z``
    is taken only from finite cache values; if it is never cached during the
    epoch (``beta=0`` with no soft norm control), :meth:`finalize` falls back to
    a single post-epoch ``log_partition_function()`` snapshot.
    """

    def __init__(self):
        self._logZ_sum, self._logZ_n = 0.0, 0
        self._logZ_max, self._logZ_min = -math.inf, math.inf
        self._amp_sum, self._amp_n = 0.0, 0
        self._amp_max, self._amp_min = -math.inf, math.inf

    def record_amp(self, cbm) -> None:
        """Fold in the log|ψ|² stats of the last ``mixed_nll`` batch."""
        d = cbm.forward_stats()
        if "log_amp_sq_mean" not in d:
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
        """Fold in this step's log Z if a forward formed it and it is finite."""
        v = cbm.forward_stats().get("log_Z")
        if v is None:
            return
        if math.isfinite(v):
            self._logZ_sum += v
            self._logZ_n += 1
            self._logZ_max = max(self._logZ_max, v)
            self._logZ_min = min(self._logZ_min, v)

    def finalize(self, cbm) -> Dict[str, float]:
        if self._logZ_n == 0:
            # beta=0 without soft norm control never forms log Z during the
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
        n = cbm.n_features
        if self._logZ_n:
            out["norm/log_Z_mean"] = self._logZ_sum / self._logZ_n / n
            out["norm/log_Z_max"] = self._logZ_max / n
            out["norm/log_Z_min"] = self._logZ_min / n
            # Amplitudes overflow once ‖ψ‖ = exp(log_Z/2) crosses the dtype max,
            # i.e. log_Z > 2·log(finfo.max) (≈177.45 for float32/complex64).
            ceiling = 2.0 * math.log(torch.finfo(cbm.dtype).max)
            out["norm/log_Z_headroom"] = ceiling - self._logZ_max
        # Emit each amp stat on its own guard: a step can contribute a finite
        # max/min even if its mean was non-finite (and vice versa).
        if self._amp_n:
            out["norm/log_amp_sq_mean"] = self._amp_sum / self._amp_n / n
        if math.isfinite(self._amp_max):
            out["norm/log_amp_sq_max"] = self._amp_max / n
        if math.isfinite(self._amp_min):
            out["norm/log_amp_sq_min"] = self._amp_min / n
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
    beta: float = 0.0, attack=None, eps_abs: float = 0.0,
    clean_weight: float = 1.0, adv_indices=(), progress: bool = False,
) -> dict:
    """Validation (or test) metrics, and the training objective mirrored on them.

    Without an attack, ``objective = mix(L_dis, L_gen, β, N)``. With one, it
    mirrors the AT objective of :class:`bm4tc.core.train.Trainer`:

        objective = (1-β)·[ (1-cw)·mean_{S_adv} L_dis(x_adv)
                        +    cw ·mean_{S_cln} L_dis(x)     ]
                +  (β/N)·mean_{all} L_gen(x)

    ``S_adv`` is the fixed sample subset given by ``adv_indices`` (positions in the
    loader's iteration order; non-train splits are built with ``shuffle=False``,
    so they are stable across epochs); ``S_cln`` is its complement. Sizing
    ``|S_adv| = (1-cw)·n`` makes the two weighted means reconstruct a single pass
    over the set while attacking only a ``(1-cw)`` fraction of it.

    Every mean is over samples. ``loss_dis``, ``loss_x`` and ``acc`` are clean and
    over the full set; ``loss_x = -log p(x) / n`` is in nats per feature (D86). With an attack, ``n_rob`` is ``|S_adv|``, and ``loss_adv``
    (mean L_dis on x_adv) and ``rob`` are over ``S_adv``, omitted when it is empty
    (``clean_weight == 1``).

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

    adv_indices = set(adv_indices) if attack is not None else set()
    offset = 0
    dis_sum = gen_sum = marg_sum = 0.0      # clean, full set
    dis_adv_sum = 0.0            # adversarial, S_adv
    dis_cln_sum = 0.0            # clean, S_cln
    correct = total = 0
    rob_correct = n_adv = 0

    for data, labels in tqdm(
        loader, desc="eval", unit="batch", leave=False,
        dynamic_ncols=True, disable=not progress,
    ):
        data, labels = data.to(device), labels.to(device)
        B = len(labels)
        mask = None
        if adv_indices:
            mask = torch.tensor(
                [(offset + i) in adv_indices for i in range(B)],
                dtype=torch.bool, device=device,
            )
        offset += B

        with torch.no_grad():
            # MPS: log|ψ|², the loss's entry point, so evaluation matches
            # training's numerics.
            las = cbm.log_joint(data)                        # (B, C)
            log_sq_obs = las[range(B), labels]
            dis = torch.logsumexp(las, dim=1) - log_sq_obs    # (B,)
            correct += (las.argmax(dim=1) == labels).sum().item()
            total += B
            dis_sum += dis.sum().item()
            dis_cln_sum += dis[~mask].sum().item() if mask is not None else dis.sum().item()
            if gen_finite:
                gen_sum += (log_Z - log_sq_obs).sum().item()
                marg_sum += (log_Z - torch.logsumexp(las, dim=1)).sum().item()

        if mask is not None and bool(mask.any()):
            sub_data, sub_labels = data[mask], labels[mask]
            adv = attack.generate(model=cbm, naturals=sub_data, labels=sub_labels,
                                  eps_abs=eps_abs, device=device)
            with torch.no_grad():
                las_adv = cbm.log_joint(adv)
                n_sub = len(sub_labels)
                log_sq_adv = las_adv[range(n_sub), sub_labels]
                dis_adv_sum += (torch.logsumexp(las_adv, dim=1) - log_sq_adv).sum().item()
                rob_correct += (las_adv.argmax(dim=1) == sub_labels).sum().item()
            n_adv += n_sub

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

    # Weighted means use the realised subset sizes, so a rounded |S_adv| stays
    # consistent with the weight it is combined under.
    n_cln = total - n_adv
    dis_term = 0.0
    if n_adv:
        dis_term += (1.0 - clean_weight) * _mean(dis_adv_sum, n_adv)
    if n_cln:
        dis_term += clean_weight * _mean(dis_cln_sum, n_cln)
    out["objective"] = mix(dis_term, gen_loss, beta, n)
    out["n_rob"] = n_adv
    if n_adv:
        out["loss_adv"] = dis_adv_sum / n_adv
        out["rob"] = rob_correct / n_adv
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
