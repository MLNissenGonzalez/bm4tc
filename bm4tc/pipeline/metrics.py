"""Metric keys: the one place that turns plain metric names into logged keys (D11, D48).

Trainers and analysis produce plain names per split; this module adds the split and,
for budgeted metrics, the relative budget:

    key("objective", "valid")       -> "objective/valid"
    key("rob", "valid", 0.15)       -> "rob/valid/0.15"

Names, with what they mean on a split. Losses are in nats per variable (D86): the
objective is the weighted mean (1-s)·loss_dis + s·loss_x, s = β·n/N, over the N = n + 1
modelled variables (n features and the class).

    objective   the training objective (what selection minimises on valid), nats per
                variable; on train it excludes the norm penalty, so it compares with
                valid
    penalty     the norm-control penalty (train only; JEM: the energy penalty)
    loss_dis    -log p(c|x) on clean data, nats per label
    loss_x      -log p(x) / n on clean data, nats per feature (JEM: its estimate, with
                log Z estimated by a standardized SGLD chain; nan at beta=0)
    loss_adv    -log p(c|x_adv) on the attacked samples (AT only)
    acc         clean accuracy
    rob         robust accuracy, keyed by relative budget
    n_rob       number of attacked validation samples behind `rob`
    eps_rel     attack budget used for training this epoch (follows the curriculum)

Diagnostics (`norm/*` for the MPS, `sgld/*` for JEM) pass through unchanged; the
MPS's `norm/log_Z_*` and `norm/log_amp_sq_*` are per site (divided by N),
`norm/log_Z_headroom` is absolute.

Analysis (`analyse`, on test) uses the same rule. Budget 0 is clean data, so a
curve over budgets starts at the clean value; the last part names the defence
setting: q{percentile} (detection threshold, calibrated on valid, D2),
d{radius} (likelihood purification), k{sweeps} (Gibbs or SGLD purification):

    acc, loss_dis, loss_x                clean, on the full test split
    rob_ceiling/test/{eps}               data-only bound on robust accuracy (two classes)
    rob/test/{eps}                       accuracy on the PGD examples
    log_px/test/{eps}                    mean log p(x) / n of those examples (0: clean)
    detect/test/{eps}/q{p}               fraction flagged (0: the clean false-positive rate)
    err_detected, err_passed             error rate among flagged / passed examples
    purify/test/{eps}/d{r}               accuracy after likelihood purification (0: clean)
    recovery/test/{eps}/d{r}             fraction of misclassified examples fixed by it
    purify_gibbs, recovery_gibbs         the same for Gibbs purification (MPS), /k{n},
                                         on a fixed test subsample
    purify_sgld, recovery_sgld           the same for SGLD purification (JEM)

JEM has no loss_x on test (no exact log Z), and its log_px is unnormalised.

`_joint` on a name (`rob_joint`, `detect_joint`, ...) means the joint attack on
class and log p(x) (JOINT_PGD) instead of PGD.
"""
from typing import Mapping, Optional

from bm4tc.core.embeddings import fmt_budget

SELECTION = "objective/valid"  # selection = argmin over validation events (D8)


def key(name: str, split: str, budget: Optional[float] = None,
        setting: Optional[str] = None) -> str:
    """``{name}/{split}``, plus ``/{budget}`` for budgeted metrics, plus
    ``/{setting}`` for a defence setting (``q5``, ``d0.1``, ``k3``)."""
    out = f"{name}/{split}"
    if budget is not None:
        out = f"{out}/{fmt_budget(budget)}"
    return out if setting is None else f"{out}/{setting}"


def flatten(split: str, values: Mapping) -> dict:
    """Keys for one split. A dict value maps relative budgets to values."""
    out = {}
    for name, value in values.items():
        if isinstance(value, Mapping):
            out.update({key(name, split, b): v for b, v in value.items()})
        else:
            out[key(name, split)] = value
    return out


def flatten_epoch(record: Mapping) -> dict:
    """An epoch record ``{"train": {...}, "valid": {...}, "diagnostics": {...}}`` as logged keys."""
    out = {}
    for split, values in record.items():
        if split == "diagnostics":
            out.update(values)
        else:
            out.update(flatten(split, values))
    return out
