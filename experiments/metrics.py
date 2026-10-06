"""Metric keys: the one place that turns plain metric names into logged keys (D11, D48).

Trainers and analysis produce plain names per split; this module adds the split and,
for budgeted metrics, the relative budget:

    key("objective", "valid")       -> "objective/valid"
    key("rob", "valid", 0.15)       -> "rob/valid/0.15"

Names, with what they mean on a split:

    objective   the training objective (what selection minimises on valid); on train
                it excludes the norm penalty, so it compares with valid
    penalty     the norm-control penalty (train only)
    loss_dis    -log p(c|x) on clean data
    loss_gen    -log p(x, c) on clean data
    loss_adv    -log p(c|x_adv) on the attacked samples (AT only)
    acc         clean accuracy
    rob         robust accuracy, keyed by relative budget
    n_rob       number of attacked validation samples behind `rob`
    eps_rel     attack budget used for training this epoch (follows the curriculum)

`norm/*` diagnostics pass through unchanged.
"""
from typing import Mapping, Optional

from src.utils.embeddings import fmt_budget

SELECTION = "objective/valid"  # selection = argmin over validation events (D8)


def key(name: str, split: str, budget: Optional[float] = None) -> str:
    """``{name}/{split}``, plus ``/{budget}`` for budgeted metrics."""
    out = f"{name}/{split}"
    return out if budget is None else f"{out}/{fmt_budget(budget)}"


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
    """An epoch record ``{"train": {...}, "valid": {...}, "norm": {...}}`` as logged keys."""
    out = {}
    for split, values in record.items():
        if split == "norm":
            out.update(values)
        else:
            out.update(flatten(split, values))
    return out
