"""Membership inference attack (MIA) against a classifier with ``class_probabilities``.

Standalone (D1): not part of the pipeline and not in the paper. It answers one
question: can an attacker tell training samples (members) from test samples
(non-members) by looking at p(c|x)?

Two attacks on six confidence features of p(c|x):
- **Logistic regression** on all features, fitted and scored in-sample on balanced
  member/non-member sets (an upper bound; chance = 0.5).
- **Worst-case threshold**, per feature: the threshold that maximises balanced
  accuracy on the full sets. An oracle bound, since a real attacker cannot tune it
  without membership labels.

With ``adv_eps_abs`` set, the worst-case threshold attack is repeated on features of
PGD adversarial examples.

    result = membership_inference(model, train_loader, test_loader, device)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.utils.evasion import ProjectedGradientDescent
from src.utils.train import CriterionConfig

logger = logging.getLogger(__name__)

FEATURES = ("max_prob", "entropy", "correct_prob", "loss", "margin", "modified_entropy")
# Features where a lower value indicates membership; their sign is flipped for scoring.
_LOWER_IS_MEMBER = ("loss", "entropy")


@dataclass
class MIAResult:
    """Attack outcome. Worst-case dicts map a feature name to its best balanced accuracy."""
    attack_accuracy: float
    auc_roc: float
    worst_case: Dict[str, float]
    adversarial_worst_case: Optional[Dict[str, float]] = None


def features(probs: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """The six confidence features of p(c|x), shape (batch, 6), in ``FEATURES`` order.

    ``correct_prob`` and ``loss`` use the true labels (a worst-case attacker).
    """
    eps = 1e-10
    clamped = probs.clamp(min=eps, max=1.0 - eps)
    idx = torch.arange(probs.shape[0], device=probs.device)
    entropy = -(clamped * torch.log(clamped)).sum(dim=-1)
    top2 = probs.sort(dim=-1, descending=True)[0]
    columns = {
        "max_prob": probs.max(dim=-1)[0],
        "entropy": entropy,
        "correct_prob": probs[idx, labels],
        "loss": -torch.log(clamped[idx, labels]),
        "margin": top2[:, 0] - top2[:, 1],
        "modified_entropy": 1.0 - entropy / np.log(probs.shape[-1]),
    }
    return np.hstack([columns[name].cpu().numpy().reshape(-1, 1) for name in FEATURES])


def worst_case_threshold(members: np.ndarray, non_members: np.ndarray) -> Dict[str, float]:
    """Best balanced accuracy, (TPR + TNR) / 2, of a single-feature threshold attack.

    For each feature, every midpoint between consecutive unique scores is tried as the
    threshold; a sample is called a member when its score is at or above it.
    """
    n_members = len(members)
    result = {}
    for i, name in enumerate(FEATURES):
        scores = np.concatenate([members[:, i], non_members[:, i]])
        if name in _LOWER_IS_MEMBER:
            scores = -scores
        unique = np.unique(scores)
        if len(unique) <= 1:
            result[name] = n_members / len(scores)
            continue
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        preds = (midpoints[:, None] <= scores[None, :]).astype(np.float64)
        tpr = preds[:, :n_members].mean(axis=1)
        tnr = (1 - preds[:, n_members:]).mean(axis=1)
        result[name] = float(((tpr + tnr) / 2).max())
    return result


def _features_of(model, loader: DataLoader, device, pgd=None, eps_abs=None) -> np.ndarray:
    model.to(device)
    rows = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if pgd is not None:
            x = pgd.generate(model, x, y, eps_abs, device)
        with torch.no_grad():
            rows.append(features(model.class_probabilities(x), y))
    return np.vstack(rows)


def membership_inference(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    adv_eps_abs: Optional[float] = None,
    adv_num_steps: int = 20,
    adv_norm: Union[str, int] = "inf",
    seed: int = 42,
) -> MIAResult:
    """Run both attacks; members come from ``train_loader``, non-members from ``test_loader``.

    Rows with non-finite features are dropped, and members are subsampled to the
    number of non-members so that chance is 0.5. ``adv_eps_abs`` is an absolute
    (model-domain) budget; None skips the adversarial attack.
    """
    members = _features_of(model, train_loader, device)
    non_members = _features_of(model, test_loader, device)

    member_ok = np.isfinite(members).all(axis=1)
    non_member_ok = np.isfinite(non_members).all(axis=1)
    if not (member_ok.all() and non_member_ok.all()):
        logger.warning(
            f"MIA: dropping {int((~member_ok).sum())} member and "
            f"{int((~non_member_ok).sum())} non-member rows with non-finite features."
        )
    members, non_members = members[member_ok], non_members[non_member_ok]
    if len(members) == 0 or len(non_members) == 0:
        raise ValueError("No finite feature rows remain after NaN filtering.")

    subsample = None
    if len(members) > len(non_members):
        subsample = np.random.default_rng(seed).permutation(len(members))[: len(non_members)]
        members = members[subsample]

    X = np.vstack([members, non_members])
    y = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
    clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X, y)

    result = MIAResult(
        attack_accuracy=float(accuracy_score(y, clf.predict(X))),
        auc_roc=float(roc_auc_score(y, clf.predict_proba(X)[:, 1])),
        worst_case=worst_case_threshold(members, non_members),
    )

    if adv_eps_abs is not None:
        pgd = ProjectedGradientDescent(
            norm=adv_norm,
            criterion=CriterionConfig(name="nll", kwargs=None),
            num_steps=adv_num_steps,
            step_size=None,
            random_start=True,
        )
        adv_members = _features_of(model, train_loader, device, pgd, adv_eps_abs)[member_ok]
        adv_non_members = _features_of(model, test_loader, device, pgd, adv_eps_abs)[non_member_ok]
        if subsample is not None:
            adv_members = adv_members[subsample]
        result.adversarial_worst_case = worst_case_threshold(adv_members, adv_non_members)

    return result
