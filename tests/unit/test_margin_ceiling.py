"""Data-only robust-accuracy ceiling (src/analysis/margin.py).

The ceiling is what makes an impossible `rob` visible: no classifier can exceed it, so a
measured robust accuracy above it proves the attack under-searched.
"""

import numpy as np
import pytest

from src.analysis.margin import conflict_pairs, robust_accuracy_ceiling


def test_separated_classes_have_no_conflicts():
    """Classes further apart than 2*eps in Linf: every point can be robustly correct."""
    X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    y = np.array([0, 0, 1, 1])

    assert len(conflict_pairs(X, y, eps_abs=0.5)) == 0
    assert robust_accuracy_ceiling(X, y, eps_abs=0.5) == 1.0


def test_coincident_opposite_class_points_cap_at_half():
    """n/2 disjoint conflicting pairs force n/2 errors."""
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])

    assert robust_accuracy_ceiling(X, y, eps_abs=0.0) == pytest.approx(0.5)


def test_hand_computable_matching():
    """One class-1 point conflicts with two class-0 points: matching is 1, not 2.

    Points at 0.0 and 0.4 are class 0; 0.2 is class 1. At eps=0.15 the conflict
    threshold is 0.3, so 0.2 conflicts with both class-0 points, but the two edges share
    a vertex — a matching can use only one of them. The far point at 9.0 is isolated.
    """
    X = np.array([[0.0], [0.4], [0.2], [9.0]])
    y = np.array([0, 0, 1, 1])

    assert len(conflict_pairs(X, y, eps_abs=0.15)) == 2
    assert robust_accuracy_ceiling(X, y, eps_abs=0.15) == pytest.approx(1 - 1 / 4)


def test_ceiling_is_non_increasing_in_eps():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    ceilings = [robust_accuracy_ceiling(X, y, eps_abs=e) for e in (0.0, 0.02, 0.05, 0.2, 0.5)]
    assert all(a >= b - 1e-12 for a, b in zip(ceilings, ceilings[1:])), ceilings
    assert ceilings[0] == 1.0  # no duplicated points in a continuous sample


def test_norms_disagree_as_expected():
    """L2 conflicts are a subset of Linf conflicts at the same budget."""
    X = np.array([[0.0, 0.0], [0.25, 0.25]])
    y = np.array([0, 1])

    # Linf distance 0.25 <= 2*0.15; L2 distance 0.354 > 0.3.
    assert len(conflict_pairs(X, y, eps_abs=0.15, norm="inf")) == 1
    assert len(conflict_pairs(X, y, eps_abs=0.15, norm=2)) == 0


def test_multiclass_returns_nan():
    """Bipartite matching does not apply beyond two classes; skip rather than mislead."""
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])

    assert np.isnan(robust_accuracy_ceiling(X, y, eps_abs=0.1))


def test_max_points_guard_returns_nan():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(50, 2))
    y = (X[:, 0] > 0).astype(int)

    assert np.isnan(robust_accuracy_ceiling(X, y, eps_abs=0.1, max_points=10))
    assert not np.isnan(robust_accuracy_ceiling(X, y, eps_abs=0.1, max_points=50))


@pytest.mark.slow
def test_spirals_ceiling_is_below_the_reported_at_robustness():
    """The register's D1 finding, as a regression test.

    On the spirals test split the ceiling is ~0.68 at abs 0.2 and ~0.63 at abs 0.3, while
    `analysis/outputs/spirals/at/legendre/d10r6/seed_sweep_0206` reports 0.929 and 0.829.
    Measured `rob` >= true robust accuracy, so that gap is an attack failure, not a model
    property. If this test ever passes trivially the dataset generation has changed.
    """
    from omegaconf import OmegaConf

    from src.datahandler import DataHandler

    cfg = OmegaConf.load("configs/dataset/2Dtoy/spirals.yaml")
    OmegaConf.update(cfg, "overwrite", True, force_add=True)

    class _RangeOnly:
        """split_and_rescale only consults input_range; legendre is [-1, 1]."""
        input_range = (-1.0, 1.0)

    dh = DataHandler(cfg)
    dh.load()
    dh.split_and_rescale(_RangeOnly())
    X = dh.data["test"].numpy()
    y = dh.labels["test"].numpy()

    c02 = robust_accuracy_ceiling(X, y, eps_abs=0.2)
    c03 = robust_accuracy_ceiling(X, y, eps_abs=0.3)

    assert c02 == pytest.approx(0.68, abs=0.03)
    assert c03 == pytest.approx(0.63, abs=0.03)
    assert c02 < 0.929 and c03 < 0.829
