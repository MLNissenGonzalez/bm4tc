"""Data-only upper bound on robust accuracy.

If two points of *opposite* class lie within ``2 * eps_abs`` of each other, some single
input is within ``eps_abs`` of both, so no classifier — deterministic or randomized — can
be robustly correct on both: at least one of the two is an error at that budget. Picking a
set of such pairs that share no point (a *matching* in the conflict graph) therefore forces
at least one error per pair, giving

    robust accuracy  <=  1 - |maximum matching| / n

For two classes the conflict graph is bipartite, so the maximum matching is computable
exactly in polynomial time and the bound is the tightest one obtainable this way (König).
The bound is a property of the *data*, not of any model: a measured ``rob`` above it means
the attack failed to find adversarial examples that provably exist, never that the model is
that robust.

Budget convention (see "Budget vocabulary" in CLAUDE.md): ``eps_abs`` is an ABSOLUTE
model-domain budget, matching ``eval_rob(eps_abs=)``. Callers convert with
``rel_to_abs(eps_rel, range_size_of(cbm))``. It must be measured in the same coordinates as
``X`` — i.e. after ``DataHandler.split_and_rescale``.
"""

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Above this many points the pair query is not worth its cost; the ceiling is a sanity
# check, not a headline metric.
DEFAULT_MAX_POINTS = 20_000


def _minkowski_p(norm: Union[int, str]) -> float:
    """Map this repo's norm spelling onto scipy's Minkowski ``p``."""
    if norm in ("inf", "Linf", float("inf")):
        return float("inf")
    if isinstance(norm, (int, float)) and norm >= 1:
        return float(norm)
    raise ValueError(f"unsupported norm {norm!r}")


def conflict_pairs(
    X: np.ndarray,
    y: np.ndarray,
    eps_abs: float,
    norm: Union[int, str] = "inf",
) -> np.ndarray:
    """Index pairs ``(i, j)`` of opposite class within ``2 * eps_abs`` of each other.

    Args:
        X: ``(n, d)`` inputs, in the same coordinates as ``eps_abs``.
        y: ``(n,)`` integer labels.
        eps_abs: absolute attack budget.
        norm: ``"inf"`` (the attack norm used throughout) or an integer ``p >= 1``.

    Returns:
        ``(m, 2)`` integer array; empty ``(0, 2)`` when no pair conflicts.
    """
    from scipy.spatial import cKDTree

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if len(y) != len(X):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")

    pairs = cKDTree(X).query_pairs(
        r=2.0 * float(eps_abs), p=_minkowski_p(norm), output_type="ndarray"
    )
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int)
    return pairs[y[pairs[:, 0]] != y[pairs[:, 1]]]


def robust_accuracy_ceiling(
    X: np.ndarray,
    y: np.ndarray,
    eps_abs: float,
    norm: Union[int, str] = "inf",
    max_points: int = DEFAULT_MAX_POINTS,
) -> float:
    """Largest robust accuracy any classifier can attain on ``(X, y)`` at ``eps_abs``.

    Args:
        X: ``(n, d)`` inputs, in the same coordinates as ``eps_abs``.
        y: ``(n,)`` integer labels. Exactly two classes; see Returns.
        eps_abs: absolute attack budget.
        norm: ``"inf"`` or an integer ``p >= 1``.
        max_points: skip (return ``nan``) above this many points.

    Returns:
        The ceiling in ``[0, 1]``, or ``nan`` when it is not computed: more than two
        classes (the matching bound still holds, but a non-bipartite maximum matching is
        needed and no multiclass dataset here calls for it), fewer than two points, or
        ``n > max_points``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n = len(X)
    classes = np.unique(y)

    if n < 2:
        return float("nan")
    if len(classes) != 2:
        logger.info(
            f"robust_accuracy_ceiling: {len(classes)} classes; bipartite matching does "
            f"not apply, skipping"
        )
        return float("nan")
    if n > max_points:
        logger.info(f"robust_accuracy_ceiling: {n} > max_points={max_points}, skipping")
        return float("nan")

    pairs = conflict_pairs(X, y, eps_abs, norm=norm)
    if len(pairs) == 0:
        return 1.0

    # Bipartite graph: rows are the class-0 endpoints, columns the class-1 endpoints.
    # Only points that appear in some conflict pair get a row/column.
    left_mask = y[pairs[:, 0]] == classes[0]
    left = np.where(left_mask, pairs[:, 0], pairs[:, 1])
    right = np.where(left_mask, pairs[:, 1], pairs[:, 0])

    left_ids, left_idx = np.unique(left, return_inverse=True)
    right_ids, right_idx = np.unique(right, return_inverse=True)

    graph = csr_matrix(
        (np.ones(len(pairs), dtype=np.int8), (left_idx, right_idx)),
        shape=(len(left_ids), len(right_ids)),
    )
    matched = maximum_bipartite_matching(graph, perm_type="column")
    matching_size = int((matched >= 0).sum())

    return 1.0 - matching_size / n
