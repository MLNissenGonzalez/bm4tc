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
that robust. It is tight on the sample, so it is optimistic about what a model can reach
on the data law (notes:adaptive §2.1).

``selective`` bounds a classifier that may reject (the matching bound at ``eps_abs / 2``),
and :func:`population_ceiling` brackets the bound for a known 2-D data law instead of a
sample (notes:adaptive §2.2, §2.3).

Budget convention (GUIDE.md, glossary): ``eps_abs`` is an ABSOLUTE
model-domain budget, matching ``eval_rob(eps_abs=)``. Callers convert with
``rel_to_abs(eps_rel, range_size_of(cbm))``. It must be measured in the same coordinates as
``X`` — i.e. after ``DataHandler.split_and_rescale``.
"""

import logging
from typing import Tuple, Union

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
    selective: bool = False,
) -> float:
    """Largest robust accuracy any classifier can attain on ``(X, y)`` at ``eps_abs``.

    With ``selective``, the bound on selective robust accuracy (a classifier that may
    reject: the clean point accepted and correct, the attacked one rejected or correct).
    Two opposite-class points within ``eps_abs`` of each other cannot both be robust in
    that sense, which is the matching bound at ``eps_abs / 2`` (notes:adaptive §2.2).

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

    pairs = conflict_pairs(X, y, eps_abs / 2 if selective else eps_abs, norm=norm)
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


def population_ceiling(
    mass0: np.ndarray,
    mass1: np.ndarray,
    cell: float,
    eps_abs: float,
    selective: bool = False,
) -> Tuple[float, float]:
    """Bracket on the best robust accuracy any classifier attains on a 2-D data law.

    With equal class priors the robust Bayes accuracy is ``1 - F/2``, where ``F`` is the
    largest mass that can be coupled between the two class laws with every pair at most
    ``2 * eps_abs`` apart in L∞ (``eps_abs`` with ``selective``, the bound for a
    classifier that may reject; notes:adaptive §2). On a grid of square cells ``F`` is
    a maximum flow from class-0 cells to class-1 cells within reach. A cell blurs a
    distance by up to ``cell`` per axis, so connecting the cells whose farthest points
    are within reach gives a lower bound on ``F`` and connecting those whose nearest
    points are gives an upper bound: the bracket is certain, and shrinks with ``cell``.

    Args:
        mass0, mass1: ``(nx, ny)`` probability of each class in each cell, each summing
            to 1 (the class's whole law, tails included in the outer cells).
        cell: the cells' side, in the same units as ``eps_abs``.
        eps_abs: absolute attack budget.
        selective: bound selective robust accuracy (reach ``eps_abs`` instead of
            ``2 * eps_abs``).

    Returns:
        ``(lo, hi)`` with ``lo <= acc* <= hi``.
    """
    reach = (1.0 if selective else 2.0) * float(eps_abs) / float(cell)
    k_sure = int(np.floor(reach - 1 + 1e-9))       # every pair of points within reach
    k_maybe = int(np.floor(reach + 1 + 1e-9))      # some pair of points within reach
    flow_lo, slack = _coupled_mass(mass0, mass1, k_sure)
    flow_hi, _ = _coupled_mass(mass0, mass1, k_maybe)
    return max(0.0, 1 - (flow_hi + slack) / 2), min(1.0, 1 - (flow_lo - slack) / 2)


_SCALE = 10 ** 9   # probability -> integer capacity (scipy's max flow is int32)


def _coupled_mass(mass0: np.ndarray, mass1: np.ndarray, k: int) -> Tuple[float, float]:
    """Largest mass coupled between cells at most ``k`` cells apart on each axis, and
    the rounding error of the integer capacities.

    The move "≤ k cells in x and ≤ k in y" goes through a middle layer, one node per
    cell: first ≤ k along x, then ≤ k along y. That is exact (the middle nodes have no
    capacity) and costs O(cells · k) edges instead of O(cells · k²).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    if k < 0:
        return 0.0, 0.0
    nx, ny = mass0.shape
    cap0 = np.rint(mass0 * _SCALE).astype(np.int64)
    cap1 = np.rint(mass1 * _SCALE).astype(np.int64)
    slack = (np.count_nonzero(mass0) + np.count_nonzero(mass1)) / (2 * _SCALE)

    # Nodes: source, class-0 cells, middle cells, class-1 cells, sink.
    cells = nx * ny
    src, a, m, b, sink = 0, 1, 1 + cells, 1 + 2 * cells, 1 + 3 * cells
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    rows, cols, caps = [], [], []

    def add(r, c, cap):
        rows.append(r), cols.append(c), caps.append(cap)

    on0, on1 = cap0.ravel() > 0, cap1.ravel() > 0
    flat = np.arange(cells)
    add(np.full(on0.sum(), src), a + flat[on0], cap0.ravel()[on0])
    add(b + flat[on1], np.full(on1.sum(), sink), cap1.ravel()[on1])
    big = np.iinfo(np.int32).max
    for d in range(-k, k + 1):
        # class-0 cell (i, j) -> middle cell (i + d, j)
        ok = on0 & (ii + d >= 0) & (ii + d < nx)
        add(a + flat[ok], m + flat[ok] + d * ny, np.full(ok.sum(), big))
        # middle cell (i, j) -> class-1 cell (i, j + d)
        ok = (jj + d >= 0) & (jj + d < ny)
        ok[ok] &= on1[flat[ok] + d]
        add(m + flat[ok], b + flat[ok] + d, np.full(ok.sum(), big))

    graph = csr_matrix(
        (np.concatenate(caps).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))),
        shape=(sink + 1, sink + 1),
    )
    flow = maximum_flow(graph, src, sink, method="dinic").flow_value
    return flow / _SCALE, slack
