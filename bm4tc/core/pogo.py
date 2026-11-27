"""Training on the isometric form of the MPS Born machine, by POGO (D192, D195).

Every core left of the class core is kept a left isometry, every core right of it a right
isometry, and the class core (the root) a unit vector. Seen from the root, each core is then
an isometry towards its parent leg (the bond that leads to the class core):

    X Xᴴ = I,   X = the core as the matrix (parent leg) × (its other legs),

and the root, as one row, has ‖X‖ = 1. Then Z = Σ_{x,c} |ψ(x, c)|² = ‖root‖² = 1: the norm
is held by the constraint, not by a penalty, and the gauge is fixed once (the canonical form
centred at the class core), so the optimiser's state never has to follow a change of gauge
(the failure of Adam with the balanced gauge, D191).

POGO, the Proximal One-step Geometric Orthoptimizer (Javaloy and Vergari, arXiv
2602.14656; ported from tpm-tn-benchmark's ``pogo.py``): the base optimiser proposes a
step X → X − Δ, Δ scaled to ‖Δ‖_F ≤ 1 per core, and the core moves instead to

    M = X − ½(X Xᴴ Δ − X Δᴴ X)         along the tangent space,
    X ← M + ½(I − M Mᴴ) M              the landing step (λ = ½),

which keeps X Xᴴ − I of the order of the step's square, step after step. λ is fixed at ½
(tpm_tn's default), not the root of the landing polynomial, so the step has no host sync
and a CUDA graph can capture it (D90).

The paper (Alg. 1, Def. 1) gives the base optimiser the Euclidean gradient of the NLL with
log Z read from the centre, log ‖root‖² (``log_Z_from_centre``: no norm contraction;
``trainer.riemannian: centre``), and asks it to be linear in the gradient up to a scalar:
VAdam (the paper's and ours), SGD or ``normalized_gd``. Then the gradient's part normal to
the manifold, which the two log Z reads differ in, is projected away. POGO takes any base
optimiser all the same (D210): with element-wise Adam, which does not commute with the
projection, that normal part leaks into the step and the log Z read matters (D192's
variant contracts the norm, ``trainer.riemannian: contracted``). The landing step keeps
the cores isometries either way. Evaluation, and so the selection, contracts log Z
exactly. The cap on ‖Δ‖ is not the paper's (it asks for ξ < 1 through lr or a normalising
optimiser): a scalar per core, so a linear step stays linear; VAdam below lr ≈ 1 hardly
meets it, plain SGD diverges without it.

The edge cores are seen through the fixed obc boundary vectors (D87), as in ``gauge.py``;
the components the boundary vector does not reach are never read and never written. A
bond must not be wider than the cores on its outer side can fill (an edge core of d rows is
no isometry onto r > d components): build the model with ``trim_bonds`` (D192).
"""

from typing import List, Optional

import torch

# A core counts as an isometry at the start when ‖X Xᴴ − I‖_F is below this (complex64's
# √ε with a margin: ``isometric_start_`` reaches ≈ 1e-6).
_START_DEFECT = 1e-3


class _Layout:
    """Which leg of each core leads to the class core, and the boundary vectors."""

    def __init__(self, cbm):
        self.cores = [node.tensor for node in cbm._mats_env]
        self.n = len(self.cores)
        self.root = cbm.out_position
        obc = cbm._boundary == "obc"
        self.left = cbm._left_node.tensor.detach() if obc else None
        self.right = cbm._right_node.tensor.detach() if obc else None

    def parent(self, v: int) -> Optional[int]:
        """Core v's leg towards the root (2 left of it, 0 right of it), None for the root."""
        return None if v == self.root else (2 if v < self.root else 0)

    def live(self, v: int, w: torch.Tensor) -> torch.Tensor:
        """Core v (or a tensor of its shape) with the boundary vectors contracted in, the
        contracted leg kept with size 1."""
        if v == 0 and self.left is not None:
            w = torch.einsum("l,lpr->pr", self.left.to(w.device, w.dtype), w).unsqueeze(0)
        if v == self.n - 1 and self.right is not None:
            w = torch.einsum("lpr,r->lp", w, self.right.to(w.device, w.dtype)).unsqueeze(-1)
        return w

    def set_live_(self, v: int, new: torch.Tensor) -> None:
        """Core v's live part set to ``new`` (the shape ``live`` gives), in place; the
        components no boundary vector reaches kept as they are."""
        core = self.cores[v].data
        if v == 0 and self.left is not None:
            b = self.left.to(core.dtype)
            delta = (new - self.live(v, core))[0]
            core += torch.einsum("l,pr->lpr", b.conj() / b.abs().square().sum(), delta)
        elif v == self.n - 1 and self.right is not None:
            b = self.right.to(core.dtype)
            delta = (new - self.live(v, core))[..., 0]
            core += torch.einsum("lp,r->lpr", delta, b.conj() / b.abs().square().sum())
        else:
            core.copy_(new)

    def rows(self, v: int, w: torch.Tensor) -> torch.Tensor:
        """The live core as the matrix (parent leg) × (other legs); the root as one row."""
        parent = self.parent(v)
        if parent is None:
            return w.reshape(1, -1)
        return w.movedim(parent, 0).reshape(w.shape[parent], -1)

    def from_rows(self, v: int, x: torch.Tensor, shape) -> torch.Tensor:
        parent = self.parent(v)
        if parent is None:
            return x.reshape(shape)
        moved = [shape[parent]] + [s for k, s in enumerate(shape) if k != parent]
        return x.reshape(moved).movedim(0, parent)


def isometry_defects(cbm) -> List[float]:
    """‖X Xᴴ − I‖_F of every core (the root: |‖root‖² − 1|), in chain order."""
    layout = _Layout(cbm)
    defects = []
    with torch.no_grad():
        for v, core in enumerate(layout.cores):
            x = layout.rows(v, layout.live(v, core.detach()))
            eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
            defects.append(torch.linalg.norm(x @ x.mH - eye).item())
    return defects


def log_Z_from_centre(cbm) -> torch.Tensor:
    """log Z = log ‖root‖², with its gradient (it reaches the class core only): exact on
    the isometric form, off it up to the isometry defects."""
    layout = _Layout(cbm)
    return layout.live(layout.root, layout.cores[layout.root]).abs().square().sum().log()


@torch.no_grad()
def isometric_start_(cbm) -> None:
    """The cores brought into the isometric form, in place: QR sweeps from both edges
    towards the class core (a gauge change: ψ unchanged), then the root normalised (ψ
    divided by √Z: p(c | x) unchanged, Z = 1). Computed in complex128 on the CPU, written
    back once. ValueError, naming the bond and ``trim_bonds``, where a bond is wider than
    the cores on its outer side can fill. The caller invalidates the model's log Z caches."""
    layout = _Layout(cbm)
    work = [layout.live(v, c.detach().to("cpu", torch.complex128)) for v, c in enumerate(layout.cores)]
    for v in range(layout.root):                       # left isometries, R into the next core
        r_l, d, r_r = work[v].shape
        _check_support(v, v, r_r, r_l * d)
        q, r = torch.linalg.qr(work[v].reshape(r_l * d, r_r))
        work[v] = q.reshape(r_l, d, r_r)
        work[v + 1] = torch.einsum("rs,spq->rpq", r, work[v + 1])
    for v in range(layout.n - 1, layout.root, -1):     # right isometries, R into the previous
        r_l, d, r_r = work[v].shape
        _check_support(v, v - 1, r_l, d * r_r)
        q, r = torch.linalg.qr(work[v].reshape(r_l, d * r_r).mH)
        work[v] = q.mH.reshape(r_l, d, r_r)
        work[v - 1] = torch.einsum("lps,rs->lpr", work[v - 1], r.conj())
    work[layout.root] = work[layout.root] / torch.linalg.norm(work[layout.root])
    for v, w in enumerate(work):
        core = layout.cores[v]
        layout.set_live_(v, w.to(core.device, core.dtype))


def _check_support(v: int, bond: int, width: int, reach: int) -> None:
    if width > reach:
        raise ValueError(f"trainer.riemannian: bond {bond} has {width} components, but core {v} "
                         f"on its outer side reaches only {reach}, so it cannot be an isometry; "
                         f"build the model with born.init_kwargs.trim_bonds: true")


def pogo_step(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """POGO's new point from X (p, n) and the base optimiser's step Δ (X − Δ is where the
    base optimiser alone would go), Δ capped at ‖Δ‖_F ≤ 1 first: the step's analysis assumes
    ξ < 1, and the NLL's gradient is unbounded (−2∇ψ/ψ as ψ → 0), so plain SGD needs the
    cap; a step within it is unchanged."""
    delta = delta / torch.linalg.norm(delta).clamp(min=1.0)
    m = x - 0.5 * ((x @ x.mH) @ delta - (x @ delta.mH) @ x)
    return m + 0.5 * (m - (m @ m.mH) @ m)


class POGO:
    """The base optimiser's steps, turned into POGO's on every core of ``cbm``.

    ``step()``: the base optimiser steps; then each core's step Δ = X − X′ (X before, X′
    after, as the matrix of ``_Layout.rows``) is replaced by ``pogo_step``. The cores must
    be in the isometric form already (``isometric_start_``): ValueError otherwise, and for a
    trainable parameter of the base optimiser that is not a core of ``cbm``. Any base
    optimiser (the module docstring says what Def. 1 asks of it). Its state is its own and
    stays where it is (no gauge ever changes)."""

    def __init__(self, cbm, base: torch.optim.Optimizer):
        self.base = base
        self._layout = _Layout(cbm)
        cores = {id(c) for c in self._layout.cores}
        for group in base.param_groups:
            for p in group["params"]:
                if p.requires_grad and id(p) not in cores:
                    raise ValueError(f"POGO: a trainable parameter of shape {tuple(p.shape)} is "
                                     "not a core of the model")
        bad = [v for v, e in enumerate(isometry_defects(cbm)) if not e < _START_DEFECT]
        if bad:
            raise ValueError(f"POGO: cores {bad} are not isometries towards the class core; "
                             "start from isometric_start_")

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        layout = self._layout
        before = [layout.live(v, c).clone() for v, c in enumerate(layout.cores)]
        loss = self.base.step(closure)
        for v, (core, old) in enumerate(zip(layout.cores, before)):
            x = layout.rows(v, old)
            new = pogo_step(x, x - layout.rows(v, layout.live(v, core)))
            layout.set_live_(v, layout.from_rows(v, new, old.shape))
        return loss
