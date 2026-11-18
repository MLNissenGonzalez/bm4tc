"""Gauge control of the MPS Born machine during training (D190).

A gauge transformation on the bond between cores v and v + 1,

    A ← A S,   B ← B S⁻ᵀ,   S invertible,

with A core v as the matrix (left, phys) × bond and B core v + 1 as (phys, right) × bond,
leaves ψ, log Z and p(c | x) unchanged. It reaches training only through the optimiser:
Adam's element-wise second moments make it move the cores along the gauge orbit, which
plain gradient descent would not (notes:gauge). ``gauge_step_`` brings the cores towards
the gauge of minimal node norm after an optimiser step; ``gauge_diagnostics`` measures how
far they are from it.

    kind       what one step does
    none       nothing: the gauge is where the optimiser leaves it
    scalar     every core rescaled to the geometric mean of the core norms: the minimum of
               Σ_v ‖W_v‖² over scalar gauges, in closed form. A positive diagonal gauge, so
               both of the optimiser's moments move with it exactly.
    balanced   towards A†A = (B†B)ᵀ at every bond, the minimum of Σ_v ‖W_v‖² over the whole
               orbit (Furman et al. 2026, Def. 4.1, Prop. F.10). ``rate`` None: Gauss-Seidel
               sweeps of exact one-bond balances, S = Y^½ with Y P Y = Q, to ``_TOLERANCE``;
               ``rate`` > 0: one step S = exp(−τ D) on every bond from the same point,
               D = P − Q, τ = rate / (2 (‖P‖₂ + ‖Q‖₂)). First moments move as gradients do;
               second moments are kept (no exact transform under a non-diagonal S).

The edge cores are balanced through the fixed boundary vectors (D87): their live slice is
what enters P and Q; the transformation is applied to the whole core.

Dead and live components. Near the edges a bond is wider than what the cores on one side
can reach (bond 0: core 0 has d rows, the bond r components), so the other side's
components along that side's exact null space (singular values below ``_NULL`` of the
largest) carry nothing of ψ, now or later: the exact balance sets them to zero first, the
minimum over the orbit's closure (the transformation is then a projection, not a gauge, and
ψ is unchanged). A component above that but below ``_LIVE`` on either side, e.g. randn_eye's
1e-9 noise directions at initialisation, is live for the gradient: it is left as it is,
never zeroed (its parameters would get zero gradient for ever) and never amplified. The
exact balance and the defect act on the block live on both sides (exposed by a unitary of
the bond that the balance undoes, as tpm_tn's ``_live``).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

KINDS = ("none", "scalar", "balanced")

# The exact sweeps stop once every bond's defect is at most this (the cores are float32 /
# complex64, so a defect much below 1e-5 is rounding), or after _MAX_SWEEPS.
_TOLERANCE = 1e-5
_MAX_SWEEPS = 200
_NULL = 1e-12
_LIVE = 1e-6
_STALL = 1e-9

# The optimiser state entries that move with the cores: first moments transform as the
# gradient, second moments (element-wise squares of gradients) only under a diagonal gauge.
_FIRST_MOMENTS = ("exp_avg", "momentum_buffer", "grad_avg")
_SECOND_MOMENTS = ("exp_avg_sq", "max_exp_avg_sq", "square_avg")

# The keys gauge_diagnostics returns (the training log's `diagnostics`).
GAUGE_DIAGNOSTICS = ("gauge/log_norm", "gauge/log_norm_over_scalar", "gauge/defect_median",
                     "gauge/defect_max")


@dataclass
class GaugeConfig:
    """``trainer.gauge``: the gauge step after every ``every``-th optimiser step."""
    kind: str = "none"
    every: int = 1
    rate: Optional[float] = None    # balanced only: None = exact sweeps, > 0 = one soft step

    def __post_init__(self):
        check(self)


def check(cfg) -> None:
    """ValueError, naming the options, for an unknown kind, ``every`` < 1, or a ``rate``
    that does not fit the kind (only ``balanced`` has a soft step)."""
    if cfg.kind not in KINDS:
        raise ValueError(f"trainer.gauge.kind must be one of {KINDS}, got {cfg.kind!r}")
    if cfg.every < 1:
        raise ValueError(f"trainer.gauge.every must be >= 1, got {cfg.every}")
    if cfg.rate is not None:
        if cfg.kind != "balanced":
            raise ValueError(f"trainer.gauge.rate is the soft step of kind 'balanced'; "
                             f"kind {cfg.kind!r} takes none (got rate={cfg.rate})")
        if not cfg.rate > 0:
            raise ValueError(f"trainer.gauge.rate must be > 0, got {cfg.rate}")


# ----------------------------------------------------------------------
# The interface
# ----------------------------------------------------------------------

@torch.no_grad()
def gauge_step_(cbm, optimizer, cfg) -> int:
    """One gauge step of kind ``cfg.kind`` on ``cbm``'s cores, in place, with the state of
    ``optimizer`` (None for none) carried along; ψ is unchanged. Returns the sweeps it took
    (1 for the one-pass steps, 0 for ``none``). The caller invalidates the model's log Z
    caches (the cores are written through ``.data``)."""
    if cfg.kind == "none":
        return 0
    if cfg.kind == "scalar":
        _scalar_(cbm, optimizer)
        return 1
    net = _Cores(cbm, optimizer)
    if cfg.rate is not None:
        net.soft_(cfg.rate)
        return 1
    return net.sweep_()


@torch.no_grad()
def gauge_diagnostics(cbm) -> dict:
    """How far the cores are from the minimal-norm gauge, without changing them:
    log Σ_v ‖W_v‖², its excess over the scalar minimum (log N − log N_scalar, ≥ 0), and
    the median and largest bond defect ‖P − Q‖_F / ‖P‖_F."""
    net = _Cores(cbm, None)
    logs = net.log_norms2().real
    log_n = torch.logsumexp(logs, 0).item()
    log_scalar = math.log(len(logs)) + logs.mean().item()
    defects = torch.tensor([net.defect(b) for b in range(net.n - 1)])
    return {"gauge/log_norm": log_n, "gauge/log_norm_over_scalar": log_n - log_scalar,
            "gauge/defect_median": defects.median().item(),
            "gauge/defect_max": defects.max().item()}


# ----------------------------------------------------------------------
# The cores of one model, as the gauge sees them
# ----------------------------------------------------------------------

class _Cores:
    """The cores (r_l, d, r_r) of ``cbm._mats_env`` with the obc boundary vectors, and the
    optimiser state of each. A step works on complex128 copies, composing each bond's
    transformation, and ``commit_`` writes the cores and the first moments back once, so
    the float32 / complex64 cores take one rounding per step, not one per bond visit."""

    def __init__(self, cbm, optimizer):
        self.params = [node.tensor for node in cbm._mats_env]
        # On the CPU: the bond matrices are small (r × r, (r d) × r), so float64 there beats
        # the GPU's launch-bound complex128 kernels (measured on G21G01: minutes per sweep).
        self.work = [p.detach().to('cpu', torch.complex128) for p in self.params]
        self.n = len(self.params)
        obc = cbm._boundary == "obc"
        self.left = cbm._left_node.tensor.detach().to('cpu', torch.complex128) if obc else None
        self.right = cbm._right_node.tensor.detach().to('cpu', torch.complex128) if obc else None
        eye = lambda b: torch.eye(self.work[b].shape[-1], dtype=torch.complex128,
                                  device=self.work[b].device)
        # per bond, the composed matrices of its two sides: A ← A M_A, B ← B M_B (a gauge
        # S gives M_A = S, M_B = S⁻ᵀ; a projection onto the other side's range, M = Π̄)
        self.m_a = [eye(b) for b in range(self.n - 1)]
        self.m_b = [eye(b) for b in range(self.n - 1)]
        self.state = _states(optimizer, self.params)

    def _entries(self, v: int, keys) -> list:
        """Core v's optimiser state tensors under ``keys`` (an entry may be None: SGD
        without momentum keeps ``momentum_buffer = None``)."""
        return [self.state[v][k] for k in keys if torch.is_tensor(self.state[v].get(k))]

    # -- the effective cores ------------------------------------------------
    def _live(self, v: int) -> torch.Tensor:
        """Core v with the boundary vectors contracted in (a leg of size 1 kept)."""
        w = self.work[v]
        if v == 0 and self.left is not None:
            w = torch.einsum("l,lpr->pr", self.left, w).unsqueeze(0)
        if v == self.n - 1 and self.right is not None:
            w = torch.einsum("lpr,r->lp", w, self.right).unsqueeze(-1)
        return w

    def log_norms2(self) -> torch.Tensor:
        return torch.stack([self._live(v).abs().square().sum().log() for v in range(self.n)])

    def log_norm(self) -> float:
        return torch.logsumexp(self.log_norms2(), 0).item()

    def _matrices(self, b: int):
        """A (core b as (left, phys) × bond) and B (core b + 1 as (phys, right) × bond)."""
        a = self._live(b)
        bb = self._live(b + 1)
        return a.reshape(-1, a.shape[-1]), bb.permute(1, 2, 0).reshape(-1, bb.shape[0])

    def _block(self, b: int):
        """The bond's live block: (U, k, P_k, Q_k) with U a unitary of the bond whose first
        k columns are the live components (A's singular directions above _LIVE, then B's
        within them), and P_k, Q_k the Gram matrices there."""
        a, bb = self._matrices(b)
        r = a.shape[1]
        _, sa, vh = torch.linalg.svd(a, full_matrices=True)
        ka = int((sa > _LIVE * sa[0]).sum()) if sa.numel() else 0
        u = vh.mH
        _, sb, wh = torch.linalg.svd((bb @ u.conj())[:, :ka], full_matrices=True)
        k = int((sb > _LIVE * sb[0]).sum()) if sb.numel() else 0
        rot = torch.eye(r, dtype=u.dtype, device=u.device)
        rot[:ka, :ka] = wh.mH.conj()       # B's block ← B's block · W, A's · W̄
        u = u @ rot
        a_k, b_k = (a @ u)[:, :k], (bb @ u.conj())[:, :k]
        return u, k, a_k.mH @ a_k, (b_k.mH @ b_k).T

    def defect(self, b: int) -> float:
        """‖P − Q‖_F / ‖P‖_F on the bond's live block (0 for a bond without one)."""
        _, k, p, q = self._block(b)
        return (torch.linalg.norm(p - q) / torch.linalg.norm(p)).item() if k else 0.0

    # -- transformations ----------------------------------------------------
    def _bond_(self, b: int, m_a: torch.Tensor, m_b: torch.Tensor) -> None:
        """A ← A M_A on core b's right leg, B ← B M_B on core b + 1's left leg (the
        copies); T = A Bᵀ unchanged is the caller's (a gauge: M_B = M_A⁻ᵀ)."""
        self.work[b] = torch.einsum("lpr,rs->lps", self.work[b], m_a)
        self.work[b + 1] = torch.einsum("lpr,ls->spr", self.work[b + 1], m_b)
        self.m_a[b], self.m_b[b] = self.m_a[b] @ m_a, self.m_b[b] @ m_b

    def _gauge_(self, b: int, s: torch.Tensor, s_inv: torch.Tensor) -> None:
        self._bond_(b, s, s_inv.mT)

    def commit_(self) -> None:
        """The copies into the cores; each core's first moments as its gradient moves
        (∇A is linear in B̄ and ∇B in Ā, T fixed): through core v's left bond
        ∇ ← ∇ ·ₗ conj(M_A of bond v − 1), through its right bond ∇ ← ∇ ·ᵣ conj(M_B of
        bond v) (a gauge: conj(S), S⁻ᴴ); second moments kept."""
        for v, p in enumerate(self.params):
            p.data.copy_(self.work[v].to(p.device, p.dtype))
            moments = [m for m in self._entries(v, _FIRST_MOMENTS)]
            if not moments:
                continue
            left = self.m_a[v - 1].conj() if v > 0 else None
            right = self.m_b[v].conj() if v < self.n - 1 else None
            for m in moments:
                t = m.to('cpu', torch.complex128)
                if left is not None:
                    t = torch.einsum("lpr,ls->spr", t, left)
                if right is not None:
                    t = torch.einsum("lpr,rs->lps", t, right)
                m.copy_(t.to(m.device, m.dtype))

    def _equalise_(self) -> None:
        """The scalar minimum on the copies, as per-bond scalars: bond b times
        exp(Σ_{u≤b} (mean − log‖W_u‖)), so every core reaches the geometric-mean norm.
        The sweeps' one-bond balances spread a norm profile along the chain only one bond
        per visit; this moves it in one pass."""
        half = self.log_norms2().real / 2
        steps = torch.cumsum(half.mean() - half, 0)[:-1]
        for b, log_s in enumerate(steps.tolist()):
            eye = torch.eye(self.m_a[b].shape[0], dtype=torch.complex128, device=self.m_a[b].device)
            self._gauge_(b, math.exp(log_s) * eye, math.exp(-log_s) * eye)

    def _drop_dead_(self, b: int) -> None:
        """B's components along A's exact null space to zero (B ← B Π̄_A, Π_A the
        projector onto A's row space in the bond), then A's along B's (A ← A Π̄_B): T is
        unchanged (A = A Π_A, B = B Π_B), the norm only falls. No-op on a bond with none."""
        for side in (0, 1):
            a, bb = self._matrices(b)
            x = a if side == 0 else bb
            _, sv, vh = torch.linalg.svd(x, full_matrices=False)
            rank = int((sv > _NULL * sv[0]).sum()) if sv.numel() and sv[0] > 0 else 0
            if rank == x.shape[1]:
                continue
            proj = (vh[:rank].mH @ vh[:rank]).conj()
            eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
            self._bond_(b, eye, proj) if side == 0 else self._bond_(b, proj, eye)

    def _exact(self, b: int) -> float:
        """The exact balance of bond b: its dead components dropped (``_drop_dead_``),
        then on the live block S = U diag(S_k, I) Uᴴ with S_k = Y^½, Y P_k Y = Q_k;
        returns the block's defect before."""
        self._drop_dead_(b)
        u, k, p, q = self._block(b)
        if k == 0:
            return 0.0
        defect = (torch.linalg.norm(p - q) / torch.linalg.norm(p)).item()
        if defect <= _TOLERANCE:
            return defect
        half, inv_half = _roots(p)
        middle, _ = _roots(half @ q @ half)
        s_k, s_k_inv = _roots(inv_half @ middle @ inv_half)
        r = u.shape[0]
        s, s_inv = (torch.eye(r, dtype=u.dtype, device=u.device) for _ in range(2))
        s[:k, :k], s_inv[:k, :k] = s_k, s_k_inv
        self._gauge_(b, u @ s @ u.mH, u @ s_inv @ u.mH)
        return defect

    def sweep_(self) -> int:
        """Gauss-Seidel sweeps (left to right, then back) of exact one-bond balances, each
        followed by the scalar equalisation (``_equalise_``), until
        every bond's defect is at most _TOLERANCE or a sweep lowers log Σ‖W‖² by less than
        _STALL (a rank-deficient bond's dead components only shrink towards the
        infimum), at most _MAX_SWEEPS; commits once; returns the sweeps."""
        order = list(range(self.n - 1)) + list(range(self.n - 2, -1, -1))
        last = self.log_norm()
        for sweep in range(1, _MAX_SWEEPS + 1):
            worst = max(self._exact(b) for b in order)
            self._equalise_()
            now = self.log_norm()
            if worst <= _TOLERANCE or last - now < _STALL:
                break
            last = now
        self.commit_()
        return sweep

    def soft_(self, rate: float) -> None:
        """One step S = exp(−τ D) on every bond, all computed from the same point (the
        bonds act on different legs, so the order of applying them does not matter)."""
        steps = []
        for b in range(self.n - 1):
            a, bb = self._matrices(b)
            p, q = a.mH @ a, (bb.mH @ bb).T
            tau = rate / (2 * (torch.linalg.matrix_norm(p, ord=2) + torch.linalg.matrix_norm(q, ord=2)))
            d = p - q
            values, vectors = torch.linalg.eigh((d + d.mH) / 2)
            down = (vectors * torch.exp(-tau * values).to(vectors.dtype)) @ vectors.mH
            up = (vectors * torch.exp(tau * values).to(vectors.dtype)) @ vectors.mH
            steps.append((down, up))
        for b, (s, s_inv) in enumerate(steps):
            self._gauge_(b, s, s_inv)
        self.commit_()


def _scalar_(cbm, optimizer) -> None:
    """Every core times c_v = exp(mean log‖W‖ − log‖W_v‖) (∏ c_v = 1: a product of
    per-bond scalars), on the cores' device (no copies: it runs after every step); first
    moments / c_v, second / c_v², both exact."""
    params = [node.tensor for node in cbm._mats_env]
    states = _states(optimizer, params)
    obc = cbm._boundary == "obc"
    norms = []
    for v, w in enumerate(params):
        w = w.detach()
        if obc and v == 0:
            w = torch.einsum("l,lpr->pr", cbm._left_node.tensor.detach().to(w.dtype), w)
        if obc and v == len(params) - 1:
            w = torch.einsum("lpr,r->lp", w, cbm._right_node.tensor.detach().to(w.dtype))
        norms.append(w.abs().square().sum())
    half = torch.stack(norms).double().log() / 2
    factors = torch.exp(half.mean() - half)
    for v, (p, c) in enumerate(zip(params, factors.tolist())):
        p.data.mul_(c)
        state = states[v]
        for keys, power in ((_FIRST_MOMENTS, 1), (_SECOND_MOMENTS, 2)):
            for k in keys:
                if torch.is_tensor(state.get(k)):
                    state[k].div_(c ** power)


def _states(optimizer, params) -> list:
    """The optimiser's state of each core ({} without an optimiser); ValueError naming any
    state tensor whose motion under a gauge change is unknown here."""
    states = [optimizer.state.get(p, {}) if optimizer is not None else {} for p in params]
    for st in states:
        unknown = [k for k, v in st.items() if torch.is_tensor(v) and v.dim() > 0
                   and k not in _FIRST_MOMENTS + _SECOND_MOMENTS]
        if unknown:
            raise ValueError(f"gauge_step_ does not know how the optimiser state "
                             f"{unknown} moves under a gauge change")
    return states


def _roots(m: torch.Tensor):
    """M^½ and M^-½ of a Hermitian positive definite M."""
    values, vectors = torch.linalg.eigh((m + m.mH) / 2)
    values = values.clamp(min=values.max().item() * 1e-30).to(vectors.dtype)
    return (vectors * values.sqrt()) @ vectors.mH, (vectors * values.rsqrt()) @ vectors.mH
