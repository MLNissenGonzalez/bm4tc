import math
import torch
import tensorkrowch as tk
from dataclasses import dataclass, field
from typing import List, Optional, Text
from omegaconf import OmegaConf
from bm4tc.core.embeddings import embedding, range_from_embedding
import logging

logger = logging.getLogger(__name__)

# Floor for log computations: only clamps actual float32 underflow (exact 0.0).
# All normal float32 amplitudes pass through with correct gradients.
_LOG_PROB_EPS: float = float(torch.finfo(torch.float32).tiny)


def draw_from_grid(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Shared multinomial leaf: draw one grid value per batch element proportional to p.

    Hard (non-differentiable) inverse-CDF sampling via torch.multinomial. Handles
    degenerate inputs (NaN, inf, all-zero rows) gracefully.

    Parameters
    ----------
    p : torch.Tensor
        Unnormalized probability weights, shape (batch, num_bins). Must be >= 0.
    z : torch.Tensor
        Grid of candidate values, shape (num_bins,).

    Returns
    -------
    torch.Tensor
        Sampled grid values, shape (batch,).
    """
    p_clean = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    row_sums = p_clean.sum(dim=-1, keepdim=True)
    p_clean = torch.where(row_sums > 0, p_clean, torch.ones_like(p_clean))
    indices = torch.multinomial(p_clean, num_samples=1).squeeze(1)
    return z[indices]


def draw_from_grid_log(log_p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Log-domain counterpart of :func:`draw_from_grid`.

    Same contract, but takes *log* weights, so callers whose weights would
    overflow in linear space (a full-chain |ψ|² over many sites) never have to
    materialize them. Only relative magnitudes matter: the per-row max is
    subtracted before exponentiating, which is exact and puts every row in
    (0, 1]. Excluded bins should be passed as ``-inf`` (they map to weight 0).

    Parameters
    ----------
    log_p : torch.Tensor
        Unnormalized log weights, shape (batch, num_bins). ``-inf`` allowed.
    z : torch.Tensor
        Grid of candidate values. Either shape (num_bins,) — one grid shared by
        every row — or (batch, num_bins), a per-row grid, so callers whose
        candidate values differ per sample (e.g. a window centred on each
        sample) can sample without a shared discretization.

    Returns
    -------
    torch.Tensor
        Sampled grid values, shape (batch,).
    """
    lp = torch.nan_to_num(log_p.float(), nan=float("-inf"), posinf=float("inf"))
    row_max = lp.amax(dim=-1, keepdim=True)
    # Rows that are entirely -inf (e.g. a fully-masked radius window) have no
    # admissible bin; fall back to uniform rather than producing NaN.
    finite_row = torch.isfinite(row_max)
    lp = torch.where(finite_row, lp - torch.where(finite_row, row_max,
                                                  torch.zeros_like(row_max)),
                     torch.zeros_like(lp))
    p = lp.exp()
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    row_sums = p.sum(dim=-1, keepdim=True)
    p = torch.where(row_sums > 0, p, torch.ones_like(p))
    indices = torch.multinomial(p, num_samples=1).squeeze(1)
    if z.ndim == 1:
        return z[indices]
    return z.gather(1, indices.unsqueeze(1)).squeeze(1)


@dataclass
class MPSInitConfig:
    in_dim: int = 4
    bond_dim: int = 3
    out_position: Optional[int] = None
    boundary: Text = "obc"
    init_method: Text = "randn_eye"
    std: float = 1e-9
    n_features: Optional[int] = None
    out_dim: Optional[int] = None
    dtype: str = "complex64"  # D4; real dtypes stay available, none is in a study


@dataclass
class CBMConfig:
    init_kwargs: MPSInitConfig = field(default_factory=MPSInitConfig)
    embedding: str = "legendre"
    model_path: Optional[str] = None


class ConditionalBornMachine(tk.models.MPS):
    """
    Single MPS-based model for both discriminative and generative inference.

    Replaces BornMachine + BornClassifier + BornGenerator. The class site at
    out_features=[cls_pos] is left open; log_amp_sq() returns log|ψ(x,c)|² (B,
    num_classes) from one norm-accumulating contraction (forward()).

    One auxiliary network shares the same Parameter objects:
      norm_net — used for log_partition_function() during training

    auto_stack=True and auto_unbind=False are hardcoded on both the main MPS
    and norm_net so they always share the same parameter view.
    """

    def __init__(
        self,
        cfg: CBMConfig,
        data_dim: int | None = None,
        num_classes: int | None = None,
        device: torch.device | None = None,
        tensors: List[torch.Tensor] | None = None,
    ):
        # ── Config normalisation ──────────────────────────────────────────
        # One typed copy of whatever came in (dataclass, plain or typed config, a
        # checkpoint's dict): every field exists, unknown keys fail (D25), and the
        # caller's config is never mutated.
        cfg = OmegaConf.merge(OmegaConf.structured(CBMConfig), cfg)
        if cfg.init_kwargs.n_features is None:
            if data_dim is None:
                raise ValueError("Provide data_dim or set cfg.init_kwargs.n_features.")
            cfg.init_kwargs.n_features = data_dim + 1
        if cfg.init_kwargs.out_dim is None:
            if num_classes is None:
                raise ValueError("Provide num_classes or set cfg.init_kwargs.out_dim.")
            cfg.init_kwargs.out_dim = num_classes

        n_features = cfg.init_kwargs.n_features
        _data_dim = n_features - 1
        _num_classes = cfg.init_kwargs.out_dim
        _in_dim = cfg.init_kwargs.in_dim
        _bond_dim = cfg.init_kwargs.bond_dim

        # ── Dtype ─────────────────────────────────────────────────────────
        _DTYPE_MAP = {
            "float32": torch.float32, "float64": torch.float64,
            "complex64": torch.complex64, "complex128": torch.complex128,
        }
        if tensors is not None:
            _dtype = tensors[0].dtype
        else:
            _dtype = _DTYPE_MAP[cfg.init_kwargs.dtype]

        # ── Embedding ─────────────────────────────────────────────────────
        self.embedding_name = cfg.embedding
        self.embedding = embedding(self.embedding_name, _in_dim, dtype=_dtype)
        self.input_range = range_from_embedding(self.embedding_name)
        self.dtype = _dtype

        # ── cls_pos + phys_dim ────────────────────────────────────────────
        _cls_pos = cfg.init_kwargs.out_position
        if _cls_pos is None:
            _cls_pos = n_features // 2
        _phys_dim = [_in_dim] * n_features
        _phys_dim[_cls_pos] = _num_classes

        # ── MPS super().__init__ ──────────────────────────────────────────
        _init_method = cfg.init_kwargs.init_method
        _std = cfg.init_kwargs.std
        _boundary = cfg.init_kwargs.boundary

        super().__init__(
            n_features=n_features,
            phys_dim=_phys_dim,
            bond_dim=_bond_dim,
            boundary=_boundary,
            out_features=[_cls_pos],
            tensors=tensors,
            init_method=_init_method if tensors is None else None,
            device=device,
            dtype=_dtype,
            std=_std,
        )

        # ── Hardcoded contraction modes ───────────────────────────────────
        self.auto_stack = False
        self._auto_unbind = False

        # ── abs_square ────────────────────────────────────────────────────
        if _dtype.is_complex:
            self.abs_square = lambda t: t.real ** 2 + t.imag ** 2
        else:
            self.abs_square = lambda t: t ** 2

        # ── norm_net ──────────────────────────────────────────────────────
        self.norm_net = self.copy(share_tensors=True)
        self.norm_net.auto_stack = False
        self.norm_net._auto_unbind = False
        # copy() creates boundary nodes as float32 on cpu regardless of the
        # model dtype/device. Re-cast them to match the cores so norm_net is
        # self-consistent right after construction. Needed for float64 (to()
        # moves device but never changes dtype, so the float32 boundary would
        # stay mismatched and break log_partition_function's boundary
        # contraction); complex was already handled, real float32 is a no-op.
        _core_device = self.tensors[0].device
        for _bn in (self.norm_net._left_node, self.norm_net._right_node):
            if _bn is not None:
                _bn.set_tensor(_bn.tensor.to(dtype=_dtype, device=_core_device))
        self._freeze_boundaries()

        # ── randn_eye phi_0 rescaling ─────────────────────────────────────
        # randn_eye sets T[:,0,:] ≈ I; initial amplitude ≈ phi_0^n_sites.
        # For non-Fourier embeddings phi_0 ≠ 1 → float32 underflow on MNIST.
        # Rescale by 1/phi_0 so (phi_0 * 1/phi_0)^n = 1. Exact for Legendre
        # (constant phi_0); partial for Hermite/Chebyshev (use 'canonical').
        if tensors is None and _init_method == "randn_eye":
            _x0 = torch.zeros(1)
            _phi_0 = float(self.embedding(_x0).view(-1)[0])
            if abs(_phi_0 - 1.0) > 1e-6:
                _scale = 1.0 / _phi_0
                _expected = _phi_0 ** n_features
                logger.warning(
                    f"[CBM] randn_eye + '{self.embedding_name}': phi_0={_phi_0:.4f} "
                    f"(expected 1.0). Amplitude ≈ {_expected:.2e} for n_sites={n_features} "
                    f"— float32 underflow risk. Rescaling tensors by 1/phi_0={_scale:.4f}."
                )
                with torch.no_grad():
                    for t in self.tensors:
                        t.data.mul_(_scale)

        # ── Sampling nodes (TK integration) ──────────────────────────────
        # _h_node: batch × bond  — running left boundary (batch of amplitude vectors)
        # _u_node: left × input × right  — embedded MPS site tensor
        #   'input' carries num_bins grid bins (replaces discrete phys_dim from reference)
        # Connected along 'bond'/'left'; tensors updated via _direct_set_tensor() per step.
        # Pattern mirrors reference/tn4dd_bm.py:113–118.
        H_init = torch.ones(1, 1, dtype=_dtype)
        U_init = torch.zeros(1, 1, 1, dtype=_dtype)
        self._h_node = tk.Node(tensor=H_init, axes_names=('batch', 'bond'))
        self._u_node = tk.Node(tensor=U_init, axes_names=('left', 'input', 'right'))
        self._h_node['bond'] ^ self._u_node['left']

        # ── Saved attributes ──────────────────────────────────────────────
        cfg.init_kwargs.out_position = _cls_pos
        self.cfg = cfg
        self._data_dim = _data_dim
        self.in_dim = _in_dim           # physical dim per input site
        self.out_dim = _num_classes     # number of classes (phys dim at cls_pos)
        self.out_position = _cls_pos
        # self.bond_dim — inherited property from tk.models.MPS
        # self.n_features — inherited property from tk.models.MPS
        self.device = device
        # Detached log Z for analysis (constant w.r.t. params), stamped with
        # _params_key() like the caches below; read through log_normalizer().
        self._log_Z: tuple | None = None            # (params key, log Z)
        # Per-forward with-gradient log Z, populated by mixed_nll each training
        # forward, stamped with _params_key() so it is never served for other
        # parameter values (no caller invalidates). Read through log_Z(), not a
        # second contraction. DISTINCT from _log_Z above (detached/param-constant).
        self._log_Z_cache: tuple | None = None      # ((params key, grad mode), log Z)
        # Per-forward accumulator for the norm-accumulating (overflow-safe)
        # contraction; reset/read inside forward(). None between forwards. See
        # _inline_contraction.
        self._log_norm_acc: torch.Tensor | None = None

    # ======================================================================
    # Auxiliary network sync
    # ======================================================================

    def _sync_norm_net(self) -> None:
        """Re-link norm_net._mats_env to self._mats_env after Parameter replacement.

        Called only from initialize(), which replaces Parameter objects. With
        auto_stack=False, forward passes never replace _mats_env[i].tensor, so
        re-linking is only needed after initialize().
        """
        self.norm_net.reset()
        for aux_node, main_node in zip(self.norm_net._mats_env, self._mats_env):
            aux_node._direct_set_tensor(main_node.tensor)

    def initialize(self, tensors=None, **kwargs):
        super().initialize(tensors=tensors, **kwargs)
        if hasattr(self, "norm_net"):
            self._sync_norm_net()
            self._freeze_boundaries()
        self._invalidate_log_Z_cache()

    def _freeze_boundaries(self) -> None:
        """Fix the obc boundary vectors, and give norm_net the model's own.

        tensorkrowch makes them ParamNodes (trainable), and ``copy(share_tensors=
        True)`` gives norm_net a separate pair, so training moved the two pairs
        apart and log Z stopped being the normaliser of ψ (D87). They carry no
        freedom of their own (a boundary vector times the edge core is another edge
        core): fixed, as tensorkrowch initialises them, and equal in both networks.
        tensorkrowch's ``initialize`` resets them, so this runs after it too."""
        if self._boundary != "obc":
            return
        for name in ("_left_node", "_right_node"):
            node, aux = getattr(self, name), getattr(self.norm_net, name)
            node.tensor.requires_grad_(False)
            aux.set_tensor(node.tensor.detach().clone())
            aux.tensor.requires_grad_(False)

    # ======================================================================
    # Inference
    # ======================================================================

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embed raw input → (B, data_dim, phys_dim)."""
        return self.embedding(data)

    # ── Norm-accumulating (overflow-safe) contraction ─────────────────────
    # Contract the amplitude while renormalizing each step to keep the running
    # node O(1), but *keep* the extracted norm in log space (unlike tk's
    # renormalize op, which discards it). ψ(x,c) = psi_renorm(x,c)·exp(log_norm),
    # so log|ψ|² = 2·log|psi_renorm| + 2·log_norm never materializes an
    # overflowing amplitude. The only amplitude contraction (D89); eager, untraced.

    def _inline_contraction(self, mats_env, renormalize=False, from_left=True):
        """Inline MPS contraction (overrides tk's static helper).

        Byte-equivalent to ``tk.models.MPS._inline_contraction`` when
        ``renormalize=False`` (tk's own ``MPS.forward``, which the tests use as the
        raw-amplitude reference). When
        ``renormalize=True`` each step's bond-axis norm is computed *once*,
        folded into ``self._log_norm_acc`` at (batch, class) granularity, and
        used to divide the running node via the ``div`` op — no second norm, no
        tk ``renormalize``.
        """
        if from_left:
            result_node = mats_env[0]
            for node in mats_env[1:]:
                result_node @= node
                if renormalize:
                    axes = [ax for ax in result_node.axes_names if 'right' in ax]
                    if axes:
                        n = self._safe_bond_norm(result_node, axes)
                        self._accumulate_log_norm(n, result_node)
                        result_node = result_node / n
            return result_node
        else:
            result_node = mats_env[-1]
            for node in mats_env[-2::-1]:
                result_node = node @ result_node
                if renormalize:
                    axes = [ax for ax in result_node.axes_names if 'left' in ax]
                    if axes:
                        n = self._safe_bond_norm(result_node, axes)
                        self._accumulate_log_norm(n, result_node)
                        result_node = result_node / n
            return result_node

    @staticmethod
    def _safe_bond_norm(result_node, axes) -> torch.Tensor:
        """Bond-axis norm, floored so a vanishing partial contraction cannot NaN.

        An embedding whose basis vector is the zero vector somewhere in its own
        domain (Chebyshev T2 at x = ±1, before its range was restricted) makes
        the running node exactly zero for that sample. The unguarded form then
        did ``0 / 0`` into ``psi`` and ``log(0)`` into ``log_norm``, so both came
        back NaN and poisoned the loss — while the raw tk contraction, which clamps
        the *final* amplitude, returned a finite floor. Flooring the divisor keeps
        ``psi`` at 0 and ``log_norm`` finite, so a zero amplitude now floors on
        this path too instead of producing NaN.

        The norm/phase split is invariant to the divisor's exact value
        (``psi/n`` scaled up by ``exp(log n)`` reconstructs the same amplitude),
        so this changes nothing wherever the norm is a normal float.
        """
        n = result_node.norm(axis=axes, keepdim=True)
        return n.clamp(min=_LOG_PROB_EPS)

    def _accumulate_log_norm(self, n: torch.Tensor, node) -> None:
        """Fold a keepdim bond-norm tensor into ``self._log_norm_acc`` at
        (batch, class) granularity.

        The batch axis is found via ``Axis.is_batch()`` and the open class axis
        via the physical edge name ``'input'``; the reduced bond axes (size 1,
        keepdim) are summed away. Pre-class steps contribute (B, 1) and broadcast
        over the class axis once the output site opens it.
        """
        log_n = n.log()
        b_idx, c_idx = None, None
        for i, ax in enumerate(node.axes):
            if ax.is_batch():
                b_idx = i
            elif 'input' in ax.name:
                c_idx = i
        order = [b_idx] + ([c_idx] if c_idx is not None else [])
        rest = [i for i in range(log_n.ndim) if i not in order]
        log_n = log_n.permute(*order, *rest)
        if c_idx is None:
            contrib = log_n.reshape(log_n.shape[0], -1).sum(dim=1, keepdim=True)   # (B, 1)
        else:
            contrib = log_n.reshape(log_n.shape[0], log_n.shape[1], -1).sum(dim=2)  # (B, C)
        self._log_norm_acc = (
            contrib if self._log_norm_acc is None else self._log_norm_acc + contrib
        )

    def forward(self, data=None, *args, **kwargs):
        """The norm-accumulating contraction of embedded ``data`` →
        ``(psi_renorm (B, C), log_norm (B, C))`` with ψ = psi_renorm·exp(log_norm):
        ``psi_renorm`` is unit-modulus (pure phase/sign), ``log_norm`` the real
        log-magnitude.

        Runs ``contract`` with ``inline_mats=True`` so the sole renorm site is the
        overridden ``_inline_contraction``, with ``reset()`` around it: every call
        builds its nodes afresh (eager, no tk trace reuse).
        """
        self.reset()
        if not self._data_nodes:
            self.set_data_nodes()
        self._log_norm_acc = None
        self.add_data(data)
        result = self.contract(renormalize=True, inline_mats=True)
        out = result.tensor                                    # (B, C), finite (running node kept O(1))
        acc = self._log_norm_acc                               # (B, 1) class-independent step factors, or None
        self.reset()
        # Final split: the class site is contracted as a single output node, so
        # the per-step renorm never factors its (class-dependent) magnitude — it
        # sits in `out` (still O(1), finite). Fold |out| into log space so
        # psi_renorm is unit-modulus (pure phase/sign) and log_norm is full (B,C).
        mag = out.abs().clamp(min=_LOG_PROB_EPS)               # (B, C)
        log_norm = mag.log() if acc is None else acc + mag.log()
        psi_renorm = out / mag
        return psi_renorm, log_norm

    def log_amp_sq(self, data: torch.Tensor) -> torch.Tensor:
        """log|ψ(x,c)|² (B, C) = 2·log|psi_renorm| + 2·log_norm: the entry point of
        the loss, evaluation and analysis. It never materialises an overflowing
        amplitude.

        Where ψ = 0 exactly (an embedding that vanishes in its own domain) the
        true value is −inf; this returns a finite floor (``_safe_bond_norm``
        contributes a floored log per vanishing step).
        """
        psi, log_norm = self(self.embed(data))
        log_abs = torch.log(psi.abs().clamp(min=_LOG_PROB_EPS))
        return 2.0 * log_abs + 2.0 * log_norm

    def class_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        """Born-rule normalized class probabilities → (B, num_classes)."""
        las = self.log_amp_sq(data)
        log_probs = las - torch.logsumexp(las, dim=-1, keepdim=True)
        return log_probs.exp()

    def log_partition_function(self) -> torch.Tensor:
        """
        log Z = log Σ_{x,c} |ψ(x,c)|² via norm_net self-contraction.

        Ported verbatim from BornGenerator.log_partition_function() with
        virtual_mps → norm_net. Supports complex tensors.
        """
        if self.norm_net._data_nodes:
            self.norm_net.unset_data_nodes()

        all_nodes = self.norm_net.mats_env[:]

        if self.norm_net._boundary == "obc":
            all_nodes[0] = self.norm_net._left_node @ all_nodes[0]
            all_nodes[-1] = all_nodes[-1] @ self.norm_net._right_node

        create_copies = []
        for node in all_nodes:
            neighbour = node.neighbours("input")
            if neighbour is None:
                create_copies.append(True)
            else:
                if "virtual_result_copy" not in neighbour.name:
                    raise ValueError(
                        f"Node {node} is already connected to another node at axis "
                        '"input". Reset the network before calling log_partition_function().'
                    )
                else:
                    create_copies.append(False)

        if any(create_copies) and not all(create_copies):
            raise ValueError(
                "Some norm_net nodes are connected and some disconnected at axis "
                '"input". Reset the network first.'
            )
        create_copies = any(create_copies)

        _is_complex = self.dtype.is_complex

        if create_copies:
            copied_nodes = []
            for node in all_nodes:
                copied_node = node.__class__(
                    shape=node._shape,
                    axes_names=node.axes_names,
                    name="virtual_result_copy",
                    network=self.norm_net,
                    virtual=True,
                )
                copied_node.set_tensor_from(node)
                copied_nodes.append(copied_node)
                for ax in copied_node.axes:
                    if ax._batch:
                        ax.name = ax.name + "_copy"

            for i in range(len(copied_nodes)):
                if (i == 0) and (self.norm_net._boundary == "pbc"):
                    if all_nodes[i - 1].is_connected_to(all_nodes[i]):
                        copied_nodes[i - 1]["right"] ^ copied_nodes[i]["left"]
                elif i > 0:
                    copied_nodes[i - 1]["right"] ^ copied_nodes[i]["left"]

            for node, copied_node in zip(all_nodes, copied_nodes):
                node.reattach_edges(axes=["input"])
                copied_node["input"] ^ node["input"]
        else:
            copied_nodes = [node.neighbours("input") for node in all_nodes]

        # Conjugate the bra on EVERY call (both create and reuse paths). conj()
        # is a fresh per-call op; doing it only in the create branch left the
        # reuse path computing Σψ² instead of Σ|ψ|² for complex models — wrong
        # log Z on every training step after the first. No-op for real dtypes.
        if _is_complex:
            copied_nodes = [node.conj() for node in copied_nodes]

        # Zip-up (ladder) contraction: carry one running environment, absorbing
        # one ket node then its bra copy per site, instead of materialising all
        # L rank-4 (D,D,D,D) transfer matrices at once. Largest transient is the
        # (D,d,D) of `result @ node`, so peak memory is O(D²·d) instead of
        # O(L·D⁴) — the difference between fitting and OOM at large bond dim.
        # Mathematically identical to the old inline-transfer-matrix order; only
        # the contraction order differs. Mirrors tn4dd TTTN.norm zip-up.
        log_Z = 0
        result_node = None
        for i, (node, copied_node) in enumerate(zip(all_nodes, copied_nodes)):
            if i == 0:
                result_node = node @ copied_node
            else:
                result_node = result_node @ node          # transient (D,d,D)
                result_node = result_node @ copied_node    # back to (D,D)
            log_Z += result_node.norm().log()
            result_node = result_node.renormalize()

        if result_node.is_connected_to(result_node):       # PBC self-loop
            result_node @= result_node
            log_Z += result_node.norm().log()
            result_node = result_node.renormalize()

        return log_Z

    def log_Z(self, recompute: bool = False) -> torch.Tensor:
        """With-gradient log Z with a per-forward cache.

        recompute=True contracts the norm via log_partition_function() and
        refreshes the cache (called by mixed_nll each forward). recompute=False
        returns the cached grad tensor from the most recent forward — the norm
        regularizer reads it this way, sharing mixed_nll's contraction graph
        instead of contracting a second time. If nothing is cached yet it
        computes (and caches) once.

        The cache is served only for the parameter values (and grad mode) it was
        computed under: after ``optimizer.step()`` recompute=False contracts again.
        Distinct from the detached _log_Z used by marginal_log_probability.
        """
        key = (self._params_key(), torch.is_grad_enabled())
        if recompute or self._log_Z_cache is None or self._log_Z_cache[0] != key:
            self._log_Z_cache = (key, self.log_partition_function())
        return self._log_Z_cache[1]

    def _params_key(self) -> tuple:
        """Identity of the current parameter values. ``optimizer.step()`` bumps each
        tensor's version counter and ``initialize()`` replaces the tensors, so a
        cache stamped with this key cannot outlive the values it came from.
        (``renormalize_`` writes through ``.data``, which bumps nothing, so it
        invalidates explicitly.)"""
        return tuple((id(p), p._version) for p in self.parameters())

    def _invalidate_log_Z_cache(self) -> None:
        """Drop the norm caches after a tensor mutation."""
        self._log_Z_cache = None
        self._log_Z = None

    def cache_log_Z(self) -> float:
        """Compute and cache log Z as a detached float."""
        self.reset()
        with torch.no_grad():
            log_Z = self.log_partition_function()
        self._log_Z = (self._params_key(), float(log_Z.detach().cpu()))
        logger.info(f"[CBM] Cached log Z = {self._log_Z[1]:.6f}")
        return self._log_Z[1]

    # ── The analysis interface (bm4tc.core.interface, D69) ──────────────────

    def log_joint(self, data: torch.Tensor) -> torch.Tensor:
        """log p(x, c) + log Z = log|ψ(x,c)|² -> (B, C)."""
        return self.log_amp_sq(data)

    def log_normalizer(self) -> float:
        """log Z, detached; computed once per parameter values."""
        if self._log_Z is None or self._log_Z[0] != self._params_key():
            self.cache_log_Z()
        return self._log_Z[1]

    def marginal_log_probability(self, data: torch.Tensor) -> torch.Tensor:
        """
        log p(x) = log Σ_c |ψ(x,c)|² - log Z  →  (B,).

        Differentiable w.r.t. input data (for purification). log Z is a
        detached constant, so not differentiable w.r.t. model parameters.

        """
        return torch.logsumexp(self.log_amp_sq(data), dim=-1) - self.log_normalizer()

    # ======================================================================
    # Training
    # ======================================================================

    def mixed_nll(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        beta: float,
        debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The objective (1-β)·L_dis + β·L_gen/N, in nats per variable (D86), with
        N = ``n_features`` (the data sites and the class site), and the batch's
        log|ψ(x,c)|² (B, C), detached, for the norm statistics. Without host
        syncs (unless ``debug``), so a training step can be captured (D90): a
        non-finite log Z makes the objective non-finite, which the caller checks.

        L = -(1-β+β/N)·log|ψ(x,c)|² + (1-β)·log Σ_c |ψ(x,c)|² + (β/N)·log Z

        β=0  →  -log p(c|x)       (discriminative; log_partition_function not called)
        β=1  →  -log p(x,c) / N   (generative)

        debug=True: log per-term NaN/inf stats inside the grad-tracked forward.
        """
        def _stats(t: torch.Tensor) -> str:
            fin = t[torch.isfinite(t)]
            m = fin.mean().item() if fin.numel() else float("nan")
            nf = int((~torch.isfinite(t)).sum().item())
            return f"mean={m:.4g} nonfinite={nf}"

        B = data.shape[0]
        las = self.log_amp_sq(data)                                      # (B, C) = log|ψ|²

        if debug:
            nf_las = int((~torch.isfinite(las)).sum().item())
            logger.warning(
                f"  [mixed_nll/grad] log|ψ|²: max={las.max().item():.4g} nonfinite={nf_las}"
            )

        n_vars = self.n_features
        term1 = -(1.0 - beta + beta / n_vars) * las[torch.arange(B, device=labels.device), labels]

        if debug:
            logger.warning(f"  [mixed_nll/grad] term1(-(1-β+β/N)·log|ψ(x,c)|²): {_stats(term1)}")

        # Guard: skip when beta=1 since it contributes nothing.
        if beta < 1.0:
            term2 = (1.0 - beta) * torch.logsumexp(las, dim=-1)
            if debug:
                logger.warning(f"  [mixed_nll/grad] term2((1-β)·log Σ|ψ|²): {_stats(term2)}")
        else:
            term2 = torch.zeros(B, device=data.device)
            if debug:
                logger.warning("  [mixed_nll/grad] term2=0 (β=1, not computed)")

        if beta > 0.0:
            log_Z = self.log_Z(recompute=True)
            if debug:
                logger.warning(f"  [mixed_nll/grad] term3(β/N·log_Z): log_Z={log_Z.item():.4g}")
            term3 = (beta / n_vars) * log_Z
        else:
            term3 = 0.0

        return (term1 + term2 + term3).mean(), las.detach()

    def renormalize_(self, log_target: float = 0.0) -> None:
        """
        Rescale all MPS core tensors in-place so log Z → log_target.

        Scales _mats_env[i].tensor.data (the actual Parameters) rather than
        self.tensors, which returns boundary-contracted views that do not share
        storage with the underlying Parameters. Safe to call after optimizer.step().
        No-op if Z is non-finite.
        """
        with torch.no_grad():
            log_Z = self.log_partition_function()
            if not torch.isfinite(log_Z):
                return
            n = len(self._mats_env)
            scale = math.exp((log_target - log_Z.item()) / (2 * n))
            for node in self._mats_env:
                node.tensor.data.mul_(scale)
        self._invalidate_log_Z_cache()

    # ======================================================================
    # Conditional sampling
    # ======================================================================

    def condition_on_class(self, class_idx: int) -> List[torch.Tensor]:
        """Return data_dim raw tensors with the class site contracted and absorbed.

        Contracts the class site with one_hot(class_idx) → (D_l, D_r), then merges
        that matrix into the right neighbor (or left neighbor if cls_pos is the last
        site). Does not mutate self._mats_env.
        """
        
        if self.out_position == 0:
            t_vec = self.tensors[0][class_idx, :] # (D_1,)
            neighbor = self.tensors[1]             # (D_1, in_dim, D_2)
            with torch.no_grad():
                cond = torch.einsum('r,rij->ij', t_vec, neighbor)  # (in_dim, D_2)
            cond_tensors = [cond] + self.tensors[2:]
        elif self.out_position == self.n_features - 1:
            t_vec = self.tensors[-1][:, class_idx] # (D_{n-2},)
            neighbor = self.tensors[-2]             # (D_{n-3}, in_dim, D_{n-2})
            with torch.no_grad():
                cond = torch.einsum('ijr,r->ij', neighbor, t_vec)  # (D_{n-3}, in_dim)
            cond_tensors = self.tensors[:-2] + [cond]
        else:
            t_mat = self.tensors[self.out_position][:, class_idx, :]  # (D_l, D_r)
            left_neighbor = self.tensors[self.out_position - 1]
            with torch.no_grad():
                if left_neighbor.ndim == 2:
                    # Left boundary (out_position==1): (phys_dim, D_l) — no left bond
                    cond = torch.einsum('il,lr->ir', left_neighbor, t_mat)  # (phys_dim, D_r)
                    cond_tensors = [cond] + self.tensors[self.out_position + 1:]
                else:
                    # Internal neighbor: (D_{l-1}, phys_dim, D_l)
                    cond = torch.einsum('ijl,lr->ijr', left_neighbor, t_mat)  # (D_{l-1}, phys_dim, D_r)
                    cond_tensors = self.tensors[:self.out_position - 1] + [cond] + self.tensors[self.out_position + 1:]
        
        return cond_tensors

    def _make_conditioned_net(
        self,
        class_idx: int,
        mode: str = 'svd',
        rank: Optional[int] = None,
        cutoff: Optional[float] = 1e-6,
    ) -> tk.models.MPS:
        """Return a left-canonical MPS conditioned on class_idx.

        Builds a fresh tk.models.MPS from the class-conditioned tensors (no
        shared tensors with self._mats_env), then calls canonicalize(oc=0) on
        it. The caller owns the returned object and should delete it when done.
        Does not mutate self._mats_env.

        Two scale controls keep this usable on long chains, and BOTH are needed:

        * Per-site pre-normalization rescales each tensor to unit Frobenius
          norm. This only multiplies the whole MPS by a global scalar, but it
          keeps the SVD inside canonicalize well-conditioned — without it the
          decomposition raises _LinAlgError once the accumulated scale is large
          (observed at n_features ≳ 200 with per-site scale 2).
        * ``renormalize=True`` stops canonicalize from concentrating the product
          of all singular-value scalings into the orthogonality center. With
          renormalize=False site 0 reaches ~1e25 (overflowing the sampler) or
          underflows to a denormal, depending on the input scale.

        Both are invisible to sampling: the first is a global scalar, the second
        distributes a constant c per node, so the right environment contracts to
        c²·I instead of I — a factor uniform across grid bins, which the per-row
        normalization in torch.multinomial divides out.

        NOTE on ``cutoff``: it is an absolute singular-value threshold, so
        pre-normalization changes its meaning — it is now relative to unit-norm
        site tensors ("drop directions 1e-6 below a unit-norm tensor") instead of
        scaling with the init magnitude. This is the intended semantics: with raw
        tensors a small-std model had singular values near the cutoff itself, so
        canonicalize could truncate a bond down to dimension 1 and silently
        discard half the state.
        """
        cond_tensors = [
            t / t.norm().clamp_min(_LOG_PROB_EPS)
            for t in self.condition_on_class(class_idx)
        ]
        cond_mps = tk.models.MPS(tensors=cond_tensors)
        cond_mps.canonicalize(oc=0, mode=mode, rank=rank, cutoff=cutoff,
                              renormalize=True)
        return cond_mps

    def sample(
        self,
        class_idx: int,
        n: int,
        num_bins: int = 100,
        batch_size: int = 64,
        mode: str = 'svd',
        rank: Optional[int] = None,
        cutoff: Optional[float] = 1e-6,
    ) -> torch.Tensor:
        """Class-conditional canonical sampling.

        Conditions on class_idx, builds a fresh left-canonical MPS via
        _make_conditioned_net, then draws n samples via the sequential
        left-to-right product rule.

        _u_node is set to the pre-embedded tensor T_k_embs = einsum('ijk,bj->ibk', T_k, Φ)
        of shape (D_l, num_bins, D_r). Grid bins act as the virtual physical dimension;
        H @ T_embs contraction is otherwise identical to the tn4dd reference.

        Returns float tensor (n, data_dim) with values in self.input_range.
        """
        cond_mps = self._make_conditioned_net(class_idx, mode=mode, rank=rank,
                                              cutoff=cutoff)
        dev = self._mats_env[0].tensor.device
        grid = torch.linspace(*self.input_range, num_bins, device=dev)
        Phi = self.embedding(grid).to(self.dtype)           # (bins, in_dim)
        left = cond_mps._left_node.tensor                   # (D_left,)
        tensors = [node.tensor for node in cond_mps._mats_env]

        chunks = []
        for start in range(0, n, batch_size):
            N = min(batch_size, n - start)
            H = left.unsqueeze(0).expand(N, -1).clone().to(dev)  # (N, D_left)
            # Per-sample renormalization: sampling is invariant under per-row
            # positive rescaling of H (multinomial normalizes each row), so we
            # keep H at O(1) every site to avoid amplitude overflow/underflow on
            # long chains (mirrors the per-node renorm in log_partition_function).
            H = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            self._h_node._direct_set_tensor(H)
            samples = torch.zeros(N, self._data_dim, device=dev)
            for k, T in enumerate(tensors):
                T_embs = torch.einsum('ijk,bj->ibk', T, Phi)   # (D_l, bins, D_r)
                self._u_node._direct_set_tensor(T_embs)
                C = (self._h_node @ self._u_node).tensor        # (N, bins, D_r)
                p = (C * C.conj()).real.sum(-1)                 # (N, bins)
                # Only relative magnitudes between bins matter (multinomial
                # normalizes each row), so divide by the per-row max: this makes
                # the epsilon below scale-free. With a bare `p + 1e-15` a row
                # whose weights all sit far below 1e-15 would be swamped by the
                # epsilon and sampled uniformly.
                p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
                p = p / p.amax(dim=-1, keepdim=True).clamp_min(_LOG_PROB_EPS)
                idx = torch.multinomial(p + 1e-15, 1).squeeze(-1)
                samples[:, k] = grid[idx]
                H_next = C[torch.arange(N, device=dev), idx, :]  # (N, D_r)
                H_next = H_next / H_next.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                self._h_node._direct_set_tensor(H_next)
            chunks.append(samples.cpu())
        del cond_mps
        return torch.cat(chunks, dim=0)

    def sample_all_classes(
        self,
        n_per_class: int,
        num_bins: int = 100,
        batch_size: int = 64,
        mode: str = 'svd',
        rank: Optional[int] = None,
        cutoff: Optional[float] = 1e-6,
    ) -> tuple:
        """Sample n_per_class examples from each class.

        Returns:
            samples: float (n_per_class * num_classes, data_dim) in input_range
            labels:  long  (n_per_class * num_classes,) class indices
        """
        all_samples, all_labels = [], []
        for c in range(self.out_dim):
            s = self.sample(c, n_per_class, num_bins=num_bins, batch_size=batch_size,
                            mode=mode, rank=rank, cutoff=cutoff)
            all_samples.append(s)
            all_labels.append(torch.full((n_per_class,), c, dtype=torch.long))
        return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)

    def prepare(self, device: torch.device | None = None) -> None:
        """Reset the contraction state and move to ``device``. No tk trace: every
        forward builds its nodes afresh (D89)."""
        self.unset_data_nodes()
        self.reset()
        if device is not None:
            self.to(device)

    # ======================================================================
    # Device / mode
    # ======================================================================

    def to(self, device):
        super().to(device)
        self.norm_net.to(device)
        self.device = device
        return self

    def eval(self):
        super().eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def reset(self):
        super().reset()

    # ======================================================================
    # Checkpoint
    # ======================================================================

    def save(self, path: str) -> None:
        torch.save(
            {"tensors": self.tensors,
             "config": OmegaConf.to_container(self.cfg, resolve=True)},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ConditionalBornMachine":
        """Restore a model from a checkpoint."""
        ckpt = torch.load(path, weights_only=False)
        return cls(cfg=OmegaConf.create(ckpt["config"]), tensors=ckpt["tensors"])

