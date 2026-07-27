import math
import torch
import tensorkrowch as tk
from dataclasses import dataclass, field
from typing import List, Optional, Text
from omegaconf import OmegaConf, DictConfig
from src.utils.embeddings import embedding, range_from_embedding
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
    init_method: Text = "randn"
    std: float = 1e-9
    n_features: Optional[int] = None
    out_dim: Optional[int] = None
    dtype: Optional[str] = None


@dataclass
class CBMConfig:
    init_kwargs: MPSInitConfig = field(default_factory=MPSInitConfig)
    embedding: str = "fourier"
    model_path: Optional[str] = None
    # Opt-in overflow-safe amplitudes: route mixed_nll + class_probabilities
    # through the norm-accumulating contraction (log_amp_sq) instead of the raw
    # amplitudes() path. Off by default; enable per-run for overflow-prone
    # configs (high bond dim, alpha=1). See _log_amp_sq.
    accumulate: bool = False


class ConditionalBornMachine(tk.models.MPS):
    """
    Single MPS-based model for both discriminative and generative inference.

    Replaces BornMachine + BornClassifier + BornGenerator. The class site at
    out_features=[cls_pos] is left open; forward() returns (B, num_classes)
    amplitudes via a single parallel contraction.

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
        if not isinstance(cfg, DictConfig):
            import dataclasses
            cfg = OmegaConf.create(dataclasses.asdict(cfg))
        OmegaConf.set_struct(cfg, False)
        if getattr(cfg.init_kwargs, "n_features", None) is None:
            if data_dim is None:
                raise ValueError("Provide data_dim or set cfg.init_kwargs.n_features.")
            cfg.init_kwargs.n_features = data_dim + 1
        if getattr(cfg.init_kwargs, "out_dim", None) is None:
            if num_classes is None:
                raise ValueError("Provide num_classes or set cfg.init_kwargs.out_dim.")
            cfg.init_kwargs.out_dim = num_classes
        OmegaConf.set_struct(cfg, True)

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
            _raw = OmegaConf.to_object(cfg.init_kwargs).get("dtype")
            _dtype = _DTYPE_MAP.get(_raw, torch.float32)

        # ── Embedding ─────────────────────────────────────────────────────
        self.embedding_name = cfg.embedding
        self.embedding = embedding(self.embedding_name, _in_dim, dtype=_dtype)
        self.input_range = range_from_embedding(self.embedding_name)
        self.dtype = _dtype

        # Opt-in overflow-safe amplitude path (getattr so checkpoints whose saved
        # config predates the flag default to off). See _log_amp_sq.
        self.accumulate = bool(getattr(cfg, "accumulate", False))

        # ── cls_pos + phys_dim ────────────────────────────────────────────
        _cls_pos = getattr(cfg.init_kwargs, "out_position", None)
        if _cls_pos is None:
            _cls_pos = n_features // 2
        _phys_dim = [_in_dim] * n_features
        _phys_dim[_cls_pos] = _num_classes

        # ── MPS super().__init__ ──────────────────────────────────────────
        _init_cfg = OmegaConf.to_object(cfg.init_kwargs)
        _init_method = _init_cfg.get("init_method", "randn")
        _std = _init_cfg.get("std", 1e-9)
        _boundary = _init_cfg.get("boundary", "obc")

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
        OmegaConf.set_struct(cfg, False)
        cfg.init_kwargs.out_position = _cls_pos
        OmegaConf.set_struct(cfg, True)
        self.cfg = cfg
        self._data_dim = _data_dim
        self.in_dim = _in_dim           # physical dim per input site
        self.out_dim = _num_classes     # number of classes (phys dim at cls_pos)
        self.out_position = _cls_pos
        # self.bond_dim — inherited property from tk.models.MPS
        # self.n_features — inherited property from tk.models.MPS
        self.device = device
        # Detached log Z snapshot for purification (constant w.r.t. params); set
        # by cache_log_Z(), read by marginal_log_probability.
        self._log_Z: float | None = None
        # Per-forward with-gradient log Z + detached log|amp|² stats, populated
        # by mixed_nll each training forward. The norm regularizer and failure
        # diagnostics read these instead of contracting the norm a second time.
        # DISTINCT from _log_Z above (that one is detached/param-constant).
        self._log_Z_cache: torch.Tensor | None = None
        self._amp_diag_cache: dict | None = None
        # Per-forward accumulator for the norm-accumulating (overflow-safe)
        # contraction; reset/read inside forward(renormalize=True). None between
        # accumulate forwards. See _inline_contraction / amplitudes_accumulate.
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
        self._invalidate_log_Z_cache()

    # ======================================================================
    # Inference
    # ======================================================================

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embed raw input → (B, data_dim, phys_dim)."""
        return self.embedding(data)

    def amplitudes(self, data: torch.Tensor) -> torch.Tensor:
        """Single parallel forward pass → (B, num_classes) amplitudes ψ."""
        return super().forward(data=self.embed(data))

    # ── Norm-accumulating (overflow-safe) contraction ─────────────────────
    # Contract the amplitude while renormalizing each step to keep the running
    # node O(1), but *keep* the extracted norm in log space (unlike tk's
    # renormalize op, which discards it). ψ(x,c) = psi_renorm(x,c)·exp(log_norm),
    # so log|ψ|² = 2·log|psi_renorm| + 2·log_norm never materializes an
    # overflowing amplitude. Eager / untraced, opt-in via renormalize=True.

    def _inline_contraction(self, mats_env, renormalize=False, from_left=True):
        """Inline MPS contraction (overrides tk's static helper).

        Byte-equivalent to ``tk.models.MPS._inline_contraction`` when
        ``renormalize=False`` (the default traced path is unaffected). When
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
                        n = result_node.norm(axis=axes, keepdim=True)
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
                        n = result_node.norm(axis=axes, keepdim=True)
                        self._accumulate_log_norm(n, result_node)
                        result_node = result_node / n
            return result_node

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

    def forward(self, data=None, *args, renormalize: bool = False, **kwargs):
        """Dispatch between the default traced contraction and the eager
        norm-accumulating one.

        ``renormalize=False`` (default) → tk's traced ``MPS.forward`` unchanged
        (``data`` already embedded, matching ``amplitudes``/``trace``).
        ``renormalize=True`` → eager contraction returning
        ``(psi_renorm (B, C), log_norm (B, C))``. Runs ``contract`` directly with
        ``inline_mats=True`` so the sole renorm site is the overridden
        ``_inline_contraction``; ``reset()`` around it so this untraced path
        never pollutes (or is polluted by) the default traced ``_seq_ops``. This
        makes it eager (no trace reuse) — the accepted cost.
        """
        if not renormalize:
            return super().forward(data, *args, **kwargs)
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

    def amplitudes_accumulate(self, data: torch.Tensor):
        """Overflow-safe amplitudes → ``(psi_renorm (B, C), log_norm (B, C))``.

        ψ(x,c) = psi_renorm(x,c)·exp(log_norm(x,c)); ``psi_renorm`` is
        unit-modulus (pure phase/sign), ``log_norm`` is the real log-magnitude —
        the norm/phase split. Eager / untraced (see :meth:`forward`). Use
        :meth:`log_amp_sq` for |ψ|². Opt-in; not yet wired into training/eval.
        """
        return self(self.embed(data), renormalize=True)

    def log_amp_sq(self, data: torch.Tensor) -> torch.Tensor:
        """Overflow-safe ``log|ψ(x,c)|²`` (B, C) = 2·log|psi_renorm| + 2·log_norm.

        Drop-in replacement for ``2·log|amplitudes(data)|`` that never
        materializes an overflowing amplitude.
        """
        psi, log_norm = self.amplitudes_accumulate(data)
        log_abs = torch.log(psi.abs().clamp(min=_LOG_PROB_EPS))
        return 2.0 * log_abs + 2.0 * log_norm

    def _log_amp_sq(self, data: torch.Tensor) -> torch.Tensor:
        """log|ψ(x,c)|² (B, C) — the shared entry point for the loss and eval.

        Routes through the overflow-safe accumulate path (:meth:`log_amp_sq`)
        when ``accumulate`` is set, else the direct traced path
        ``2·log|amplitudes|``. The two are numerically equivalent where the raw
        amplitude does not overflow (see ``test_log_amp_sq_matches_amplitudes``).
        """
        if self.accumulate:
            return self.log_amp_sq(data)
        log_abs = torch.log(self.amplitudes(data).abs().clamp(min=_LOG_PROB_EPS))
        return 2.0 * log_abs

    def class_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        """Born-rule normalized class probabilities → (B, num_classes)."""
        las = self._log_amp_sq(data)
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

        The cache reflects the LAST forward only; recompute=False assumes the
        tensors are unchanged since (i.e. you are within the same step). It is
        invalidated by the in-place value mutators renormalize_() / initialize().
        Distinct from the detached _log_Z used by marginal_log_probability.
        """
        if recompute or self._log_Z_cache is None:
            self._log_Z_cache = self.log_partition_function()
        return self._log_Z_cache

    def _invalidate_log_Z_cache(self) -> None:
        """Drop the per-forward norm/amplitude caches after a tensor mutation."""
        self._log_Z_cache = None
        self._amp_diag_cache = None

    @torch.no_grad()
    def _cache_amp_diag(self, log_abs_sq: torch.Tensor) -> None:
        """Cache detached log|amp|² summary stats from the current forward.

        log_abs_sq = 2·log|ψ(x,c)| (B, C) is already computed in mixed_nll; this
        stores its finite-masked mean/min/max plus a non-finite count, matching
        the keys _format_diagnostics() expects.
        """
        finite_mask = torch.isfinite(log_abs_sq)
        finite = log_abs_sq[finite_mask]
        self._amp_diag_cache = {
            "log_amp_sq_mean": finite.mean().item() if finite.numel() else float("nan"),
            "log_amp_sq_min": finite.min().item() if finite.numel() else float("nan"),
            "log_amp_sq_max": finite.max().item() if finite.numel() else float("nan"),
            "amp_nonfinite_count": int((~finite_mask).sum().item()),
        }

    def cache_log_Z(self) -> float:
        """Compute and cache log Z as a detached float."""
        self.reset()
        with torch.no_grad():
            log_Z = self.log_partition_function()
        self._log_Z = float(log_Z.detach().cpu())
        logger.info(f"[CBM] Cached log Z = {self._log_Z:.6f}")
        return self._log_Z

    def marginal_log_probability(self, data: torch.Tensor) -> torch.Tensor:
        """
        log p(x) = log Σ_c |ψ(x,c)|² - log Z  →  (B,).

        Differentiable w.r.t. input data (for purification). _log_Z is a
        detached constant, so not differentiable w.r.t. model parameters.

        Always routes through the overflow-safe ``log_amp_sq`` contraction,
        independent of the ``accumulate`` flag: this is an analysis primitive
        (log-density for purification/UQ/MIA), so correctness beats the fast
        traced path — a raw ``amplitudes()`` here would overflow to inf and
        silently corrupt every downstream log-density on overflow-prone models.
        """
        if self._log_Z is None:
            self.cache_log_Z()
        return torch.logsumexp(self.log_amp_sq(data), dim=-1) - self._log_Z

    # ======================================================================
    # Training
    # ======================================================================

    def mixed_nll(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        alpha: float,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Mixed NLL loss interpolating between discriminative (α=0) and generative (α=1).

        L = -log|ψ(x,c)|² + (1-α)·log Σ_c |ψ(x,c)|² + α·log Z

        α=0  →  -log p(c|x)   (pure discriminative; log_partition_function not called)
        α=1  →  -log p(x,c)   (pure generative)

        debug=True: log per-term NaN/inf stats inside the grad-tracked forward.
        Non-finite log_Z is logged rather than raised so all terms are visible.
        """
        def _stats(t: torch.Tensor) -> str:
            fin = t[torch.isfinite(t)]
            m = fin.mean().item() if fin.numel() else float("nan")
            nf = int((~torch.isfinite(t)).sum().item())
            return f"mean={m:.4g} nonfinite={nf}"

        B = data.shape[0]
        las = self._log_amp_sq(data)                                      # (B, C) = log|ψ|²
        self._cache_amp_diag(las)                                         # detached, for diagnostics

        if debug:
            nf_las = int((~torch.isfinite(las)).sum().item())
            logger.warning(
                f"  [mixed_nll/grad] log|ψ|²: max={las.max().item():.4g} nonfinite={nf_las}"
            )

        term1 = -las[torch.arange(B), labels]

        if debug:
            logger.warning(f"  [mixed_nll/grad] term1(-log|ψ(x,c)|²): {_stats(term1)}")

        # Guard: skip when alpha=1 since it contributes nothing.
        if alpha < 1.0:
            term2 = (1.0 - alpha) * torch.logsumexp(las, dim=-1)
            if debug:
                logger.warning(f"  [mixed_nll/grad] term2((1-α)·log Σ|ψ|²): {_stats(term2)}")
        else:
            term2 = torch.zeros(B, device=data.device)
            if debug:
                logger.warning("  [mixed_nll/grad] term2=0 (α=1, not computed)")

        if alpha > 0.0:
            log_Z = self.log_Z(recompute=True)
            if not torch.isfinite(log_Z):
                if debug:
                    logger.warning(f"  [mixed_nll/grad] log_Z={log_Z.item():.4g} (non-finite)")
                else:
                    raise RuntimeError(
                        f"log_partition_function returned non-finite value: {log_Z.item():.4g}. "
                        "MPS has collapsed or exploded."
                    )
            elif debug:
                logger.warning(f"  [mixed_nll/grad] term3(α·log_Z): log_Z={log_Z.item():.4g}")
            term3 = alpha * log_Z
        else:
            term3 = 0.0

        return (term1 + term2 + term3).mean()

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
            alpha = math.exp((log_target - log_Z.item()) / (2 * n))
            for node in self._mats_env:
                node.tensor.data.mul_(alpha)
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

    def prepare(self, device: torch.device | None = None, train_cfg=None) -> None:
        """
        Reset state, move to device, and trace for efficient contraction.

        train_cfg is accepted for API forward-compatibility with NLLConfig
        (Phase 3) but ignored in Phase 1. auto_stack/auto_unbind are
        hardcoded and never modified via train_cfg.
        """
        self.unset_data_nodes()
        self.reset()
        if device is not None:
            self.to(device)
        self.trace(
            torch.zeros(1, self._data_dim, self.in_dim, dtype=self.dtype,
                        device=self.device)
        )

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
    def load(cls, path: str, accumulate: bool | None = None) -> "ConditionalBornMachine":
        """Restore a model from a checkpoint.

        ``accumulate`` overrides the overflow-safe amplitude flag on the loaded
        model: ``None`` (default) keeps the saved-config value — used by training
        entry points; the analysis pipeline passes ``True`` so eval/analysis
        always uses the overflow-safe path regardless of the checkpoint's flag
        (numerically identical where nothing overflows).
        """
        ckpt = torch.load(path, weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])
        inst = cls(cfg=cfg, tensors=ckpt["tensors"])
        if accumulate is not None:
            inst.accumulate = bool(accumulate)
        return inst


# ==============================================================================
# Smoke test
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

    device = torch.device("cpu")
    cfg = CBMConfig(
        embedding="legendre",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
    )
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2, device=device)

    # Tensor sharing: norm_net must reference the same Parameter objects
    for i, (a, m) in enumerate(zip(cbm.norm_net._mats_env, cbm._mats_env)):
        assert a.tensor is m.tensor, f"norm_net tensor {i} not shared"
    print("  tensor sharing OK")

    # auto_stack consistency
    assert cbm.auto_stack is False and cbm.norm_net.auto_stack is False
    print("  auto_stack=False on both main and norm_net OK")

    x = torch.linspace(-1.0, 1.0, 8).unsqueeze(1).expand(8, 2).clone()
    y = torch.randint(0, 2, (8,))

    amps = cbm.amplitudes(x)
    assert amps.shape == (8, 2), f"amplitudes shape {amps.shape}"
    print(f"  amplitudes shape={tuple(amps.shape)} OK")

    probs = cbm.class_probabilities(x)
    assert probs.shape == (8, 2)
    assert (probs >= 0).all() and (probs.sum(dim=1) - 1).abs().max() < 1e-5
    print(f"  class_probabilities shape={tuple(probs.shape)} sum_ok=True OK")

    cbm.reset()
    log_Z = cbm.log_partition_function()
    assert log_Z.isfinite(), f"log_Z non-finite: {log_Z}"
    print(f"  log_Z={log_Z.item():.4f} OK")

    cbm.reset()
    loss0 = cbm.mixed_nll(x, y, alpha=0.0)
    assert loss0.isfinite(), f"mixed_nll(alpha=0) non-finite: {loss0}"
    print(f"  mixed_nll(alpha=0)={loss0.item():.4f} OK")

    cbm.reset()
    loss1 = cbm.mixed_nll(x, y, alpha=1.0)
    assert loss1.isfinite(), f"mixed_nll(alpha=1) non-finite: {loss1}"
    print(f"  mixed_nll(alpha=1)={loss1.item():.4f} OK")

    log_px = cbm.marginal_log_probability(x)
    assert log_px.shape == (8,) and log_px.isfinite().all()
    print(f"  marginal_log_prob shape={tuple(log_px.shape)} finite=True OK")

    # norm-accumulating (overflow-safe) contraction
    psi_r, log_norm = cbm.amplitudes_accumulate(x)
    las = cbm.log_amp_sq(x)
    ref = 2.0 * torch.log(cbm.amplitudes(x).abs().clamp(min=1e-30))
    assert psi_r.shape == log_norm.shape == (8, 2)
    assert torch.allclose(las, ref, atol=1e-4), "log_amp_sq mismatch vs amplitudes()"
    assert torch.allclose(psi_r.abs(), torch.ones_like(psi_r.abs()), atol=1e-5)
    print(f"  amplitudes_accumulate |ψ|² err={ (las-ref).abs().max().item():.2e} OK")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    cbm.save(path)
    cbm2 = ConditionalBornMachine.load(path)
    probs2 = cbm2.class_probabilities(x)
    assert torch.allclose(probs, probs2, atol=1e-5), "save/load mismatch"
    ckpt = torch.load(path, weights_only=False)
    assert "regime" not in ckpt, "checkpoint should not contain 'regime'"
    os.unlink(path)
    print("  save/load roundtrip OK, no 'regime' field OK")

    # ── Sampling smoke tests ─────────────────────────────────────────────
    print()

    # condition_on_class: correct length, no mutation
    mats_before = [n.tensor.clone() for n in cbm._mats_env]
    cond = cbm.condition_on_class(0)
    assert len(cond) == cbm._data_dim, f"condition_on_class length {len(cond)}"
    assert all(t.ndim == 3 for t in cond), "not all conditioned tensors are 3D"
    for i, (snap, node) in enumerate(zip(mats_before, cbm._mats_env)):
        assert torch.allclose(snap, node.tensor), f"_mats_env[{i}] mutated by condition_on_class"
    print(f"  condition_on_class len={len(cond)} no-mutation OK")

    # _make_conditioned_net: returns left-canonical list
    ct = cbm._make_conditioned_net(0)
    assert isinstance(ct, list) and len(ct) == cbm._data_dim
    print(f"  _make_conditioned_net len={len(ct)} OK")

    # sample: shape, range, finite
    s = cbm.sample(class_idx=0, n=6, num_bins=20)
    lo, hi = cbm.input_range
    assert s.shape == (6, cbm._data_dim), f"sample shape {s.shape}"
    assert s.is_floating_point(), "sample output not float"
    assert torch.isfinite(s).all(), "sample contains non-finite values"
    assert (s >= lo).all() and (s <= hi).all(), f"sample out of range [{lo}, {hi}]"
    for i, (snap, node) in enumerate(zip(mats_before, cbm._mats_env)):
        assert torch.allclose(snap, node.tensor), f"_mats_env[{i}] mutated by sample"
    print(f"  sample shape={tuple(s.shape)} in_range=True finite=True no-mutation OK")

    # sample with chunking (n > batch_size)
    s_chunked = cbm.sample(class_idx=1, n=7, num_bins=20, batch_size=3)
    assert s_chunked.shape == (7, cbm._data_dim)
    print(f"  sample (chunked) shape={tuple(s_chunked.shape)} OK")

    # sample_all_classes: shapes and label counts
    num_classes = cbm.out_dim
    n_per = 5
    all_s, all_l = cbm.sample_all_classes(n_per_class=n_per, num_bins=20)
    assert all_s.shape == (n_per * num_classes, cbm._data_dim), f"sample_all_classes shape {all_s.shape}"
    assert all_l.shape == (n_per * num_classes,)
    for c in range(num_classes):
        assert (all_l == c).sum().item() == n_per, f"class {c} count wrong"
    print(f"  sample_all_classes shape={tuple(all_s.shape)} label_counts OK")

    # complex dtype: output is always real float
    cfg_cx = CBMConfig(embedding="fourier",
                       init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, dtype="complex64", std=1e-3))
    cbm_cx = ConditionalBornMachine(cfg=cfg_cx, data_dim=2, num_classes=2, device=device)
    s_cx = cbm_cx.sample(class_idx=0, n=4, num_bins=10)
    assert s_cx.is_floating_point() and not s_cx.is_complex(), "complex CBM sample not real float"
    print(f"  complex CBM sample dtype={s_cx.dtype} (real float) OK")

    print("\nmodel.py smoke test passed.")
