import math
import torch
import tensorkrowch as tk
from dataclasses import dataclass, field
from typing import List, Optional, Text
from omegaconf import OmegaConf, DictConfig
import src.utils.get as get
import logging

logger = logging.getLogger(__name__)


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


class ConditionalBornMachine(tk.models.MPS):
    """
    Single MPS-based model for both discriminative and generative inference.

    Replaces BornMachine + BornClassifier + BornGenerator. The class site at
    out_features=[cls_pos] is left open; forward() returns (B, num_classes)
    amplitudes via a single parallel contraction.

    Two auxiliary networks share the same Parameter objects:
      norm_net      — used for log_partition_function() during training
      sampling_net  — deferred (canonical-form sampling, see reference/tn4dd_bm.py)

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
        self.embedding = get.embedding(self.embedding_name, _in_dim, dtype=_dtype)
        self.input_range = get.range_from_embedding(self.embedding_name)
        self.dtype = _dtype

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
        # copy() creates boundary nodes with float32 even for complex models
        if _dtype.is_complex:
            if self.norm_net._left_node is not None:
                self.norm_net._left_node.set_tensor(
                    self.norm_net._left_node.tensor.to(_dtype))
            if self.norm_net._right_node is not None:
                self.norm_net._right_node.set_tensor(
                    self.norm_net._right_node.tensor.to(_dtype))

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
        self._log_Z: float | None = None

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

    # ======================================================================
    # Inference
    # ======================================================================

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embed raw input → (B, data_dim, phys_dim)."""
        return self.embedding(data)

    def amplitudes(self, data: torch.Tensor) -> torch.Tensor:
        """Single parallel forward pass → (B, num_classes) amplitudes ψ."""
        return super().forward(data=self.embed(data))

    def class_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        """Born-rule normalized class probabilities → (B, num_classes)."""
        abs_sq = self.abs_square(self.amplitudes(data))
        denom = abs_sq.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(abs_sq.dtype).tiny)
        return abs_sq / denom

    def log_partition_function(self) -> torch.Tensor:
        """
        log Z = log Σ_{x,c} |ψ(x,c)|² via norm_net self-contraction.

        Ported verbatim from BornGenerator.log_partition_function() with
        virtual_mps → norm_net. Supports complex tensors.
        """
        # Reset norm_net to clear any stale computation-graph nodes from prior calls.
        # With auto_stack=False, _mats_env[i].tensor always returns the original
        # Parameter; no re-linking needed here.
        self.norm_net.reset()
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

            if _is_complex:
                for i, node in enumerate(copied_nodes):
                    copied_nodes[i] = node.conj()
        else:
            copied_nodes = [node.neighbours("input") for node in all_nodes]

        mats_out_env = self.norm_net._input_contraction(
            nodes_env=all_nodes,
            input_nodes=copied_nodes,
            inline_input=True,
        )

        log_Z = 0
        result_node = mats_out_env[0]
        log_Z += result_node.norm().log()
        result_node = result_node.renormalize()

        for node in mats_out_env[1:]:
            result_node @= node
            log_Z += result_node.norm().log()
            result_node = result_node.renormalize()

        if result_node.is_connected_to(result_node):
            result_node @= result_node
            log_Z += result_node.norm().log()
            result_node = result_node.renormalize()

        return log_Z

    def cache_log_Z(self) -> float:
        """Compute and cache log Z as a detached float."""
        self.reset()
        with torch.no_grad():
            log_Z = self.log_partition_function()
        self._log_Z = float(log_Z.detach().cpu())
        logger.info(f"[CBM] Cached log Z = {self._log_Z:.6f}")
        return self._log_Z

    def marginal_log_probability(self, data: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        log p(x) = log Σ_c |ψ(x,c)|² - log Z  →  (B,).

        Differentiable w.r.t. input data (for purification). _log_Z is a
        detached constant, so not differentiable w.r.t. model parameters.
        """
        if self._log_Z is None:
            self.cache_log_Z()
        abs_sq = self.abs_square(self.amplitudes(data))
        return torch.log(abs_sq.sum(dim=-1).clamp(min=eps)) - self._log_Z

    # ======================================================================
    # Training
    # ======================================================================

    def mixed_nll(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        alpha: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Mixed NLL loss interpolating between discriminative (α=0) and generative (α=1).

        L = -log|ψ(x,c)|² + (1-α)·log Σ_c |ψ(x,c)|² + α·log Z

        α=0  →  -log p(c|x)   (pure discriminative; log_partition_function not called)
        α=1  →  -log p(x,c)   (pure generative)
        """
        B = data.shape[0]
        abs_sq = self.abs_square(self.amplitudes(data))              # (B, C)
        term1 = -torch.log(abs_sq[torch.arange(B), labels].clamp(min=eps))
        term2 = (1.0 - alpha) * torch.log(abs_sq.sum(dim=-1).clamp(min=eps))
        if alpha > 0.0:
            log_Z = self.log_partition_function()
            if not torch.isfinite(log_Z):
                raise RuntimeError(
                    f"log_partition_function returned non-finite value: {log_Z.item():.4g}. "
                    "MPS has collapsed or exploded."
                )
            term3 = alpha * log_Z
        else:
            term3 = 0.0
        return (term1 + term2 + term3).mean()

    def renormalize_(self, target: float = 1.0) -> None:
        """
        Rescale all MPS core tensors in-place so Z → target.

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
            alpha = math.exp((math.log(target) - log_Z.item()) / (2 * n))
            for node in self._mats_env:
                node.tensor.data.mul_(alpha)

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
    def load(cls, path: str) -> "ConditionalBornMachine":
        ckpt = torch.load(path, weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])
        return cls(cfg=cfg, tensors=ckpt["tensors"])


# ==============================================================================
# Smoke test
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

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

    print("\ncbm.py smoke test passed.")
    