import inspect
import math
import pytest
import torch
import tensorkrowch as tk
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset
from src.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
from src.utils.train import eval_metrics


def _tiny_cbm(embedding="fourier", dtype="float32", data_dim=2, num_classes=2,
              bond_dim=2, std=1e-3):
    cfg = CBMConfig(
        embedding=embedding,
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=bond_dim, dtype=dtype, std=std),
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=data_dim, num_classes=num_classes)


def test_cbmconfig_defaults():
    cfg = CBMConfig()
    assert cfg.embedding == "fourier"
    assert cfg.model_path is None
    assert cfg.init_kwargs.in_dim == 4
    assert cfg.init_kwargs.bond_dim == 3


def test_abs_square_real():
    cbm = _tiny_cbm(dtype="float32")
    t = torch.tensor([2.0, -3.0])
    result = cbm.abs_square(t)
    expected = t ** 2
    assert torch.allclose(result, expected)


def test_abs_square_complex():
    cbm = _tiny_cbm(dtype="complex64")
    t = torch.tensor([1.0 + 2.0j, -1.0 + 0.5j])
    result = cbm.abs_square(t)
    expected = t.real ** 2 + t.imag ** 2
    assert torch.allclose(result, expected)


def test_mixed_nll_alpha0_skips_logZ():
    cbm = _tiny_cbm()
    x = torch.rand(4, 2)
    y = torch.randint(0, 2, (4,))
    with patch.object(cbm, "log_partition_function", wraps=cbm.log_partition_function) as mock_logZ:
        cbm.mixed_nll(x, y, alpha=0.0)
    mock_logZ.assert_not_called()


def test_mixed_nll_scalar_output():
    cbm = _tiny_cbm()
    x = torch.rand(4, 2)
    y = torch.randint(0, 2, (4,))
    loss = cbm.mixed_nll(x, y, alpha=0.0)
    assert loss.ndim == 0
    assert loss.isfinite()


# ── log_Z cache (with-gradient, per-forward) ───────────────────────────────

def test_log_Z_cache_reuses_within_forward():
    """recompute=True contracts once; recompute=False reuses the cached tensor."""
    cbm = _tiny_cbm()
    with patch.object(cbm, "log_partition_function",
                      wraps=cbm.log_partition_function) as mock_logZ:
        a = cbm.log_Z(recompute=True)
        b = cbm.log_Z(recompute=False)
        c = cbm.log_Z(recompute=False)
    assert mock_logZ.call_count == 1
    assert b is a and c is a


def test_log_Z_recompute_contracts_again():
    cbm = _tiny_cbm()
    with patch.object(cbm, "log_partition_function",
                      wraps=cbm.log_partition_function) as mock_logZ:
        cbm.log_Z(recompute=True)
        cbm.log_Z(recompute=True)
    assert mock_logZ.call_count == 2


def test_log_Z_recompute_false_computes_when_empty():
    """recompute=False with an empty cache still computes (and caches) once."""
    cbm = _tiny_cbm()
    assert cbm._log_Z_cache is None
    val = cbm.log_Z(recompute=False)
    assert torch.isfinite(val.real) and cbm._log_Z_cache is val


def test_renormalize_invalidates_log_Z_cache():
    cbm = _tiny_cbm()
    cbm.log_Z(recompute=True)
    assert cbm._log_Z_cache is not None
    cbm.renormalize_(log_target=0.0)
    assert cbm._log_Z_cache is None and cbm._amp_diag_cache is None


def test_initialize_invalidates_log_Z_cache():
    cbm = _tiny_cbm()
    cbm.log_Z(recompute=True)
    cbm.initialize()
    assert cbm._log_Z_cache is None and cbm._amp_diag_cache is None


# ── amp-diag cache populated by mixed_nll ──────────────────────────────────

_AMP_KEYS = {"log_amp_sq_mean", "log_amp_sq_min", "log_amp_sq_max", "amp_nonfinite_count"}


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_mixed_nll_populates_amp_diag_cache(alpha):
    cbm = _tiny_cbm()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    cbm.mixed_nll(x, y, alpha=alpha)
    assert cbm._amp_diag_cache is not None
    assert set(cbm._amp_diag_cache) == _AMP_KEYS
    assert math.isfinite(cbm._amp_diag_cache["log_amp_sq_mean"])


def test_mixed_nll_logZ_cache_only_when_alpha_positive():
    cbm = _tiny_cbm()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    cbm.mixed_nll(x, y, alpha=0.0)
    assert cbm._log_Z_cache is None          # alpha=0 never contracts the norm
    cbm.mixed_nll(x, y, alpha=1.0)
    assert cbm._log_Z_cache is not None


def test_regularizer_reuses_mixed_nll_contraction():
    """alpha>0: mixed_nll + NormRegularizer contract the norm once, total."""
    from src.utils.train import NormRegularizer
    cbm = _tiny_cbm()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    reg = NormRegularizer(strength=1.0, log_target=0.0)
    with patch.object(cbm, "log_partition_function",
                      wraps=cbm.log_partition_function) as mock_logZ:
        loss = cbm.mixed_nll(x, y, alpha=1.0) + reg(cbm)
    assert mock_logZ.call_count == 1
    # Shared log_Z node receives gradient from both terms.
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in cbm.parameters())


def test_norm_net_auto_stack_matches():
    cbm = _tiny_cbm()
    assert cbm.auto_stack is False
    assert cbm.norm_net.auto_stack is False
    assert cbm._auto_unbind is False
    assert cbm.norm_net._auto_unbind is False


# ── condition_on_class ──────────────────────────────────────────────────────

def test_condition_on_class_length():
    cbm = _tiny_cbm(data_dim=4, num_classes=3)
    assert len(cbm.condition_on_class(0)) == 4


def test_condition_on_class_tensor_ranks():
    cbm = _tiny_cbm(data_dim=4, num_classes=3)
    tensors = cbm.condition_on_class(0)
    assert tensors[0].ndim == 2   # left boundary
    assert tensors[-1].ndim == 2  # right boundary
    for t in tensors[1:-1]:
        assert t.ndim == 3


def test_condition_on_class_no_mutation():
    cbm = _tiny_cbm(data_dim=4, num_classes=3)
    before = [n.tensor.clone() for n in cbm._mats_env]
    cbm.condition_on_class(0)
    cbm.condition_on_class(1)
    for snap, node in zip(before, cbm._mats_env):
        assert torch.allclose(snap, node.tensor)


def test_condition_on_class_last_position():
    # out_position at the last site uses the left-merge path
    cbm = _tiny_cbm(data_dim=2, num_classes=2)
    # out_position defaults to n_features // 2 = 1; test with forced last position
    from omegaconf import OmegaConf
    cfg = OmegaConf.structured(
        {"embedding": "fourier", "model_path": None,
         "init_kwargs": {"in_dim": 2, "bond_dim": 2, "out_position": 2,
                         "boundary": "obc", "init_method": "randn",
                         "dtype": "float32", "n_features": None,
                         "out_dim": None, "std": 1e-3}}
    )
    from src.model import CBMConfig
    cfg_dc = CBMConfig(embedding="fourier",
                       init_kwargs={"in_dim": 2, "bond_dim": 2, "out_position": 2,
                                    "std": 1e-3, "dtype": "float32"})
    cbm2 = ConditionalBornMachine(cfg=cfg_dc, data_dim=2, num_classes=2)
    tensors = cbm2.condition_on_class(0)
    assert len(tensors) == 2
    # both tensors are boundary nodes (2D) when out_position is the last site
    for t in tensors:
        assert t.ndim == 2


# ── _make_conditioned_net ────────────────────────────────────────────────────

def test_make_conditioned_net_returns_mps():
    cbm = _tiny_cbm()
    result = cbm._make_conditioned_net(0)
    assert isinstance(result, tk.models.MPS)
    assert result.n_features == cbm._data_dim


def test_make_conditioned_net_is_canonical():
    cbm = _tiny_cbm()
    cond_mps = cbm._make_conditioned_net(0)
    # Sites 1..n-1 must be left-isometric: A†A ≈ I (summed over left and phys dims)
    for k in range(1, cond_mps.n_features):
        A = cond_mps._mats_env[k].tensor   # (D_l, d, D_r)
        D_l, d, D_r = A.shape
        ATA = torch.einsum('ijk,ijl->kl', A.conj(), A)  # (D_r, D_r)
        assert torch.allclose(ATA, torch.eye(D_r, dtype=ATA.dtype), atol=1e-5), \
            f"Site {k} not left-isometric: max err {(ATA - torch.eye(D_r, dtype=ATA.dtype)).abs().max():.2e}"


def test_make_conditioned_net_no_mutation():
    cbm = _tiny_cbm()
    before = [n.tensor.clone() for n in cbm._mats_env]
    cbm._make_conditioned_net(0)
    for snap, node in zip(before, cbm._mats_env):
        assert torch.allclose(snap, node.tensor)


# ── sample ──────────────────────────────────────────────────────────────────

def test_sample_shape():
    cbm = _tiny_cbm()
    assert cbm.sample(0, n=5, num_bins=10).shape == (5, cbm._data_dim)


def test_sample_in_range():
    cbm = _tiny_cbm()
    lo, hi = cbm.input_range
    for c in range(2):
        s = cbm.sample(c, n=8, num_bins=10)
        assert (s >= lo).all() and (s <= hi).all()


def test_sample_finite():
    cbm = _tiny_cbm()
    assert torch.isfinite(cbm.sample(0, n=8, num_bins=10)).all()


def test_sample_chunks_match_shape():
    # n > batch_size exercises the chunked path
    cbm = _tiny_cbm()
    s = cbm.sample(0, n=7, num_bins=10, batch_size=3)
    assert s.shape == (7, cbm._data_dim)


def test_sample_no_mutation():
    cbm = _tiny_cbm()
    before = [n.tensor.clone() for n in cbm._mats_env]
    cbm.sample(0, n=4, num_bins=10)
    for snap, node in zip(before, cbm._mats_env):
        assert torch.allclose(snap, node.tensor)


def test_sample_complex_dtype():
    cbm = _tiny_cbm(dtype="complex64")
    s = cbm.sample(0, n=4, num_bins=10)
    assert s.is_floating_point()  # output always real


# ── sampling numerical stability (per-site H renormalization) ─────────────────

def _old_sample_loop(cbm, class_idx, n, num_bins):
    """Pre-fix sampling loop: byte-for-byte ConditionalBornMachine.sample()
    EXCEPT the two H-renormalization lines are removed. Uses the same node
    contraction path so the only difference under test is the renormalization."""
    cond = cbm._make_conditioned_net(class_idx)
    dev = cbm._mats_env[0].tensor.device
    grid = torch.linspace(*cbm.input_range, num_bins, device=dev)
    Phi = cbm.embedding(grid).to(cbm.dtype)
    left = cond._left_node.tensor
    tensors = [node.tensor for node in cond._mats_env]
    H = left.unsqueeze(0).expand(n, -1).clone().to(dev)
    cbm._h_node._direct_set_tensor(H)
    samples = torch.zeros(n, cbm._data_dim, device=dev)
    for k, T in enumerate(tensors):
        T_embs = torch.einsum('ijk,bj->ibk', T, Phi)
        cbm._u_node._direct_set_tensor(T_embs)
        C = (cbm._h_node @ cbm._u_node).tensor
        p = (C * C.conj()).real.sum(-1).clamp(min=0)
        idx = torch.multinomial(p + 1e-15, 1).squeeze(-1)
        samples[:, k] = grid[idx]
        cbm._h_node._direct_set_tensor(C[torch.arange(n, device=dev), idx, :])
    return samples.cpu()


def _argmax_multinomial(weights, num_samples, *args, **kwargs):
    """Deterministic stand-in for torch.multinomial(·, 1): argmax is exactly
    scale-invariant, so it exposes any change in sampling decisions."""
    return weights.argmax(dim=-1, keepdim=True)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_sample_renorm_matches_old_method(dtype):
    # On a short chain (no overflow) the renormalized sampler must reproduce the
    # old method bit-for-bit: per-row positive rescaling of H leaves the draw
    # unchanged. Patch multinomial -> argmax to remove RNG. Seed the random init
    # too: an unseeded init can land on a near-tie between bins where the renorm's
    # float noise flips argmax, making the comparison order-dependently flaky.
    torch.manual_seed(0)
    cbm = _tiny_cbm(dtype=dtype, data_dim=4, bond_dim=3, std=0.5)
    with patch("torch.multinomial", _argmax_multinomial):
        s_new = cbm.sample(0, n=16, num_bins=12)
        s_old = _old_sample_loop(cbm, 0, 16, 12)
    assert torch.equal(s_new, s_old)


def test_sample_keeps_boundary_normalized_long_chain():
    # MNIST-scale regime: a long chain. The fix keeps the running boundary H at
    # unit norm every site, so output stays finite and non-degenerate.
    cbm = _tiny_cbm(dtype="complex64", data_dim=250)
    cbm.eval()
    cbm.renormalize_(log_target=0.0)
    with patch.object(cbm._h_node, "_direct_set_tensor",
                      wraps=cbm._h_node._direct_set_tensor) as spy:
        s = cbm.sample(0, n=8, num_bins=8)
    assert torch.isfinite(s).all()
    assert s.unique().numel() > 1  # not collapsed to a single value
    # Every boundary set into the contraction node is per-sample unit-normalized.
    assert spy.call_args_list, "H was never set"
    for call in spy.call_args_list:
        norms = call.args[0].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ── sample_all_classes ──────────────────────────────────────────────────────

def test_sample_all_classes_shapes():
    cbm = _tiny_cbm(num_classes=3)
    samples, labels = cbm.sample_all_classes(n_per_class=5, num_bins=10)
    assert samples.shape == (15, cbm._data_dim)
    assert labels.shape == (15,)


def test_sample_all_classes_label_counts():
    cbm = _tiny_cbm(num_classes=3)
    _, labels = cbm.sample_all_classes(n_per_class=4, num_bins=10)
    for c in range(3):
        assert (labels == c).sum().item() == 4


# ── _LOG_PROB_EPS / gradient-at-small-amplitude tests ──────────────────────

def test_mixed_nll_no_eps_param():
    sig = inspect.signature(ConditionalBornMachine.mixed_nll)
    assert "eps" not in sig.parameters


def test_marginal_log_prob_no_eps_param():
    sig = inspect.signature(ConditionalBornMachine.marginal_log_probability)
    assert "eps" not in sig.parameters


def test_large_amplitude_numerical_stability():
    """logsumexp path stays finite when |ψ| > sqrt(float32_max) ≈ 1.84e19 (amp² overflows)."""
    cbm = _tiny_cbm()
    x = torch.rand(4, 2)
    y = torch.randint(0, 2, (4,))

    with torch.no_grad():
        amp0 = cbm.amplitudes(x)
        max_abs = float(amp0.abs().max().clamp(min=1e-30))
        c = (2e19 / max_abs) ** (1.0 / cbm.n_features)
        for node in cbm._mats_env:
            node.tensor.data.mul_(c)

    cbm.reset()
    with torch.no_grad():
        amp = cbm.amplitudes(x)
    assert amp.abs().max().item() > 1.84e19, "setup: amplitudes not large enough"
    assert amp.isfinite().all(), "setup: amplitudes must be finite (not yet overflowed)"

    cbm.reset()
    probs = cbm.class_probabilities(x)
    assert probs.isfinite().all() and (probs >= 0).all()

    cbm.cache_log_Z()
    cbm.reset()
    log_px = cbm.marginal_log_probability(x)
    assert log_px.isfinite().all()

    cbm.reset()
    loss = cbm.mixed_nll(x, y, alpha=0.5)
    assert loss.isfinite()


def test_mixed_nll_term1_gradient_at_small_amplitude():
    """With the old eps=1e-12 floor on abs_sq, amplitudes ~1e-7 (abs_sq~1e-14)
    hit the clamp and produced zero/wrong gradients.  The 2·log(abs) form with
    tiny floor should give finite, non-zero gradients at this scale.
    """
    # std=1e-5 → typical |ψ(x,c)| ≈ 1e-7, abs_sq ≈ 1e-14  (below old floor)
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, dtype="float32", std=1e-5),
    )
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2)

    x = torch.rand(4, 2, requires_grad=False)
    y = torch.zeros(4, dtype=torch.long)

    loss = cbm.mixed_nll(x, y, alpha=0.0)
    assert loss.isfinite(), "loss must be finite for small-amplitude CBM"
    loss.backward()

    grads = [p.grad for p in cbm.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients computed"
    assert all(g.isfinite().all() for g in grads), "gradients contain non-finite values"
    assert any(g.abs().max() > 0 for g in grads), "all gradients are zero (clamp floor too high)"


# ── Norm-accumulating (overflow-safe) contraction ────────────────────────────
# amplitudes_accumulate / log_amp_sq contract while keeping the running node O(1)
# and accumulate the extracted norm in log space, so log|ψ|² is recoverable
# without ever materializing an overflowing amplitude. Verified against the
# default (non-accumulating) amplitudes() path.

def _acc_cbm(dtype="float32", data_dim=3, num_classes=2, bond_dim=3,
             std=0.3, out_position=None, accumulate=False):
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=bond_dim, dtype=dtype,
                                  std=std, out_position=out_position),
        accumulate=accumulate,
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=data_dim, num_classes=num_classes)


def _overflow_scale_(cbm, scale=1e10):
    """Blow up the amplitude past float32 (no single contraction step overflows,
    so the accumulate/log_Z paths stay finite; the raw product does not)."""
    with torch.no_grad():
        for node in cbm._mats_env:
            node.tensor.data.mul_(scale)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_log_amp_sq_matches_amplitudes(dtype):
    """log_amp_sq(x) reproduces 2·log|amplitudes(x)| where the latter doesn't
    overflow (real + complex)."""
    torch.manual_seed(0)
    cbm = _acc_cbm(dtype=dtype)
    x = torch.rand(5, 3) * 1.6 - 0.8
    ref = 2.0 * torch.log(cbm.amplitudes(x).abs().clamp(min=1e-30))
    got = cbm.log_amp_sq(x)
    assert got.shape == ref.shape == (5, cbm.out_dim)
    assert torch.allclose(got, ref, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_amplitude_reconstruction(dtype):
    """psi_renorm · exp(log_norm) reconstructs the true amplitude magnitude."""
    torch.manual_seed(1)
    cbm = _acc_cbm(dtype=dtype)
    x = torch.rand(4, 3) * 1.6 - 0.8
    psi, log_norm = cbm.amplitudes_accumulate(x)
    assert psi.shape == log_norm.shape == (4, cbm.out_dim)
    assert not torch.is_complex(log_norm)
    assert torch.allclose(psi.abs(), torch.ones_like(psi.abs()), atol=1e-5)  # unit-modulus
    recon = psi.abs() * torch.exp(log_norm)
    assert torch.allclose(recon, cbm.amplitudes(x).abs(), atol=1e-5)


def test_class_probabilities_from_log_amp_sq():
    """class_probabilities rebuilt from log_amp_sq matches the native method —
    guards the per-(batch, class) accumulation (a per-batch collapse fails here)."""
    torch.manual_seed(2)
    cbm = _acc_cbm(num_classes=3)
    x = torch.rand(6, 3) * 1.6 - 0.8
    las = cbm.log_amp_sq(x)
    log_probs = las - torch.logsumexp(las, dim=-1, keepdim=True)
    assert torch.allclose(log_probs.exp(), cbm.class_probabilities(x), atol=1e-5)


def test_mixed_nll_alpha0_from_log_amp_sq():
    """mixed_nll(alpha=0) = mean[-las[c] + logsumexp(las)] reconstructed from
    log_amp_sq (term1 + term2)."""
    torch.manual_seed(3)
    cbm = _acc_cbm()
    x = torch.rand(5, 3) * 1.6 - 0.8
    y = torch.randint(0, cbm.out_dim, (5,))
    las = cbm.log_amp_sq(x)
    term1 = -las[torch.arange(5), y]
    term2 = torch.logsumexp(las, dim=-1)
    recon = (term1 + term2).mean()
    assert torch.allclose(recon, cbm.mixed_nll(x, y, alpha=0.0), atol=1e-4)


def test_log_amp_sq_overflow_safe():
    """When the raw amplitude overflows to inf, log_amp_sq stays finite and
    equals the pre-scale value plus the analytic shift 2·n_sites·log(scale)."""
    torch.manual_seed(4)
    cbm = _acc_cbm(data_dim=4)
    x = torch.rand(3, 4) * 1.6 - 0.8
    las_base = cbm.log_amp_sq(x).detach().clone()
    # Per-site scale whose product over the chain overflows the amplitude, while
    # no single contraction step does (that would overflow the norm itself).
    scale = 1e10
    with torch.no_grad():
        for node in cbm._mats_env:
            node.tensor.data.mul_(scale)
    amp = cbm.amplitudes(x)
    las = cbm.log_amp_sq(x)
    assert (~torch.isfinite(amp)).any(), "test scale did not overflow the amplitude"
    assert torch.isfinite(las).all()
    shift = 2.0 * cbm.n_features * math.log(scale)
    assert torch.allclose(las, las_base + shift, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("out_position", [0, 2, 4])
def test_log_amp_sq_out_position(out_position):
    """Correct across class site at first / middle / last position (obc
    boundaries + accumulator alignment across regions)."""
    torch.manual_seed(5)
    cbm = _acc_cbm(data_dim=4, num_classes=3, out_position=out_position)
    x = torch.rand(4, 4) * 1.6 - 0.8
    ref = 2.0 * torch.log(cbm.amplitudes(x).abs().clamp(min=1e-30))
    assert torch.allclose(cbm.log_amp_sq(x), ref, atol=1e-4)


def test_accumulate_gradients_match():
    """Gradients of log_amp_sq match the direct 2·log|amplitudes| path on the
    parameters that participate in the amplitude forward."""
    torch.manual_seed(6)
    cbm = _acc_cbm()
    x = torch.rand(4, 3) * 1.6 - 0.8

    cbm.zero_grad()
    cbm.log_amp_sq(x).sum().backward()
    g_acc = [None if p.grad is None else p.grad.clone() for p in cbm.parameters()]

    cbm.zero_grad()
    amp = cbm.amplitudes(x)
    (2.0 * torch.log(amp.abs().clamp(min=1e-30))).sum().backward()
    g_ref = [None if p.grad is None else p.grad.clone() for p in cbm.parameters()]

    assert [a is None for a in g_acc] == [b is None for b in g_ref]
    for a, b in zip(g_acc, g_ref):
        if a is not None:
            assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_amplitudes_accumulate_forward_backward(dtype):
    """The accumulate path is differentiable end-to-end: forward returns
    grad-tracking outputs and backward through *both* psi_renorm and log_norm
    reaches the parameters with finite, non-zero gradients (the reset() around
    the eager contraction must not sever the autograd graph)."""
    torch.manual_seed(8)
    cbm = _acc_cbm(dtype=dtype)
    x = torch.rand(4, 3) * 1.6 - 0.8

    psi, log_norm = cbm.amplitudes_accumulate(x)          # forward
    assert psi.requires_grad and log_norm.requires_grad
    assert psi.grad_fn is not None and log_norm.grad_fn is not None

    cbm.zero_grad()
    (psi.abs().sum() + log_norm.sum()).backward()          # backward through both
    grads = [p.grad for p in cbm.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max() > 0 for g in grads)


def test_accumulate_mode_isolation():
    """A renormalize=True call resets around itself, leaving the default traced
    amplitudes() path bit-identical before and after."""
    torch.manual_seed(7)
    cbm = _acc_cbm()
    x = torch.rand(4, 3) * 1.6 - 0.8
    before = cbm.amplitudes(x)
    _ = cbm.amplitudes_accumulate(x)
    after = cbm.amplitudes(x)
    assert torch.equal(before, after)


# ── accumulate flag: opt-in routing of mixed_nll + eval ──────────────────────
# _log_amp_sq dispatches on the flag; off = direct 2·log|amplitudes|, on = the
# norm-accumulating log_amp_sq. Default off, so training/eval are unchanged
# unless a run opts in (born.accumulate=true).

def test_cbmconfig_accumulate_default_false():
    assert CBMConfig().accumulate is False
    assert _acc_cbm().accumulate is False
    assert _acc_cbm(accumulate=True).accumulate is True


def test_accumulate_flag_off_matches_direct():
    """Flag off: mixed_nll and class_probabilities equal the direct (raw
    amplitudes) computation — regression guard on the default path."""
    torch.manual_seed(10)
    cbm = _acc_cbm()  # flag off
    x = torch.rand(5, 3) * 1.6 - 0.8
    y = torch.randint(0, cbm.out_dim, (5,))

    log_abs = torch.log(cbm.amplitudes(x).abs().clamp(min=1e-30))
    ref_probs = (2 * log_abs
                 - torch.logsumexp(2 * log_abs, dim=-1, keepdim=True)).exp()
    assert torch.allclose(cbm.class_probabilities(x), ref_probs, atol=1e-6)

    term1 = -2 * log_abs[torch.arange(5), y]
    term2 = torch.logsumexp(2 * log_abs, dim=-1)
    assert torch.allclose(cbm.mixed_nll(x, y, alpha=0.0),
                          (term1 + term2).mean(), atol=1e-5)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_accumulate_flag_mixed_nll_finite_on_overflow(alpha):
    """Flag on: mixed_nll stays finite when the raw amplitude overflows (both
    the discriminative and generative ends); flag off it is non-finite."""
    torch.manual_seed(11)
    cbm_on = _acc_cbm(data_dim=4, accumulate=True)
    cbm_off = _acc_cbm(data_dim=4, accumulate=False)
    x = torch.rand(3, 4) * 1.6 - 0.8
    y = torch.randint(0, cbm_on.out_dim, (3,))
    # 1e8 overflows the raw amplitude (product over 5 sites > float32 max) while
    # keeping log_Z finite, so the α=1 term3 does not raise on the off path.
    _overflow_scale_(cbm_on, scale=1e8)
    _overflow_scale_(cbm_off, scale=1e8)

    assert (~torch.isfinite(cbm_off.amplitudes(x))).any(), "scale did not overflow"
    assert torch.isfinite(cbm_on.mixed_nll(x, y, alpha=alpha))
    assert not torch.isfinite(cbm_off.mixed_nll(x, y, alpha=alpha))


def test_accumulate_flag_class_probabilities_finite_on_overflow():
    """Flag on: class_probabilities stays finite and normalized when the raw
    amplitude overflows (covers eval / all class_probabilities consumers)."""
    torch.manual_seed(12)
    cbm = _acc_cbm(data_dim=4, num_classes=3, accumulate=True)
    x = torch.rand(4, 4) * 1.6 - 0.8
    _overflow_scale_(cbm)
    probs = cbm.class_probabilities(x)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_eval_metrics_accumulate_parity():
    """eval_metrics returns the same (dis_loss, acc, gen_loss) with the flag on
    or off on a non-overflowing model — the accumulate path only changes the
    contraction, not the result. Regression guard that existing (flag-off) valid
    numbers are unchanged by routing eval through _log_amp_sq."""
    torch.manual_seed(13)
    cbm = _acc_cbm(data_dim=4, num_classes=3)
    ds = TensorDataset(torch.rand(12, 4) * 1.6 - 0.8, torch.randint(0, 3, (12,)))
    loader = DataLoader(ds, batch_size=5)

    cbm.accumulate = False
    dis_off, acc_off, gen_off = eval_metrics(cbm, loader, "cpu")
    cbm.accumulate = True
    dis_on, acc_on, gen_on = eval_metrics(cbm, loader, "cpu")

    assert acc_off == acc_on
    assert dis_on == pytest.approx(dis_off, abs=1e-4)
    assert gen_on == pytest.approx(gen_off, abs=1e-4)


def test_eval_metrics_accumulate_finite_on_overflow():
    """eval_metrics valid losses stay finite with accumulate on when the raw
    amplitude overflows; with it off they are nan — the MNIST-resize symptom
    (stable training, nan valid) that motivated routing eval through the same
    _log_amp_sq path as the loss."""
    torch.manual_seed(14)
    cbm = _acc_cbm(data_dim=4, num_classes=3)
    _overflow_scale_(cbm, scale=1e8)  # overflows the raw amplitude, keeps log_Z finite
    x = torch.rand(9, 4) * 1.6 - 0.8
    assert (~torch.isfinite(cbm.amplitudes(x))).any(), "scale did not overflow"
    ds = TensorDataset(x, torch.randint(0, 3, (9,)))
    loader = DataLoader(ds, batch_size=4)

    cbm.accumulate = False
    dis_off, _, gen_off = eval_metrics(cbm, loader, "cpu")
    assert math.isnan(dis_off) and math.isnan(gen_off)

    cbm.accumulate = True
    dis_on, _, gen_on = eval_metrics(cbm, loader, "cpu")
    assert math.isfinite(dis_on) and math.isfinite(gen_on)


def test_marginal_log_probability_safe_parity():
    """marginal_log_probability (always overflow-safe) equals the raw
    2·log|amplitudes| formula where the amplitude does not overflow — including
    the input gradient purification relies on."""
    torch.manual_seed(15)
    cbm = _acc_cbm(data_dim=4, num_classes=3)
    cbm.cache_log_Z()

    x = (torch.rand(5, 4) * 1.6 - 0.8).requires_grad_(True)
    lp = cbm.marginal_log_probability(x)
    lp.sum().backward()
    g_safe = x.grad.clone()

    x_raw = x.detach().clone().requires_grad_(True)
    log_abs = torch.log(cbm.amplitudes(x_raw).abs().clamp(min=1e-30))
    lp_raw = torch.logsumexp(2.0 * log_abs, dim=-1) - cbm._log_Z
    lp_raw.sum().backward()

    assert torch.allclose(lp.detach(), lp_raw.detach(), atol=1e-4)
    assert torch.isfinite(g_safe).all() and g_safe.abs().max() > 0
    assert torch.allclose(g_safe, x_raw.grad, atol=1e-4)


def test_marginal_log_probability_finite_on_overflow():
    """marginal_log_probability stays finite when the raw amplitude overflows —
    the log-density primitive behind purification/UQ/MIA, so it must not go inf
    regardless of the accumulate flag."""
    torch.manual_seed(16)
    cbm = _acc_cbm(data_dim=4, num_classes=3)  # accumulate flag off — must still be safe
    _overflow_scale_(cbm, scale=1e8)
    cbm.cache_log_Z()
    x = torch.rand(4, 4) * 1.6 - 0.8

    assert (~torch.isfinite(cbm.amplitudes(x))).any(), "scale did not overflow"
    assert torch.isfinite(cbm.marginal_log_probability(x)).all()


def test_accumulate_save_load_roundtrip(tmp_path):
    """accumulate is persisted in the saved config and restored by load()."""
    p = str(tmp_path / "model")
    _acc_cbm(accumulate=True).save(p)
    assert ConditionalBornMachine.load(p).accumulate is True
    _acc_cbm(accumulate=False).save(p)
    assert ConditionalBornMachine.load(p).accumulate is False


def test_accumulate_is_settable_post_load(tmp_path):
    """The loaded model's accumulate can be overridden (train.py honors the
    current run's born.accumulate on model_path loads, since a0 checkpoints
    predate the flag) and the override actually routes _log_amp_sq."""
    p = str(tmp_path / "model")
    _acc_cbm(data_dim=4, accumulate=False).save(p)
    cbm = ConditionalBornMachine.load(p)
    assert cbm.accumulate is False
    cbm.accumulate = True  # what train.py does after load
    x = torch.rand(3, 4) * 1.6 - 0.8
    _overflow_scale_(cbm)
    assert torch.isfinite(cbm.class_probabilities(x)).all()  # accumulate path active
