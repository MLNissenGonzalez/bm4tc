import inspect
import math
import pytest
import torch
import tensorkrowch as tk
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset
from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
from bm4tc.core.objective import evaluate
from tests.conftest import raw_amplitudes


def _tiny_cbm(embedding="fourier", dtype="float32", data_dim=2, num_classes=2,
              bond_dim=2, std=1e-3):
    cfg = CBMConfig(
        embedding=embedding,
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=bond_dim, dtype=dtype, std=std),
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=data_dim, num_classes=num_classes)


def test_cbmconfig_defaults():
    cfg = CBMConfig()
    assert cfg.embedding == "legendre"
    assert cfg.model_path is None
    assert cfg.init_kwargs.dtype == "complex64"  # D4
    assert cfg.init_kwargs.init_method == "randn_eye"
    assert cfg.init_kwargs.std == 1e-9


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


def test_mixed_nll_beta0_skips_logZ():
    cbm = _tiny_cbm()
    x = torch.rand(4, 2)
    y = torch.randint(0, 2, (4,))
    with patch.object(cbm, "log_partition_function", wraps=cbm.log_partition_function) as mock_logZ:
        cbm.mixed_nll(x, y, beta=0.0)[0]
    mock_logZ.assert_not_called()


def test_mixed_nll_scalar_output():
    cbm = _tiny_cbm()
    x = torch.rand(4, 2)
    y = torch.randint(0, 2, (4,))
    loss = cbm.mixed_nll(x, y, beta=0.0)[0]
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


def _contracts_log_Z(cbm) -> bool:
    """Whether log_Z(recompute=False) contracts the norm, i.e. no log Z is cached
    for the current parameters (it caches one either way)."""
    with patch.object(cbm, "log_partition_function",
                      wraps=cbm.log_partition_function) as mock_logZ:
        cbm.log_Z(recompute=False)
    return mock_logZ.call_count == 1


def test_log_Z_recompute_false_computes_when_empty():
    """recompute=False with an empty cache still computes (and caches) once."""
    cbm = _tiny_cbm()
    assert cbm._log_Z_cache is None
    val = cbm.log_Z(recompute=False)
    assert torch.isfinite(val.real) and cbm.log_Z(recompute=False) is val


def _stepped(cbm, optimizer):
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    loss = cbm.mixed_nll(x, y, beta=0.5)[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("lr", [1e-3, 0.0])
def test_optimizer_step_expires_the_caches(lr):
    """D31: no caller invalidates. After optimizer.step() (even a no-op lr=0 one)
    log_Z(recompute=False) contracts again."""
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    _stepped(cbm, torch.optim.Adam(cbm.parameters(), lr=lr))
    assert _contracts_log_Z(cbm)


def test_log_Z_cache_does_not_cross_grad_modes():
    """A log Z formed under no_grad has no graph; it is never served with grad on."""
    cbm = _tiny_cbm()
    with torch.no_grad():
        cbm.log_Z(recompute=True)
    assert cbm.log_Z(recompute=False).requires_grad


def test_renormalize_invalidates_log_Z_cache():
    cbm = _tiny_cbm()
    cbm.log_Z(recompute=True)
    assert not _contracts_log_Z(cbm)
    cbm.renormalize_(log_target=0.0)
    assert _contracts_log_Z(cbm)


def test_initialize_invalidates_log_Z_cache():
    cbm = _tiny_cbm()
    cbm.log_Z(recompute=True)
    cbm.initialize()
    assert _contracts_log_Z(cbm)


# ── What mixed_nll returns and caches ──────────────────────────────────────

@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_mixed_nll_returns_its_log_amp_sq_detached(beta):
    cbm = _tiny_cbm()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    _, log_amp_sq = cbm.mixed_nll(x, y, beta=beta)
    assert not log_amp_sq.requires_grad
    assert torch.equal(log_amp_sq, cbm.log_amp_sq(x).detach())


def test_mixed_nll_logZ_cache_only_when_beta_positive():
    cbm = _tiny_cbm()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    cbm.mixed_nll(x, y, beta=0.0)
    assert _contracts_log_Z(cbm)       # beta=0 never contracts the norm
    cbm.mixed_nll(x, y, beta=1.0)
    assert not _contracts_log_Z(cbm)


def test_regularizer_reuses_mixed_nll_contraction():
    """beta>0: mixed_nll + NormRegularizer contract the norm once, total."""
    from bm4tc.core.objective import NormRegularizer
    cbm = _tiny_cbm()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    reg = NormRegularizer(strength=1.0, log_target=0.0)
    with patch.object(cbm, "log_partition_function",
                      wraps=cbm.log_partition_function) as mock_logZ:
        loss = cbm.mixed_nll(x, y, beta=1.0)[0] + reg(cbm)
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
        amp0 = raw_amplitudes(cbm, x)
        max_abs = float(amp0.abs().max().clamp(min=1e-30))
        c = (2e19 / max_abs) ** (1.0 / cbm.n_features)
        for node in cbm._mats_env:
            node.tensor.data.mul_(c)

    cbm.reset()
    with torch.no_grad():
        amp = raw_amplitudes(cbm, x)
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
    loss = cbm.mixed_nll(x, y, beta=0.5)[0]
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

    loss = cbm.mixed_nll(x, y, beta=0.0)[0]
    assert loss.isfinite(), "loss must be finite for small-amplitude CBM"
    loss.backward()

    grads = [p.grad for p in cbm.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients computed"
    assert all(g.isfinite().all() for g in grads), "gradients contain non-finite values"
    assert any(g.abs().max() > 0 for g in grads), "all gradients are zero (clamp floor too high)"


# ── Norm-accumulating (overflow-safe) contraction ────────────────────────────
# log_amp_sq contracts while keeping the running node O(1)
# and accumulate the extracted norm in log space, so log|ψ|² is recoverable
# without ever materializing an overflowing amplitude. Verified against the
# raw tensorkrowch contraction (raw_amplitudes).

def _acc_cbm(dtype="float32", data_dim=3, num_classes=2, bond_dim=3,
             std=0.3, out_position=None):
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=bond_dim, dtype=dtype,
                                  std=std, out_position=out_position),
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=data_dim, num_classes=num_classes)


def _overflow_scale_(cbm, scale=1e10):
    """Blow up the amplitude past float32 (no single contraction step overflows,
    so the accumulate/log_Z paths stay finite; the raw product does not)."""
    with torch.no_grad():
        for node in cbm._mats_env:
            node.tensor.data.mul_(scale)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def testlog_amp_sq_matches_amplitudes(dtype):
    """log_amp_sq(x) reproduces 2·log|amplitudes(x)| where the latter doesn't
    overflow (real + complex)."""
    torch.manual_seed(0)
    cbm = _acc_cbm(dtype=dtype)
    x = torch.rand(5, 3) * 1.6 - 0.8
    ref = 2.0 * torch.log(raw_amplitudes(cbm, x).abs().clamp(min=1e-30))
    got = cbm.log_amp_sq(x)
    assert got.shape == ref.shape == (5, cbm.out_dim)
    assert torch.allclose(got, ref, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def testlog_amp_sq_finite_when_embedding_vanishes(dtype):
    """A zero embedding vector must floor, not NaN, on the accumulate path.

    Regression: Chebyshev T2 vanishes at x = ±1, so the running contraction hit
    exactly zero and the unguarded renormalization did 0/0 → NaN in both
    psi_renorm and log_norm, killing training at step 2. T2's range is now
    restricted so data cannot land there, but the contraction itself must be
    safe for any future embedding with a node — hence x = ±1 is passed here
    deliberately, outside the declared domain.
    """
    cfg = CBMConfig(
        embedding="chebychev2",
        init_kwargs=MPSInitConfig(in_dim=6, bond_dim=3, dtype=dtype, std=0.3),
    )
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2)

    x = torch.tensor([[0.3, -0.2], [1.0, 0.5], [-1.0, -1.0]])
    psi, log_norm = cbm(cbm.embed(x))
    assert torch.isfinite(psi.abs()).all()
    assert torch.isfinite(log_norm).all()

    las = cbm.log_amp_sq(x)
    assert torch.isfinite(las).all()
    # The vanishing rows floor far below the interior row rather than NaN-ing.
    assert las[0].max() > las[1].max()
    assert las[0].max() > las[2].max()


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_amplitude_reconstruction(dtype):
    """psi_renorm · exp(log_norm) reconstructs the true amplitude magnitude."""
    torch.manual_seed(1)
    cbm = _acc_cbm(dtype=dtype)
    x = torch.rand(4, 3) * 1.6 - 0.8
    psi, log_norm = cbm(cbm.embed(x))
    assert psi.shape == log_norm.shape == (4, cbm.out_dim)
    assert not torch.is_complex(log_norm)
    assert torch.allclose(psi.abs(), torch.ones_like(psi.abs()), atol=1e-5)  # unit-modulus
    recon = psi.abs() * torch.exp(log_norm)
    assert torch.allclose(recon, raw_amplitudes(cbm, x).abs(), atol=1e-5)


def test_class_probabilities_fromlog_amp_sq():
    """class_probabilities rebuilt from log_amp_sq matches the native method —
    guards the per-(batch, class) accumulation (a per-batch collapse fails here)."""
    torch.manual_seed(2)
    cbm = _acc_cbm(num_classes=3)
    x = torch.rand(6, 3) * 1.6 - 0.8
    las = cbm.log_amp_sq(x)
    log_probs = las - torch.logsumexp(las, dim=-1, keepdim=True)
    assert torch.allclose(log_probs.exp(), cbm.class_probabilities(x), atol=1e-5)


def test_mixed_nll_beta0_fromlog_amp_sq():
    """mixed_nll(beta=0) = mean[-las[c] + logsumexp(las)] reconstructed from
    log_amp_sq (term1 + term2)."""
    torch.manual_seed(3)
    cbm = _acc_cbm()
    x = torch.rand(5, 3) * 1.6 - 0.8
    y = torch.randint(0, cbm.out_dim, (5,))
    las = cbm.log_amp_sq(x)
    term1 = -las[torch.arange(5), y]
    term2 = torch.logsumexp(las, dim=-1)
    recon = (term1 + term2).mean()
    assert torch.allclose(recon, cbm.mixed_nll(x, y, beta=0.0)[0], atol=1e-4)


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.9, 1.0])
def test_mixed_nll_is_the_per_variable_objective(beta):
    """D86: mixed_nll = (1-β)·L_dis + (β/N)·L_gen, N = n_features, which is
    c(β)·[(1-α)·L_dis + α·L_gen] with β/(1-β) = N·α/(1-α), c = 1 - β + β/N."""
    torch.manual_seed(4)
    cbm = _acc_cbm()
    x = torch.rand(5, 3) * 1.6 - 0.8
    y = torch.randint(0, cbm.out_dim, (5,))
    with torch.no_grad():
        las = cbm.log_amp_sq(x)
        log_Z = cbm.log_partition_function()
        l_dis = (torch.logsumexp(las, dim=-1) - las[torch.arange(5), y]).mean()
        l_gen = (log_Z - las[torch.arange(5), y]).mean()
        loss = cbm.mixed_nll(x, y, beta=beta)[0]
    n = cbm.n_features
    assert n == x.shape[1] + 1
    assert torch.allclose(loss, (1 - beta) * l_dis + beta / n * l_gen, atol=1e-4)
    if 0 < beta < 1:
        alpha = beta / (n - (n - 1) * beta)
        c = 1 - beta + beta / n
        assert torch.allclose(loss, c * ((1 - alpha) * l_dis + alpha * l_gen), atol=1e-4)


def testlog_amp_sq_overflow_safe():
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
    amp = raw_amplitudes(cbm, x)
    las = cbm.log_amp_sq(x)
    assert (~torch.isfinite(amp)).any(), "test scale did not overflow the amplitude"
    assert torch.isfinite(las).all()
    shift = 2.0 * cbm.n_features * math.log(scale)
    assert torch.allclose(las, las_base + shift, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("out_position", [0, 2, 4])
def testlog_amp_sq_out_position(out_position):
    """Correct across class site at first / middle / last position (obc
    boundaries + accumulator alignment across regions)."""
    torch.manual_seed(5)
    cbm = _acc_cbm(data_dim=4, num_classes=3, out_position=out_position)
    x = torch.rand(4, 4) * 1.6 - 0.8
    ref = 2.0 * torch.log(raw_amplitudes(cbm, x).abs().clamp(min=1e-30))
    assert torch.allclose(cbm.log_amp_sq(x), ref, atol=1e-4)


def test_accumulate_gradients_match():
    """Gradients of log_amp_sq match the direct 2·log|amplitudes| path on the
    parameters that participate in the amplitude forward."""
    torch.manual_seed(6)
    cbm = _acc_cbm()
    x = torch.rand(4, 3) * 1.6 - 0.8

    # By name: tk's reset re-registers the parameters, which reorders parameters().
    cbm.zero_grad()
    cbm.log_amp_sq(x).sum().backward()
    g_acc = {n: None if p.grad is None else p.grad.clone() for n, p in cbm.named_parameters()}

    cbm.zero_grad()
    amp = raw_amplitudes(cbm, x)
    (2.0 * torch.log(amp.abs().clamp(min=1e-30))).sum().backward()
    g_ref = {n: None if p.grad is None else p.grad.clone() for n, p in cbm.named_parameters()}

    assert {n: g is None for n, g in g_acc.items()} == {n: g is None for n, g in g_ref.items()}
    for n, a in g_acc.items():
        if a is not None:
            assert torch.allclose(a, g_ref[n], atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_amplitudes_accumulate_forward_backward(dtype):
    """The accumulate path is differentiable end-to-end: forward returns
    grad-tracking outputs and backward through *both* psi_renorm and log_norm
    reaches the parameters with finite, non-zero gradients (the reset() around
    the eager contraction must not sever the autograd graph)."""
    torch.manual_seed(8)
    cbm = _acc_cbm(dtype=dtype)
    x = torch.rand(4, 3) * 1.6 - 0.8

    psi, log_norm = cbm(cbm.embed(x))          # forward
    assert psi.requires_grad and log_norm.requires_grad
    assert psi.grad_fn is not None and log_norm.grad_fn is not None

    cbm.zero_grad()
    (psi.abs().sum() + log_norm.sum()).backward()          # backward through both
    grads = [p.grad for p in cbm.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max() > 0 for g in grads)


def test_accumulate_mode_isolation():
    """The norm-accumulating forward resets around itself, leaving the raw
    tensorkrowch contraction bit-identical before and after."""
    torch.manual_seed(7)
    cbm = _acc_cbm()
    x = torch.rand(4, 3) * 1.6 - 0.8
    before = raw_amplitudes(cbm, x)
    _ = cbm(cbm.embed(x))
    after = raw_amplitudes(cbm, x)
    assert torch.equal(before, after)


# ── The norm-accumulating contraction against the raw one ───────────────────

def test_mixed_nll_matches_raw_amplitudes():
    """mixed_nll and class_probabilities equal the raw-amplitude computation
    where it does not overflow."""
    torch.manual_seed(10)
    cbm = _acc_cbm()
    x = torch.rand(5, 3) * 1.6 - 0.8
    y = torch.randint(0, cbm.out_dim, (5,))

    log_abs = torch.log(raw_amplitudes(cbm, x).abs().clamp(min=1e-30))
    ref_probs = (2 * log_abs
                 - torch.logsumexp(2 * log_abs, dim=-1, keepdim=True)).exp()
    assert torch.allclose(cbm.class_probabilities(x), ref_probs, atol=1e-6)

    term1 = -2 * log_abs[torch.arange(5), y]
    term2 = torch.logsumexp(2 * log_abs, dim=-1)
    assert torch.allclose(cbm.mixed_nll(x, y, beta=0.0)[0],
                          (term1 + term2).mean(), atol=1e-5)


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_mixed_nll_finite_on_overflow(beta):
    """mixed_nll stays finite when the raw amplitude overflows (both the
    discriminative and generative ends)."""
    torch.manual_seed(11)
    cbm = _acc_cbm(data_dim=4)
    x = torch.rand(3, 4) * 1.6 - 0.8
    y = torch.randint(0, cbm.out_dim, (3,))
    # 1e8 overflows the raw amplitude (product over 5 sites > float32 max) while
    # keeping log_Z finite.
    _overflow_scale_(cbm, scale=1e8)

    assert (~torch.isfinite(raw_amplitudes(cbm, x))).any(), "scale did not overflow"
    assert torch.isfinite(cbm.mixed_nll(x, y, beta=beta)[0])


def test_class_probabilities_finite_on_overflow():
    """class_probabilities stays finite and normalized when the raw amplitude
    overflows (covers eval / all class_probabilities consumers)."""
    torch.manual_seed(12)
    cbm = _acc_cbm(data_dim=4, num_classes=3)
    x = torch.rand(4, 4) * 1.6 - 0.8
    _overflow_scale_(cbm)
    probs = cbm.class_probabilities(x)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def _eval_metrics(cbm, loader):
    out = evaluate(cbm, loader, "cpu")
    return out["loss_dis"], out["acc"], out["loss_x"]


def test_evaluate_finite_on_overflow():
    """evaluate's valid losses stay finite when the raw amplitude overflows —
    the MNIST-resize symptom (stable training, nan valid) that motivated routing
    eval through the same log_amp_sq as the loss."""
    torch.manual_seed(14)
    cbm = _acc_cbm(data_dim=4, num_classes=3)
    _overflow_scale_(cbm, scale=1e8)  # overflows the raw amplitude, keeps log_Z finite
    x = torch.rand(9, 4) * 1.6 - 0.8
    assert (~torch.isfinite(raw_amplitudes(cbm, x))).any(), "scale did not overflow"
    ds = TensorDataset(x, torch.randint(0, 3, (9,)))
    loader = DataLoader(ds, batch_size=4)

    dis, _, gen = _eval_metrics(cbm, loader)
    assert math.isfinite(dis) and math.isfinite(gen)


def test_marginal_log_probability_safe_parity():
    """marginal_log_probability (overflow-safe) equals the raw
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
    log_abs = torch.log(raw_amplitudes(cbm, x_raw).abs().clamp(min=1e-30))
    lp_raw = torch.logsumexp(2.0 * log_abs, dim=-1) - cbm.log_normalizer()
    lp_raw.sum().backward()

    assert torch.allclose(lp.detach(), lp_raw.detach(), atol=1e-4)
    assert torch.isfinite(g_safe).all() and g_safe.abs().max() > 0
    assert torch.allclose(g_safe, x_raw.grad, atol=1e-4)


def test_marginal_log_probability_finite_on_overflow():
    """marginal_log_probability stays finite when the raw amplitude overflows —
    the log-density primitive behind purification/UQ/MIA, so it must not go inf."""
    torch.manual_seed(16)
    cbm = _acc_cbm(data_dim=4, num_classes=3)
    _overflow_scale_(cbm, scale=1e8)
    cbm.cache_log_Z()
    x = torch.rand(4, 4) * 1.6 - 0.8

    assert (~torch.isfinite(raw_amplitudes(cbm, x))).any(), "scale did not overflow"
    assert torch.isfinite(cbm.marginal_log_probability(x)).all()


@pytest.mark.parametrize("data_dim,scale", [(120, 2.0), (200, 2.0), (120, 4.0)])
def test_sample_stable_on_long_overscaled_chain(data_dim, scale):
    """Class-conditional samples and a Gibbs sweep survive long, overscaled chains.

    The old canonical sampler concentrated the whole MPS norm into site 0 (~1e25 at
    data_dim=120) and raised from multinomial or from a non-converging SVD. The
    tensorkrowch sampler renormalizes each environment per sample (D100); no draw
    may fall back to uniform (that would warn)."""
    import warnings
    cbm = _acc_cbm(data_dim=data_dim, bond_dim=4)
    _overflow_scale_(cbm, scale=scale)
    torch.manual_seed(0)
    lo, hi = cbm.input_range
    grid = torch.linspace(lo, hi, 16)

    def embed(v):
        return cbm.embedding(v).to(cbm.dtype).reshape(*v.shape, -1)

    label = torch.nn.functional.one_hot(torch.tensor(0), cbm.out_dim).to(cbm.dtype)
    x = lo + (hi - lo) * torch.rand(4, data_dim)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        drawn = cbm.condition(label).sample(n_samples=4, domain=grid, embedding=embed,
                                            renormalize=True)
        swept = cbm.sample(in_condition=cbm.embed(x).to(cbm.dtype), domain=grid,
                           embedding=embed, renormalize=True)
    for s in (drawn, swept):
        assert s.shape == (4, data_dim)
        assert torch.isfinite(s).all()
        assert (s >= lo).all() and (s <= hi).all()
        # not collapsed onto a single grid value
        assert len(torch.unique(s)) > 1


# ── Boundary vectors (D87) ───────────────────────────────────────────────────

def _numeric_log_Z(cbm, grid=401):
    """log ∫ Σ_c |ψ(x, c)|² dx over the 2-D input square (trapezoid rule)."""
    lo, hi = cbm.input_range
    g = torch.linspace(lo, hi, grid)
    with torch.no_grad():
        lse = torch.logsumexp(cbm.log_joint(torch.cartesian_prod(g, g)), dim=1).double()
    w = torch.ones(grid, dtype=torch.float64)
    w[0] = w[-1] = 0.5
    weights = torch.outer(w, w).reshape(-1) * ((hi - lo) / (grid - 1)) ** 2
    return torch.logsumexp(lse + weights.log(), 0).item()


def test_boundaries_are_fixed_and_shared_with_norm_net():
    cbm = _tiny_cbm(embedding="legendre", dtype="complex64", bond_dim=3)
    for name in ("_left_node", "_right_node"):
        node, aux = getattr(cbm, name), getattr(cbm.norm_net, name)
        assert not node.tensor.requires_grad and not aux.tensor.requires_grad
        assert torch.equal(node.tensor, aux.tensor)
    cbm.initialize(tensors=[t.detach().clone() for t in cbm.tensors])
    for name in ("_left_node", "_right_node"):
        assert torch.equal(getattr(cbm, name).tensor, getattr(cbm.norm_net, name).tensor)
        assert not getattr(cbm, name).tensor.requires_grad


def test_log_Z_stays_the_normaliser_of_psi_during_training():
    """D87: training at β = 1 used to move the model's and norm_net's boundary
    vectors apart, so log Z stopped normalising ψ and the generative loss had a
    free direction. The contracted log Z must equal the integral of Σ_c |ψ|²."""
    torch.manual_seed(5)
    cbm = _tiny_cbm(embedding="legendre", dtype="complex64", bond_dim=3, std=1e-2)
    cbm.prepare(device=torch.device("cpu"))
    opt = torch.optim.Adam(cbm.parameters(), lr=0.05)
    x, y = torch.rand(64, 2) * 2 - 1, torch.randint(0, 2, (64,))
    for _ in range(30):
        opt.zero_grad()
        cbm.mixed_nll(x, y, beta=1.0)[0].backward()
        opt.step()
    log_Z = float(cbm.log_partition_function())
    assert abs(log_Z - _numeric_log_Z(cbm)) < 1e-3


def test_every_contraction_of_the_model_zips_the_data_in(cbm, x_batch):
    """D112, bm4tc-wide: the model's ``contract`` zips by default (tk's own
    ``forward`` and ``trace`` call it too), and gives tk's tensor."""
    assert inspect.signature(cbm.contract).parameters["zipup_input"].default is True
    data = cbm.embed(x_batch)
    outs = []
    for kwargs in ({}, {"zipup_input": False}):
        cbm.reset()
        outs.append(tk.models.MPS.forward(cbm, data, **kwargs))
    cbm.reset()
    assert torch.allclose(outs[0], outs[1], rtol=1e-5, atol=1e-7)


# ── renormalization every k sites (D119) ─────────────────────────────────────

def _values_and_gradients(cbm, x):
    cbm.reset()
    las = cbm.log_amp_sq(x)
    log_Z = cbm.log_partition_function()
    params = [p for _, p in sorted(cbm.named_parameters()) if p.requires_grad]
    grads = torch.autograd.grad(las.sum() + log_Z, params, allow_unused=True)
    return las.detach(), log_Z.detach(), grads


@pytest.mark.parametrize("k", [2, 3, 8, 1000])
def test_renormalizing_every_k_sites_changes_only_rounding(monkeypatch, k):
    """log|ψ|², log Z and their gradients are those of k = 1: the norm is kept
    in log space whenever the running node is renormalized."""
    import bm4tc.core.model as model_module
    torch.manual_seed(0)
    cbm = _tiny_cbm(dtype="complex64", data_dim=9, num_classes=3, bond_dim=4,
                    std=0.3)
    x = torch.rand(5, 9)
    expected = _values_and_gradients(cbm, x)
    monkeypatch.setattr(model_module, "RENORMALIZE_EVERY", k)
    result = _values_and_gradients(cbm, x)
    assert torch.allclose(result[0], expected[0], rtol=1e-5, atol=1e-5)
    assert torch.allclose(result[1], expected[1], rtol=1e-5, atol=1e-5)
    for grad, expected_grad in zip(result[2], expected[2]):
        if expected_grad is None:
            assert grad is None
        else:
            assert torch.allclose(grad, expected_grad, rtol=1e-4, atol=1e-5)


def test_the_forward_renormalizes_every_k_steps(monkeypatch):
    import bm4tc.core.model as model_module
    cbm = _tiny_cbm(data_dim=9, num_classes=3)
    calls = []
    norm = ConditionalBornMachine._safe_bond_norm
    monkeypatch.setattr(ConditionalBornMachine, "_safe_bond_norm",
                        staticmethod(lambda node, axes: calls.append(1) or norm(node, axes)))
    monkeypatch.setattr(model_module, "RENORMALIZE_EVERY", 4)
    cbm.log_amp_sq(torch.rand(2, 9))
    assert len(calls) == 2          # 9 input sites: after the 4th and the 8th step
