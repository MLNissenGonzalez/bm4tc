import inspect
import pytest
import torch
import tensorkrowch as tk
from unittest.mock import patch
from src.model import CBMConfig, ConditionalBornMachine, MPSInitConfig


def _tiny_cbm(embedding="fourier", dtype="float32", data_dim=2, num_classes=2):
    cfg = CBMConfig(
        embedding=embedding,
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, dtype=dtype, std=1e-3),
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
