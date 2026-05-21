import pytest
import torch
from unittest.mock import patch
from src.models.cbm import CBMConfig, ConditionalBornMachine, MPSInitConfig


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
