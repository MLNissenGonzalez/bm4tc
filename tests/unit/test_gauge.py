"""The gauge step (D190): ψ, log Z and p(c|x) unchanged; the node norm towards its
minimum; the optimiser's state carried along; the config checks."""
import math

import pytest
import torch

from bm4tc.core.gauge import GaugeConfig, check, gauge_diagnostics, gauge_step_
from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig

KINDS = [GaugeConfig("scalar"), GaugeConfig("balanced"), GaugeConfig("balanced", rate=0.5)]


def _unbalanced_cbm(bond_dim=3, data_dim=8, dtype="complex64", seed=0):
    """A random model whose cores sit at very different scales (ψ fixed by the seed)."""
    torch.manual_seed(seed)
    cfg = CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(
        in_dim=3, bond_dim=bond_dim, dtype=dtype, init_method="randn", std=0.5))
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=data_dim, num_classes=2)
    with torch.no_grad():
        for node in cbm._mats_env:
            node.tensor.data.mul_(math.exp(2 * torch.randn(1).item()))
    cbm.invalidate_log_Z_cache()
    return cbm


def _loss(cbm, x, y):
    return -(cbm.log_amp_sq(x).gather(1, y[:, None]).mean() - cbm.log_partition_function())


@pytest.mark.parametrize("cfg", KINDS, ids=lambda c: f"{c.kind}-{c.rate}")
@pytest.mark.parametrize("bond_dim", [3, 6])   # 6 > d: rank-deficient near the edges
def test_psi_and_log_Z_unchanged_norm_down(cfg, bond_dim):
    cbm = _unbalanced_cbm(bond_dim=bond_dim)
    x = torch.rand(16, 8) * 2 - 1
    with torch.no_grad():
        las, log_z = cbm.log_amp_sq(x), cbm.log_partition_function()
    before = gauge_diagnostics(cbm)
    sweeps = gauge_step_(cbm, None, cfg)
    cbm.invalidate_log_Z_cache()
    with torch.no_grad():
        assert torch.allclose(cbm.log_amp_sq(x), las, atol=2e-4)
        assert cbm.log_partition_function().item() == pytest.approx(log_z.item(), abs=2e-4)
    after = gauge_diagnostics(cbm)
    assert after["gauge/log_norm"] <= before["gauge/log_norm"] + 1e-6
    assert sweeps >= 1
    if cfg.kind == "scalar":
        assert after["gauge/log_norm_over_scalar"] == pytest.approx(0.0, abs=1e-5)


def test_exact_balance_reaches_the_balanced_gauge():
    """Full-rank bonds (r = d): every defect to the tolerance, in few sweeps."""
    cbm = _unbalanced_cbm(bond_dim=3)
    assert gauge_diagnostics(cbm)["gauge/defect_max"] > 0.1
    sweeps = gauge_step_(cbm, None, GaugeConfig("balanced"))
    assert sweeps < 200
    assert gauge_diagnostics(cbm)["gauge/defect_max"] < 1e-4


def test_soft_steps_converge_to_the_exact_balance():
    soft, exact = _unbalanced_cbm(), _unbalanced_cbm()
    for _ in range(300):
        gauge_step_(soft, None, GaugeConfig("balanced", rate=0.5))
    gauge_step_(exact, None, GaugeConfig("balanced"))
    assert gauge_diagnostics(soft)["gauge/log_norm"] == pytest.approx(
        gauge_diagnostics(exact)["gauge/log_norm"], abs=1e-4)


def test_randn_eye_noise_survives_a_balance():
    """randn_eye's 1e-9 noise directions carry the gradient that lets training leave rank
    1; a balance must not drop them. Only the components along an exact null space go
    (bond 0 and the last: the edge cores reach d = 3 of the r = 6 components), which
    touches cores 1 and n - 2 only."""
    torch.manual_seed(0)
    cfg = CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=3, bond_dim=6))
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=8, num_classes=2)
    x = torch.rand(4, 8) * 2 - 1
    with torch.no_grad():
        las = cbm.log_amp_sq(x)
    interior = range(2, len(cbm._mats_env) - 2)
    zeros = [int((cbm._mats_env[v].tensor == 0).sum()) for v in interior]
    for kind in KINDS:
        gauge_step_(cbm, None, kind)
    assert [int((cbm._mats_env[v].tensor == 0).sum()) for v in interior] == zeros
    cbm.invalidate_log_Z_cache()
    with torch.no_grad():
        assert torch.allclose(cbm.log_amp_sq(x)[:, 0], las[:, 0], atol=1e-5)


def _gradients(cbm, x, y):
    cbm.zero_grad(set_to_none=True)
    cbm.invalidate_log_Z_cache()
    _loss(cbm, x, y).backward()
    return [node.tensor.grad.clone() for node in cbm._mats_env]


@pytest.mark.parametrize("cfg", KINDS, ids=lambda c: f"{c.kind}-{c.rate}")
def test_first_moments_move_as_gradients(cfg):
    """SGD's momentum buffer set to the gradient, then a gauge step: the buffer equals the
    gradient at the new point (the transport is exact for a gradient)."""
    cbm = _unbalanced_cbm(dtype="complex128")
    x, y = torch.rand(16, 8) * 2 - 1, torch.randint(0, 2, (16,))
    opt = torch.optim.SGD(cbm.parameters(), lr=0.0, momentum=0.9)
    for node, g in zip(cbm._mats_env, _gradients(cbm, x, y)):
        opt.state[node.tensor]["momentum_buffer"] = g.clone()
    gauge_step_(cbm, opt, cfg)
    for node, g in zip(cbm._mats_env, _gradients(cbm, x, y)):
        buf = opt.state[node.tensor]["momentum_buffer"]
        assert torch.allclose(buf, g, rtol=1e-6, atol=1e-9 * g.abs().max().item())


def test_scalar_gauge_moves_second_moments_exactly():
    """Under a diagonal gauge |∇|² moves exactly: Adam's exp_avg_sq set to |∇|² stays it."""
    cbm = _unbalanced_cbm(dtype="complex128")
    x, y = torch.rand(16, 8) * 2 - 1, torch.randint(0, 2, (16,))
    opt = torch.optim.Adam(cbm.parameters(), lr=0.0)
    for node, g in zip(cbm._mats_env, _gradients(cbm, x, y)):
        opt.state[node.tensor].update(exp_avg=g.clone(), exp_avg_sq=g.abs().square(),
                                      step=torch.tensor(1.0))
    gauge_step_(cbm, opt, GaugeConfig("scalar"))
    for node, g in zip(cbm._mats_env, _gradients(cbm, x, y)):
        state = opt.state[node.tensor]
        scale = g.abs().max().item()
        assert torch.allclose(state["exp_avg"], g, rtol=1e-6, atol=1e-9 * scale)
        assert torch.allclose(state["exp_avg_sq"], g.abs().square(), rtol=1e-6, atol=1e-9 * scale ** 2)


def test_none_changes_nothing():
    cbm = _unbalanced_cbm()
    before = [node.tensor.clone() for node in cbm._mats_env]
    assert gauge_step_(cbm, None, GaugeConfig("none")) == 0
    assert all(torch.equal(a, node.tensor) for a, node in zip(before, cbm._mats_env))


@pytest.mark.parametrize("kwargs, match", [
    ({"kind": "isometric"}, "kind"),
    ({"kind": "scalar", "every": 0}, "every"),
    ({"kind": "scalar", "rate": 0.1}, "rate"),
    ({"kind": "none", "rate": 0.1}, "rate"),
    ({"kind": "balanced", "rate": 0.0}, "rate"),
])
def test_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GaugeConfig(**kwargs)


def test_unknown_optimizer_state_raises():
    cbm = _unbalanced_cbm()
    opt = torch.optim.Adam(cbm.parameters(), lr=1e-3)
    opt.state[cbm._mats_env[0].tensor]["mystery"] = torch.zeros_like(cbm._mats_env[0].tensor)
    with pytest.raises(ValueError, match="mystery"):
        gauge_step_(cbm, opt, GaugeConfig("scalar"))
