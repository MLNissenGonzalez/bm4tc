"""Training on the isometric form (D192): trimmed bonds; the isometric start keeps p(c|x)
and sets Z = 1; POGO's steps keep the isometries and train; the refusals."""
import pytest
import torch

from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig, trimmed_bond_dims
from bm4tc.core.objective import VectorAdam
from bm4tc.core.pogo import POGO, isometric_start_, isometry_defects, log_Z_from_centre, pogo_step


def _cbm(trim=True, bond_dim=6, data_dim=8, init_method="randn_eye", seed=0):
    torch.manual_seed(seed)
    cfg = CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(
        in_dim=3, bond_dim=bond_dim, trim_bonds=trim, init_method=init_method, std=0.5
        if init_method == "randn" else 1e-9))
    return ConditionalBornMachine(cfg=cfg, data_dim=data_dim, num_classes=2)


def _loss(cbm, x, y):
    cbm.invalidate_log_Z_cache()
    return -(cbm.log_amp_sq(x).gather(1, y[:, None]).mean() - cbm.log_partition_function())


def test_trimmed_bond_dims():
    # 9 sites, the class (2) in the middle: bond 0 reaches 3, bond 1 9 > 6
    assert trimmed_bond_dims([3, 3, 3, 3, 2, 3, 3, 3, 3], 6) == [3, 6, 6, 6, 6, 6, 6, 3]
    assert trimmed_bond_dims([3, 2, 3], 10) == [3, 3]
    assert _cbm(trim=False).bond_dim == [6] * 8
    assert _cbm(trim=True).bond_dim == [3, 6, 6, 6, 6, 6, 6, 3]


@pytest.mark.parametrize("init_method", ["randn_eye", "randn"])
def test_isometric_start_keeps_p_c_given_x_and_sets_Z_to_one(init_method):
    cbm = _cbm(init_method=init_method)
    x = torch.rand(16, 8) * 2 - 1
    with torch.no_grad():
        before = cbm.log_amp_sq(x) - cbm.log_partition_function()
    isometric_start_(cbm)
    cbm.invalidate_log_Z_cache()
    with torch.no_grad():
        assert cbm.log_partition_function().item() == pytest.approx(0.0, abs=1e-5)
        assert torch.allclose(cbm.log_amp_sq(x), before, atol=1e-4)
    assert max(isometry_defects(cbm)) < 1e-5


def test_isometric_start_refuses_untrimmed_bonds():
    with pytest.raises(ValueError, match="trim_bonds"):
        isometric_start_(_cbm(trim=False))


def test_pogo_step_lands_near_the_manifold():
    """From a point of the Stiefel manifold, a step of size s leaves a defect of order s²
    (the landing removes the first order); a step above ‖Δ‖ = 1 is capped."""
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(12, 4, dtype=torch.complex128))
    x = q.mH                                            # 4 × 12, X Xᴴ = I
    eye = torch.eye(4, dtype=x.dtype)
    direction = torch.randn_like(x)
    direction /= torch.linalg.norm(direction)
    defects = [torch.linalg.norm((y := pogo_step(x, s * direction)) @ y.mH - eye).item()
               for s in (1e-2, 1e-3)]
    assert defects[1] < defects[0] / 50
    assert torch.allclose(pogo_step(x, 5 * direction), pogo_step(x, direction), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("base", ["sgd", "vadam"])
def test_pogo_trains_and_keeps_the_isometries(base):
    cbm = _cbm(bond_dim=4)
    isometric_start_(cbm)
    params = [n.tensor for n in cbm._mats_env]
    opt = torch.optim.SGD(params, lr=0.05) if base == "sgd" else VectorAdam(params, lr=0.1)
    pogo = POGO(cbm, opt)
    torch.manual_seed(1)
    x, y = torch.rand(64, 8) * 2 - 1, torch.randint(0, 2, (64,))
    with torch.no_grad():
        first = _loss(cbm, x, y).item()
    for _ in range(40):
        pogo.zero_grad()
        _loss(cbm, x, y).backward()
        pogo.step()
    with torch.no_grad():
        assert _loss(cbm, x, y).item() < first - 0.05
        cbm.invalidate_log_Z_cache()
        assert abs(cbm.log_partition_function().item()) < 1e-3
    assert max(isometry_defects(cbm)) < 1e-3


def test_pogo_refuses_a_start_off_the_manifold():
    cbm = _cbm()
    with pytest.raises(ValueError, match="isometric_start_"):
        POGO(cbm, torch.optim.SGD([n.tensor for n in cbm._mats_env], lr=0.1))


def test_pogo_leaves_the_unreached_boundary_components_alone():
    """The edge cores' components no boundary vector reaches are never written."""
    cbm = _cbm(bond_dim=4)
    isometric_start_(cbm)
    edges = [cbm._mats_env[0].tensor[1:].clone(), cbm._mats_env[-1].tensor[..., 1:].clone()]
    params = [n.tensor for n in cbm._mats_env]
    pogo = POGO(cbm, torch.optim.SGD(params, lr=0.05))
    x, y = torch.rand(16, 8) * 2 - 1, torch.randint(0, 2, (16,))
    pogo.zero_grad()
    _loss(cbm, x, y).backward()
    pogo.step()
    assert torch.equal(cbm._mats_env[0].tensor[1:], edges[0])
    assert torch.equal(cbm._mats_env[-1].tensor[..., 1:], edges[1])


def test_a_trimmed_isometric_model_saves_and_loads(tmp_path):
    cbm = _cbm(bond_dim=4)
    isometric_start_(cbm)
    cbm.invalidate_log_Z_cache()
    cbm.save(str(tmp_path / "m.pt"))
    back = ConditionalBornMachine.load(str(tmp_path / "m.pt"))
    x = torch.rand(8, 8) * 2 - 1
    with torch.no_grad():
        assert back.bond_dim == cbm.bond_dim
        assert torch.allclose(back.log_amp_sq(x), cbm.log_amp_sq(x), atol=1e-6)
        assert back.log_partition_function().item() == pytest.approx(0.0, abs=1e-5)


def test_pogo_refuses_adam():
    cbm = _cbm(bond_dim=4)
    isometric_start_(cbm)
    with pytest.raises(ValueError, match="linear in the"):
        POGO(cbm, torch.optim.Adam([n.tensor for n in cbm._mats_env], lr=0.01))


def test_log_Z_from_the_centre():
    """On the isometric form it is the contracted log Z; its gradient is the class core's
    only, and there the contraction's."""
    cbm = _cbm(init_method="randn")
    isometric_start_(cbm)
    with torch.no_grad():
        cbm._mats_env[cbm.out_position].tensor.mul_(3.0)     # off Z = 1, still canonical
    cbm.invalidate_log_Z_cache()
    cores = [n.tensor for n in cbm._mats_env]
    read, full = log_Z_from_centre(cbm), cbm.log_partition_function()
    assert read.item() == pytest.approx(full.item(), abs=1e-5)
    got = torch.autograd.grad(read, cores, allow_unused=True)
    assert [g is not None for g in got] == [v == cbm.out_position for v in range(len(cores))]
    want = torch.autograd.grad(full, cores)[cbm.out_position]
    assert torch.allclose(got[cbm.out_position], want, atol=1e-4)


@pytest.mark.parametrize("capturable", [False, True])
def test_vector_adam_is_the_papers_rule(capturable):
    """One second moment per tensor, the squared Frobenius norm (App. B)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3, 4, 5, dtype=torch.complex128))
    opt = VectorAdam([p], lr=0.2, capturable=capturable)
    want, m, v = p.detach().clone(), torch.zeros_like(p.detach()), 0.0
    for t in range(1, 8):
        g = torch.randn(3, 4, 5, dtype=torch.complex128)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g.abs().square().sum().item()
        want -= 0.2 * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)) ** 0.5 + 1e-8)
        p.grad = g
        opt.step()
        assert torch.allclose(p.detach(), want, rtol=1e-9, atol=1e-12)
    assert opt.state[p]["exp_avg_sq"].shape == ()


def test_vector_adam_commutes_with_a_unitary_gauge():
    """Def. 1's consequence: steps on X Q are the steps on X, times Q."""
    torch.manual_seed(1)
    x = torch.randn(6, 5, dtype=torch.complex128)
    q = torch.linalg.qr(torch.randn(5, 5, dtype=torch.complex128))[0]
    a, b = torch.nn.Parameter(x.clone()), torch.nn.Parameter(x @ q)
    oa, ob = VectorAdam([a], lr=0.1), VectorAdam([b], lr=0.1)
    for _ in range(5):
        g = torch.randn(6, 5, dtype=torch.complex128)
        a.grad, b.grad = g, g @ q
        oa.step()
        ob.step()
    assert torch.allclose(a.detach() @ q, b.detach(), atol=1e-10)
