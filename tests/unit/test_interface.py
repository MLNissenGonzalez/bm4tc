"""The analysis interface (D69) on the MPS: the derived quantities are the model's own,
bit for bit where the computation is the same, and to float precision for the joint
attack (which moved from the direct amplitudes path to log_amp_sq)."""
import torch

from bm4tc.core.interface import best_wrong_log_joint, class_probabilities, log_px, nll

_EPS = float(torch.finfo(torch.float32).tiny)


def _data(cbm, n=8):
    torch.manual_seed(0)
    lo, hi = cbm.input_range
    return lo + (hi - lo) * torch.rand(n, 4), torch.randint(0, cbm.out_dim, (n,))


def test_mps_derived_quantities_are_its_own(cbm):
    x, y = _data(cbm)
    assert torch.equal(class_probabilities(cbm, x), cbm.class_probabilities(x))
    assert torch.equal(nll(cbm, x, y), cbm.mixed_nll(x, y, alpha=0.0))
    if cbm.accumulate:
        assert torch.equal(log_px(cbm, x), cbm.marginal_log_probability(x))
    else:
        torch.testing.assert_close(log_px(cbm, x), cbm.marginal_log_probability(x))


def test_nll_gradient_matches_mixed_nll(cbm):
    x, y = _data(cbm)
    grads = []
    for loss in (lambda d: nll(cbm, d, y), lambda d: cbm.mixed_nll(d, y, alpha=0.0)):
        d = x.clone().requires_grad_(True)
        loss(d).backward()
        grads.append(d.grad)
    assert torch.equal(*grads)


def test_joint_loss_matches_amplitudes(cbm):
    x, y = _data(cbm)
    log_joint = 2 * torch.log(cbm.amplitudes(x).abs().clamp(min=_EPS))
    wrong = torch.ones_like(log_joint, dtype=torch.bool)
    wrong[torch.arange(len(y)), y] = False
    old = log_joint.masked_fill(~wrong, float("-inf")).max(dim=-1).values.mean()
    torch.testing.assert_close(best_wrong_log_joint(cbm, x, y), old)


def test_log_normalizer_follows_the_parameters():
    from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
    cfg = CBMConfig(embedding="fourier", init_kwargs=MPSInitConfig(in_dim=4, bond_dim=2))
    model = ConditionalBornMachine(cfg=cfg, data_dim=4, num_classes=2, device="cpu")
    model.prepare(device="cpu")
    before = model.log_normalizer()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.0)   # bumps the version counters
    assert model.log_normalizer() != before
