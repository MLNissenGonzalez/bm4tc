import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from tests.conftest import DATA_DIM, NUM_CLASSES
from src.utils.evasion import (
    JointProjectedGradientDescent,
    RobustnessEvaluation,
)

BATCH = 16
# Relative fraction of embedding range; fourier range_size=1.0, so abs_eps=0.3×1.0=0.3
STRENGTH_FRACTION = 0.3
STRENGTH = STRENGTH_FRACTION  # equals abs_eps for fourier (range_size=1.0)
STEPS = 20


# Use std=1.0 so amplitudes are O(1) and gradients don't underflow.
# The conftest cbm fixture uses std=1e-3 → amplitudes ~1e-12 → at the
# clamp(min=1e-12) floor → zero gradients in all amplitude-based paths.
@pytest.fixture(scope="module")
def cbm_attack():
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    torch.manual_seed(42)
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=DATA_DIM, bond_dim=2, std=1.0),
    )
    model = ConditionalBornMachine(cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu")
    model.prepare(device="cpu")
    return model


@pytest.fixture
def naturals():
    torch.manual_seed(0)
    return torch.rand(BATCH, DATA_DIM)


@pytest.fixture
def labels():
    torch.manual_seed(1)
    return torch.randint(0, NUM_CLASSES, (BATCH,))


@pytest.fixture
def attack_loader():
    ds = TensorDataset(torch.rand(32, DATA_DIM), torch.randint(0, NUM_CLASSES, (32,)))
    return DataLoader(ds, batch_size=8, shuffle=False)


# ---- JointProjectedGradientDescent: gradient direction ----

def test_joint_pgd_increases_wrong_class_log_joint(cbm_attack, naturals, labels):
    """After several steps the max wrong-class log-joint should be higher than at start."""
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)

    K = cbm_attack.out_dim
    eps = 1e-12

    with torch.no_grad():
        amps_before = cbm_attack.amplitudes(naturals)
        log_joint_before = 2 * torch.log(amps_before.abs().clamp(min=eps))
        mask = torch.zeros(BATCH, K, dtype=torch.bool)
        mask[torch.arange(BATCH), labels] = True
        max_wrong_before = log_joint_before.masked_fill(mask, float("-inf")).max(dim=-1).values.mean().item()

    adversarials = attacker.generate(cbm_attack, naturals, labels, strength=STRENGTH)

    with torch.no_grad():
        amps_after = cbm_attack.amplitudes(adversarials)
        log_joint_after = 2 * torch.log(amps_after.abs().clamp(min=eps))
        max_wrong_after = log_joint_after.masked_fill(mask, float("-inf")).max(dim=-1).values.mean().item()

    assert max_wrong_after > max_wrong_before, (
        f"Joint attack should increase wrong-class log-joint: "
        f"before={max_wrong_before:.4f}, after={max_wrong_after:.4f}"
    )


def test_joint_pgd_reduces_accuracy(cbm_attack, naturals, labels):
    """Joint attack should reduce classifier accuracy (not leave it unchanged)."""
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)

    with torch.no_grad():
        clean_acc = (cbm_attack.class_probabilities(naturals).argmax(1) == labels).float().mean().item()

    chance = 1.0 / NUM_CLASSES
    if clean_acc <= chance:
        pytest.skip(f"Clean accuracy ({clean_acc:.3f}) already at chance floor; attack cannot reduce it further.")

    adversarials = attacker.generate(cbm_attack, naturals, labels, strength=STRENGTH)

    with torch.no_grad():
        adv_acc = (cbm_attack.class_probabilities(adversarials).argmax(1) == labels).float().mean().item()

    assert adv_acc < clean_acc, (
        f"Joint attack should decrease accuracy: clean={clean_acc:.3f}, adv={adv_acc:.3f}"
    )


def test_joint_pgd_perturbation_within_linf_ball(cbm_attack, naturals, labels):
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)
    adversarials = attacker.generate(cbm_attack, naturals, labels, strength=STRENGTH)
    assert (adversarials - naturals).abs().max().item() <= STRENGTH + 1e-5


# ---- RobustnessEvaluation with JOINT_PGD ----

def test_robustness_eval_joint_pgd_runs(cbm_attack, attack_loader):
    """RobustnessEvaluation with JOINT_PGD should run without errors.
    strengths=[STRENGTH_FRACTION] is a relative fraction; abs_eps = fraction × range_size.
    For fourier embedding range_size=1.0, so abs_eps=STRENGTH_FRACTION.
    """
    eval_ = RobustnessEvaluation(
        method="JOINT_PGD", norm="inf", strengths=[STRENGTH_FRACTION],
        num_steps=5, random_start=False,
    )
    accs = eval_.evaluate(cbm_attack, attack_loader)
    assert len(accs) == 1
    assert 0.0 <= accs[0] <= 1.0
