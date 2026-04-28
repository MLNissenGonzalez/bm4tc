import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tests.conftest import DATA_DIM, NUM_CLASSES
from src.utils.evasion.minimal import (
    JointProjectedGradientDescent,
    RobustnessEvaluation,
)

BATCH = 16
STRENGTH = 0.3
STEPS = 20


# Use std=1.0 so amplitudes are O(1) and gradients don't underflow.
# The conftest born_machine uses std=1e-3 → amplitudes ~1e-12 → at the
# clamp(min=1e-12) floor → zero gradients in all amplitude-based paths.
@pytest.fixture(scope="module")
def bm_attack():
    from src.models.born import BornMachine
    cfg = OmegaConf.create({
        "embedding": "fourier",
        "init_kwargs": {
            "in_dim": DATA_DIM, "bond_dim": 2, "out_position": 2,
            "boundary": "obc", "init_method": "randn",
            "dtype": "complex64", "n_features": None, "out_dim": None,
            "std": 1.0,
        },
        "model_path": None,
    })
    return BornMachine(cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu")


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

def test_joint_pgd_increases_wrong_class_log_joint(bm_attack, naturals, labels):
    """After several steps the max wrong-class log-joint should be higher than at start."""
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)

    K = bm_attack.out_dim
    eps = 1e-12

    with torch.no_grad():
        amps_before = bm_attack.classifier.amplitudes(naturals)
        log_joint_before = 2 * torch.log(amps_before.abs().clamp(min=eps))
        mask = torch.zeros(BATCH, K, dtype=torch.bool)
        mask[torch.arange(BATCH), labels] = True
        max_wrong_before = log_joint_before.masked_fill(mask, float("-inf")).max(dim=-1).values.mean().item()

    adversarials = attacker.generate(bm_attack, naturals, labels, strength=STRENGTH)

    with torch.no_grad():
        amps_after = bm_attack.classifier.amplitudes(adversarials)
        log_joint_after = 2 * torch.log(amps_after.abs().clamp(min=eps))
        max_wrong_after = log_joint_after.masked_fill(mask, float("-inf")).max(dim=-1).values.mean().item()

    assert max_wrong_after > max_wrong_before, (
        f"Joint attack should increase wrong-class log-joint: "
        f"before={max_wrong_before:.4f}, after={max_wrong_after:.4f}"
    )


def test_joint_pgd_reduces_accuracy(bm_attack, naturals, labels):
    """Joint attack should reduce classifier accuracy (not leave it unchanged)."""
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)

    with torch.no_grad():
        clean_acc = (bm_attack.classifier.probabilities(naturals).argmax(1) == labels).float().mean().item()

    adversarials = attacker.generate(bm_attack, naturals, labels, strength=STRENGTH)

    with torch.no_grad():
        adv_acc = (bm_attack.classifier.probabilities(adversarials).argmax(1) == labels).float().mean().item()

    assert adv_acc < clean_acc, (
        f"Joint attack should decrease accuracy: clean={clean_acc:.3f}, adv={adv_acc:.3f}"
    )


def test_joint_pgd_perturbation_within_linf_ball(bm_attack, naturals, labels):
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)
    adversarials = attacker.generate(bm_attack, naturals, labels, strength=STRENGTH)
    assert (adversarials - naturals).abs().max().item() <= STRENGTH + 1e-5


# ---- RobustnessEvaluation with JOINT_PGD ----

def test_robustness_eval_joint_pgd_runs(bm_attack, attack_loader):
    """RobustnessEvaluation with JOINT_PGD should run without errors."""
    eval_ = RobustnessEvaluation(
        method="JOINT_PGD", norm="inf", strengths=[STRENGTH],
        num_steps=5, random_start=False,
    )
    accs = eval_.evaluate(bm_attack, attack_loader)
    assert len(accs) == 1
    assert 0.0 <= accs[0] <= 1.0
