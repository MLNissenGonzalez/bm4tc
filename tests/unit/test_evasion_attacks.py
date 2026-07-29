import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from tests.conftest import DATA_DIM, NUM_CLASSES
from src.utils.evasion import (
    JointProjectedGradientDescent,
    ProjectedGradientDescent,
    RobustnessEvaluation,
)
from src.utils.embeddings import range_size_of, rel_to_abs

BATCH = 16
# Authored relative, as every budget is. The fixtures use fourier (width 1.0), so
# here eps_abs happens to equal eps_rel — test_budget_conversion_is_embedding_aware
# is the one that pins the cases where they differ.
STRENGTH_FRACTION = 0.3
STRENGTH = STRENGTH_FRACTION  # == eps_abs on fourier only
STEPS = 20


# Use std=1.0 so amplitudes are O(1) for numerically stable gradient checks.
# (With _LOG_PROB_EPS=float32.tiny, std=1e-3 amplitudes no longer hit the
# clamp floor, but larger amplitudes still make gradient assertions more robust.)
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

    adversarials = attacker.generate(cbm_attack, naturals, labels, eps_abs=STRENGTH)

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

    adversarials = attacker.generate(cbm_attack, naturals, labels, eps_abs=STRENGTH)

    with torch.no_grad():
        adv_acc = (cbm_attack.class_probabilities(adversarials).argmax(1) == labels).float().mean().item()

    assert adv_acc < clean_acc, (
        f"Joint attack should decrease accuracy: clean={clean_acc:.3f}, adv={adv_acc:.3f}"
    )


def test_joint_pgd_perturbation_within_linf_ball(cbm_attack, naturals, labels):
    attacker = JointProjectedGradientDescent(norm="inf", num_steps=STEPS, random_start=False)
    adversarials = attacker.generate(cbm_attack, naturals, labels, eps_abs=STRENGTH)
    assert (adversarials - naturals).abs().max().item() <= STRENGTH + 1e-5


@pytest.mark.parametrize(
    "attack_cls", [ProjectedGradientDescent, JointProjectedGradientDescent]
)
def test_pgd_projects_onto_input_domain_and_linf_ball(cbm_attack, labels, attack_cls):
    lo, hi = cbm_attack.input_range
    naturals = torch.full((BATCH, DATA_DIM), lo)
    naturals[BATCH // 2:] = hi
    strength = 0.3
    attack = attack_cls(norm="inf", num_steps=3, random_start=True)

    adversarials = attack.generate(cbm_attack, naturals, labels, eps_abs=strength)

    assert adversarials.min().item() >= lo
    assert adversarials.max().item() <= hi
    assert (adversarials - naturals).abs().max().item() <= strength + 1e-6


# ---- RobustnessEvaluation with JOINT_PGD ----

def test_robustness_eval_joint_pgd_runs(cbm_attack, attack_loader):
    """RobustnessEvaluation dispatches JOINT_PGD and respects the absolute budget.

    ``eps_rel`` is carried for provenance; ``generate`` takes ``eps_abs``, which the
    caller derives via ``rel_to_abs``. On fourier (width 1.0) the two coincide.
    """
    eval_ = RobustnessEvaluation(
        method="JOINT_PGD", norm="inf", eps_rel=[STRENGTH_FRACTION],
        num_steps=5, random_start=False,
    )
    eps_abs = rel_to_abs(STRENGTH_FRACTION, range_size_of(cbm_attack))

    naturals, labels = next(iter(attack_loader))
    adv = eval_.generate(cbm_attack, naturals, labels, eps_abs)

    assert adv.shape == naturals.shape
    assert (adv - naturals).abs().max().item() <= eps_abs + 1e-6
