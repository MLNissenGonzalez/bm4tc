"""Adaptive attacks through purifier and detector (bm4tc/analysis/adaptive.py, D96)."""
import pytest
import torch

from tests.conftest import DATA_DIM, NUM_CLASSES
from bm4tc.analysis.adaptive import AdaptivePGD, Defense
from bm4tc.analysis.purification import LikelihoodPurification
from bm4tc.core.attacks import ProjectedGradientDescent
from bm4tc.core.interface import log_px

pytestmark = pytest.mark.slow

EPS = 0.2   # fourier: input range [0, 1], so absolute = relative


@pytest.fixture(scope="module")
def model():
    from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
    torch.manual_seed(42)
    cfg = CBMConfig(embedding="fourier",
                    init_kwargs=MPSInitConfig(in_dim=DATA_DIM, bond_dim=2, std=1.0))
    m = ConditionalBornMachine(cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu")
    m.prepare(device="cpu")
    m.cache_log_Z()
    return m


@pytest.fixture(scope="module")
def data(model):
    torch.manual_seed(0)
    x = torch.rand(32, DATA_DIM)
    with torch.no_grad():
        y = model.log_joint(x).argmax(dim=1)   # clean points all right
    return x, y


def _margin(model, z, y):
    lj = model.log_joint(z)
    return lj[torch.arange(len(y)), y] - lj[torch.arange(len(y)), 1 - y]


def test_backward_is_the_gradient_at_the_purified_point(model, data):
    """BPDA: the defence's loss gradient in x is the margin's gradient at P(x)."""
    x, y = data
    defense = Defense(model, purify=lambda v: (v + 0.1).clamp(0, 1))

    xg = x.clone().requires_grad_(True)
    loss, _ = defense.attack_step(xg, y, draws=1)
    (got,) = torch.autograd.grad(loss.sum(), xg)

    z = (x + 0.1).clamp(0, 1).requires_grad_(True)
    (want,) = torch.autograd.grad(_margin(model, z, y).sum(), z)
    torch.testing.assert_close(got, want)


def test_without_defence_it_is_at_least_as_strong_as_pgd(model, data):
    """Identity purifier, no detector: the margin's sign steps are PGD's for two
    classes, and the best iterate can only add successes."""
    x, y = data
    torch.manual_seed(1)
    x_pgd = ProjectedGradientDescent(num_steps=10).generate(model, x, y, EPS)
    torch.manual_seed(1)
    x_adv = AdaptivePGD(steps=10, restarts=1).generate(Defense(model), x, y, EPS)
    with torch.no_grad():
        rob_pgd = (model.log_joint(x_pgd).argmax(1) == y).float().mean()
    right, _ = Defense(model).rates(x_adv, y)
    assert right.mean() <= rob_pgd
    assert right.mean() < 1.0   # the attack does something on this model


class _Scripted:
    """A defence whose error rate is 1 only on its ``hit``-th evaluation."""

    random = False

    def __init__(self, model, hit):
        self.model, self.hit, self.calls, self.seen = model, hit, 0, []

    def attack_step(self, x, labels, draws):
        self.calls += 1
        self.seen.append(x.detach().clone())
        rate = torch.full((len(x),), float(self.calls == self.hit))
        return x.sum(dim=1), rate


def test_best_iterate_is_kept(model, data):
    x, y = data
    scripted = _Scripted(model, hit=3)
    x_adv = AdaptivePGD(steps=6, restarts=1).generate(scripted, x, y, EPS)
    torch.testing.assert_close(x_adv, scripted.seen[2])


def test_points_stay_in_the_ball_and_the_box(model, data):
    x, y = data
    purifier = LikelihoodPurification(num_steps=3)
    defense = Defense(model, purify=lambda v: purifier.purify(model, v, 0.1)[0])
    x_adv = AdaptivePGD(steps=5, restarts=2).generate(defense, x, y, EPS)
    assert (x_adv - x).abs().max() <= EPS + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_only_accepted_wrong_answers_are_errors(model, data):
    x, y = data
    with torch.no_grad():
        lp = log_px(model, x)
    wrong = 1 - y
    # Rejecting everything: never right, never wrong.
    right, error = Defense(model, threshold=float(lp.max()) + 1).rates(x, wrong)
    assert right.sum() == 0 and error.sum() == 0
    # Accepting everything: the plain classifier.
    right, error = Defense(model, threshold=float(lp.min()) - 1).rates(x, wrong)
    assert right.sum() == 0 and error.sum() == len(x)


def test_detector_attack_beats_pgd_on_selective_accuracy(model, data):
    x, y = data
    with torch.no_grad():
        tau = float(log_px(model, x).median())
    defense = Defense(model, threshold=tau)
    torch.manual_seed(2)
    x_pgd = ProjectedGradientDescent(num_steps=20).generate(model, x, y, EPS)
    x_adv = AdaptivePGD(steps=20, restarts=3).generate(defense, x, y, EPS)

    def selective(x_attacked):
        right_clean, _ = defense.rates(x, y)
        _, error = defense.rates(x_attacked, y)
        return float((right_clean * (1 - error)).mean())

    assert selective(x_adv) <= selective(x_pgd)


def test_purifier_attack_beats_pgd_on_purified_accuracy(model, data):
    x, y = data
    purifier = LikelihoodPurification(num_steps=5)
    defense = Defense(model, purify=lambda v: purifier.purify(model, v, 0.1)[0])
    torch.manual_seed(3)
    x_pgd = ProjectedGradientDescent(num_steps=20).generate(model, x, y, EPS)
    x_adv = AdaptivePGD(steps=20, restarts=3).generate(defense, x, y, EPS)
    assert defense.rates(x_adv, y)[0].mean() <= defense.rates(x_pgd, y)[0].mean()


def test_random_defence_is_scored_on_fresh_draws(model, data):
    """EOT draws in the attack, then separate draws to score; a deterministic
    defence is evaluated once whatever ``draws`` says."""
    x, y = data
    calls = []

    def noisy(v):
        calls.append(1)
        return (v + 0.01 * torch.randn_like(v)).clamp(0, 1)

    defense = Defense(model, purify=noisy, random=True)
    AdaptivePGD(steps=2, restarts=1, eot=4).generate(defense, x, y, EPS)
    assert len(calls) == 3 * 4
    defense.rates(x, y, draws=5)
    assert len(calls) == 3 * 4 + 5

    calls.clear()
    Defense(model, purify=noisy).rates(x, y, draws=5)
    assert len(calls) == 1


def test_warm_start_is_never_worse_than_its_start(model, data):
    """D98: started from the PGD example, the adaptive attack errs at least wherever
    that example already makes the (deterministic) defence err."""
    x, y = data
    purifier = LikelihoodPurification(num_steps=5)
    defense = Defense(model, purify=lambda v: purifier.purify(model, v, 0.1)[0])
    torch.manual_seed(4)
    x_pgd = ProjectedGradientDescent(num_steps=20).generate(model, x, y, EPS)
    x_adv = AdaptivePGD(steps=3, restarts=1).generate(defense, x, y, EPS, start=x_pgd)
    _, error_pgd = defense.rates(x_pgd, y)
    _, error_adv = defense.rates(x_adv, y)
    assert (error_adv >= error_pgd).all()


def test_the_first_restart_begins_at_the_start(model, data):
    x, y = data
    scripted = _Scripted(model, hit=0)
    start = (x + 0.05).clamp(0, 1)
    AdaptivePGD(steps=2, restarts=2).generate(scripted, x, y, EPS, start=start)
    torch.testing.assert_close(scripted.seen[0], start)
    assert not torch.allclose(scripted.seen[3], start)   # the second restart is random
