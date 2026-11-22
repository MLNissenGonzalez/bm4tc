import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from tests.conftest import DATA_DIM, NUM_CLASSES, raw_amplitudes
from bm4tc.core.attacks import (
    JointProjectedGradientDescent,
    ProjectedGradientDescent,
    EvasionConfig,
    build_attack,
)
from bm4tc.analysis.adaptive import AdaptivePGD, Defense
from bm4tc.core.embeddings import range_size_of, rel_to_abs

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
    from bm4tc.core.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
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
        amps_before = raw_amplitudes(cbm_attack, naturals)
        log_joint_before = 2 * torch.log(amps_before.abs().clamp(min=eps))
        mask = torch.zeros(BATCH, K, dtype=torch.bool)
        mask[torch.arange(BATCH), labels] = True
        max_wrong_before = log_joint_before.masked_fill(mask, float("-inf")).max(dim=-1).values.mean().item()

    adversarials = attacker.generate(cbm_attack, naturals, labels, eps_abs=STRENGTH)

    with torch.no_grad():
        amps_after = raw_amplitudes(cbm_attack, adversarials)
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


# ---- build_attack with JOINT_PGD ----

def test_build_attack_joint_pgd_runs(cbm_attack, attack_loader):
    """build_attack dispatches JOINT_PGD, which respects the absolute budget
    ``generate`` takes (the caller derives it via ``rel_to_abs``)."""
    eval_ = build_attack(EvasionConfig(method="JOINT_PGD", norm="inf", num_steps=5,
                                       random_start=False))
    eps_abs = rel_to_abs(STRENGTH_FRACTION, range_size_of(cbm_attack))

    naturals, labels = next(iter(attack_loader))
    adv = eval_.generate(cbm_attack, naturals, labels, eps_abs)

    assert adv.shape == naturals.shape
    assert (adv - naturals).abs().max().item() <= eps_abs + 1e-6


@pytest.fixture(scope="module")
def cbm_cosine():
    from bm4tc.core.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    torch.manual_seed(42)
    cfg = CBMConfig(
        embedding="cosine",
        init_kwargs=MPSInitConfig(in_dim=DATA_DIM, bond_dim=2, std=1.0),
    )
    model = ConditionalBornMachine(cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu")
    model.prepare(device="cpu")
    return model


def _pgd_examples(model, x, y, eps):
    return ProjectedGradientDescent(num_steps=5).generate(model, x, y, eps_abs=eps)


def _adaptive_examples(model, x, y, eps):
    return AdaptivePGD(steps=5, restarts=1).generate(Defense(model), x, y, eps)


@pytest.mark.parametrize("attack", [_pgd_examples, _adaptive_examples])
def test_no_pixel_stays_frozen_on_a_flat_end_of_the_range(cbm_cosine, labels, attack):
    """Every cosine feature has a zero slope at 0 and 1: a pixel exactly there gets a zero
    gradient and a sign step leaves it, which masked PGD on cosine by up to .3 of `rob`
    (D178). The attack must move every such pixel, within the ball and the domain."""
    torch.manual_seed(0)
    x = torch.randint(0, 2, (BATCH, DATA_DIM)).float()   # every pixel on an end
    x_adv = attack(cbm_cosine, x, labels, 0.1)
    assert (x_adv != x).all()
    assert (x_adv - x).abs().max() <= 0.1 + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_an_attack_given_the_input_range_stays_sparse_on_a_flat_end(cbm_cosine, labels):
    """The box is the attack's: in ``input_range`` the pixels a random start clamps onto
    a flat end stay there, the sparse attack D179 needs against a detector."""
    torch.manual_seed(0)
    x = torch.randint(0, 2, (BATCH, DATA_DIM)).float()
    attack = AdaptivePGD(steps=5, restarts=1, box=cbm_cosine.input_range)
    x_adv = attack.generate(Defense(cbm_cosine), x, labels, 0.1)
    unmoved = (x_adv == x).float().mean().item()
    assert 0.2 < unmoved < 0.8


def _detector_attack_boxes(model, labels, monkeypatch, scores):
    """Run `_attack` on a detector with `_score` scripted to ``scores``, one per craft
    of the adaptive attack: (the boxes attacked, rob_sel_adapt)."""
    from types import SimpleNamespace
    from bm4tc.pipeline import analyse as A

    boxes, scripted = [], iter(scores)

    class Recording(AdaptivePGD):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            boxes.append(kwargs["box"])

    def score(defense, x, x_adv, y, selective, draws):
        return next(scripted, torch.ones(len(x)))    # the PGD examples: ones

    monkeypatch.setattr(A, "AdaptivePGD", Recording)
    monkeypatch.setattr(A, "_score", score)
    x = torch.rand(BATCH, DATA_DIM)
    analysis = SimpleNamespace(
        adaptive=SimpleNamespace(eot=1, eval_draws=1), batch_size=BATCH, detector_masks=[],
        attack=SimpleNamespace(steps=1, restarts=1, subsample=None))
    data = SimpleNamespace(data={A.SPLIT: x, "train": x}, labels={A.SPLIT: labels})
    defences = [("rob_sel", "q5", Defense(model, threshold=-1e9), True, None)]
    out = A._attack(model, data, analysis, [0.1], torch.device("cpu"),
                    lambda *_: defences)
    return boxes, out[A.key("rob_sel_adapt", A.SPLIT, 0.1, "q5")]


def test_a_detector_on_a_flat_embedding_counts_the_lower_score_of_both_boxes(
        cbm_cosine, labels, monkeypatch):
    """Neither box dominates against a detector (D179): a point is lost if either loses it."""
    first = (torch.arange(BATCH) % 2 == 0).float()
    boxes, rob_sel = _detector_attack_boxes(cbm_cosine, labels, monkeypatch,
                                            [first, torch.ones(BATCH)])
    assert boxes == [tuple(cbm_cosine.attack_range), tuple(cbm_cosine.input_range)]
    assert rob_sel == pytest.approx(first.mean().item())
    boxes, rob_sel = _detector_attack_boxes(cbm_cosine, labels, monkeypatch,
                                            [first, 1 - first])
    assert rob_sel == 0.0


def test_a_detector_on_any_other_embedding_faces_one_attack(cbm_attack, labels, monkeypatch):
    first = (torch.arange(BATCH) % 2 == 0).float()
    boxes, rob_sel = _detector_attack_boxes(cbm_attack, labels, monkeypatch, [first])
    assert boxes == [tuple(cbm_attack.attack_range)]
    assert rob_sel == pytest.approx(first.mean().item())


# ---- the detector's attack confined to an image mask (D193) ----

@pytest.mark.parametrize("model", ["cbm_attack", "cbm_cosine"])
def test_a_masked_attack_moves_only_the_masked_in_features(model, labels, request):
    """Frozen features stay exactly at x, also in cosine's inset box (D178), which would
    otherwise move a frozen end pixel by the inset; the free ones move, in the ball."""
    model = request.getfixturevalue(model)
    torch.manual_seed(0)
    x = torch.randint(0, 2, (BATCH, DATA_DIM)).float()   # every pixel on an end
    free = torch.tensor([1.0, 1.0, 0.0, 0.0])
    attack = AdaptivePGD(steps=5, restarts=2, mask=lambda x: free)
    x_adv = attack.generate(Defense(model), x, labels, 0.1)
    assert torch.equal(x_adv[:, 2:], x[:, 2:])
    assert (x_adv[:, :2] != x[:, :2]).any()
    assert (x_adv - x).abs().max() <= 0.1 + 1e-6


def test_image_masks_window_data_and_digit():
    from bm4tc.analysis.adaptive import image_masks
    side = 6
    train = torch.zeros(4, side * side)
    train[:, 0] = 1.0                      # on in every image
    train[:1, 1] = 0.5                     # on in a quarter of them
    x = torch.zeros(2, side * side)
    x[0, 2 * side + 2] = 1.0               # one pixel at (2, 2)
    window, data_quarter, data_half, digit0, digit1 = image_masks(
        [{"window": 0.5}, {"data": 0.25}, {"data": 0.5}, {"digit": 0}, {"digit": 1}],
        train, (0.0, 1.0))
    expected = torch.zeros(side, side)
    expected[1:4, 1:4] = 1                 # the centred 3×3
    assert torch.equal(window(x), expected.flatten())
    assert data_quarter(x).nonzero().flatten().tolist() == [0, 1]
    assert data_half(x).nonzero().flatten().tolist() == [0]
    assert digit0(x)[0].nonzero().flatten().tolist() == [2 * side + 2]
    assert digit0(x)[1].sum() == 0         # a blank image frees nothing
    grown = torch.zeros(side, side)
    grown[1:4, 1:4] = 1
    assert torch.equal(digit1(x)[0], grown.flatten())


@pytest.mark.parametrize("spec", [{"window": 0.5, "data": 0.1}, {"frame": 1}, {}])
def test_a_mask_spec_is_one_known_key(spec):
    from bm4tc.analysis.adaptive import image_masks
    with pytest.raises(ValueError, match="mask spec"):
        image_masks([spec], torch.zeros(2, 16), (0.0, 1.0))


def test_image_masks_need_square_images():
    from bm4tc.analysis.adaptive import image_masks
    with pytest.raises(ValueError, match="square"):
        image_masks([{"digit": 0}], torch.zeros(2, 5), (0.0, 1.0))


def test_a_window_side_is_rounded_to_the_nearest_pixel():
    """{window: 0.67} frees 19×19 on 28×28 (0.67 · 28 = 18.76), the studies' mask (D194),
    and 8×8 on 12×12; truncating would give 18×18 without changing a hash."""
    from bm4tc.analysis.adaptive import image_masks
    for side, s in ((28, 19), (12, 8)):
        (window,) = image_masks([{"window": 0.67}], torch.zeros(2, side * side), (0.0, 1.0))
        expected = torch.zeros(side, side)
        a = (side - s) // 2
        expected[a:a + s, a:a + s] = 1
        assert torch.equal(window(torch.zeros(1, side * side)), expected.flatten())


def _detector_attack(model, labels, monkeypatch, masks, score=None, budgets=(0.1,),
                     restarts=1):
    """`_attack` on a detector with ``analysis.detector_masks = masks``: (the variants
    built, as (box, masked), and rob_sel_adapt per budget). ``score`` replaces `_score`."""
    from types import SimpleNamespace
    from bm4tc.pipeline import analyse as A

    variants = []

    class Recording(AdaptivePGD):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            variants.append((kwargs["box"], kwargs["mask"] is not None))

    monkeypatch.setattr(A, "AdaptivePGD", Recording)
    if score is not None:
        monkeypatch.setattr(A, "_score", score)
    torch.manual_seed(3)
    x = torch.rand(BATCH, DATA_DIM)
    analysis = SimpleNamespace(
        adaptive=SimpleNamespace(eot=1, eval_draws=1), batch_size=BATCH, detector_masks=masks,
        attack=SimpleNamespace(steps=2, restarts=restarts, subsample=None))
    data = SimpleNamespace(data={A.SPLIT: x, "train": x}, labels={A.SPLIT: labels})
    defences = [("rob_sel", "q5", Defense(model, threshold=-1e9), True, None)]
    torch.manual_seed(0)
    out = A._attack(model, data, analysis, list(budgets), torch.device("cpu"),
                    lambda *_: defences)
    return variants, [out[A.key("rob_sel_adapt", A.SPLIT, eps, "q5")] for eps in budgets]


def test_a_detector_faces_every_mask_after_the_unmasked_attack_and_counts_the_lowest(
        cbm_attack, labels, monkeypatch):
    first = (torch.arange(BATCH) % 2 == 0).float()
    scripted = iter([torch.ones(BATCH), first])

    def score(defense, x, x_adv, y, selective, draws):
        return next(scripted, torch.ones(len(x)))   # unmasked: ones; masked: first

    variants, (rob_sel,) = _detector_attack(cbm_attack, labels, monkeypatch,
                                            [{"window": 0.5}], score)
    box = tuple(cbm_attack.attack_range)
    assert variants == [(box, False), (box, True)]
    assert rob_sel == pytest.approx(first.mean().item())


def test_a_detector_on_a_flat_embedding_faces_every_mask_in_both_boxes(
        cbm_cosine, labels, monkeypatch):
    """On cosine each mask runs in both boxes (D179, D193), after the unmasked attack in
    both."""
    variants, _ = _detector_attack(cbm_cosine, labels, monkeypatch,
                                   [{"window": 0.5}, {"digit": 0}])
    inset, full = tuple(cbm_cosine.attack_range), tuple(cbm_cosine.input_range)
    assert inset != full
    assert variants == [(inset, False), (full, False),
                        (inset, True), (full, True), (inset, True), (full, True)]


def test_masks_leave_the_unmasked_attack_unchanged(cbm_attack, labels, monkeypatch):
    """The masked attacks draw from streams of their own, so adding masks changes no draw
    of the unmasked attack, at any budget (its random starts included): its examples, and
    the stored scores, are those of a run without masks (D193)."""
    from bm4tc.pipeline import analyse as A
    true_score = A._score

    def run(masks):
        seen = []

        def score(defense, x, x_adv, y, selective, draws):
            seen.append(x_adv.detach().clone())
            return true_score(defense, x, x_adv, y, selective, draws)
        _detector_attack(cbm_attack, labels, monkeypatch, masks, score,
                         budgets=(0.1, 0.2), restarts=4)
        monkeypatch.undo()
        return seen

    plain, masked = run([]), run([{"digit": 0}])
    # per budget: plain = [unmasked, PGD examples], masked = [unmasked, masked, PGD examples]
    assert len(plain) == 4 and len(masked) == 6
    for p, m in zip(plain, [masked[0], masked[2], masked[3], masked[5]]):
        assert torch.equal(p, m)
