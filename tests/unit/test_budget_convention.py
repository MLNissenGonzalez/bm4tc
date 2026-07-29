"""Pins the rel/abs budget convention (see "Budget vocabulary" in CLAUDE.md).

Every authored budget is a *fraction* of the input domain width; absolute
model-domain values are derived at one boundary per entry point. These tests exist
because the two conventions previously coexisted with no naming marker, and the
embedding-dependent cases were only ever correct by accident on legendre.
"""

import pytest
import torch

from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.train.adversarial import AdversarialTrainer, AdversarialConfig
from src.utils.evasion import EvasionConfig, ProjectedGradientDescent
from src.utils.embeddings import fmt_budget, range_size_of, rel_to_abs
from src.analysis.purification import LikelihoodPurification, GibbsPurification

DATA_DIM = 3
NUM_CLASSES = 2
EPS_REL = 0.1


def _cbm(embedding: str) -> ConditionalBornMachine:
    torch.manual_seed(0)
    cfg = CBMConfig(
        embedding=embedding,
        init_kwargs=MPSInitConfig(in_dim=4, bond_dim=2, std=1.0),
    )
    m = ConditionalBornMachine(
        cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu"
    )
    m.prepare(device="cpu")
    return m


class _FakeDataHandler:
    data_dim = DATA_DIM

    def __init__(self, n=8, batch_size=4):
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.rand(n, DATA_DIM), torch.randint(0, NUM_CLASSES, (n,)))
        loader = DataLoader(ds, batch_size=batch_size)
        self.classification = {"train": loader, "valid": loader}

    def get_classification_loaders(self, batch_size=4):
        pass


# ---- the conversion boundary ----

@pytest.mark.parametrize(
    "embedding, range_size, expected_abs",
    [("legendre", 2.0, 0.2), ("fourier", 1.0, 0.1)],
)
def test_budget_conversion_is_embedding_aware(embedding, range_size, expected_abs):
    """The same authored eps_rel resolves to different absolute budgets per embedding.

    This is the whole point of authoring relative: legendre spans [-1,1] (width 2)
    and fourier [0,1] (width 1), so a bare absolute number means different things.
    """
    cbm = _cbm(embedding)
    assert range_size_of(cbm) == pytest.approx(range_size)
    assert rel_to_abs(EPS_REL, range_size_of(cbm)) == pytest.approx(expected_abs)


@pytest.mark.parametrize(
    "embedding, expected_abs", [("legendre", 0.2), ("fourier", 0.1)]
)
def test_trainer_resolves_eps_rel_per_embedding(embedding, expected_abs):
    """AdversarialTrainer converts the config's eps_rel once, using its own model."""
    cbm = _cbm(embedding)
    cfg = AdversarialConfig(evasion=EvasionConfig(method="PGD", eps_rel=[EPS_REL]))
    t = AdversarialTrainer(
        cbm=cbm, train_cfg=cfg, datahandler=_FakeDataHandler(),
        device=torch.device("cpu"),
    )

    assert t.base_eps_rel == pytest.approx(EPS_REL)
    assert t.base_eps_abs == pytest.approx(expected_abs)
    # The metric key states the relative budget, not the absolute one.
    assert t.rob_metric_key == f"rob/valid/{fmt_budget(EPS_REL)}"


def test_curriculum_start_is_relative_too():
    """curriculum_eps_start_rel is a fraction and ramps to base_eps_abs."""
    cbm = _cbm("legendre")
    cfg = AdversarialConfig(
        evasion=EvasionConfig(method="PGD", eps_rel=[EPS_REL]),
        curriculum=True, curriculum_eps_start_rel=0.01, curriculum_end_epoch=10,
    )
    t = AdversarialTrainer(
        cbm=cbm, train_cfg=cfg, datahandler=_FakeDataHandler(),
        device=torch.device("cpu"),
    )

    assert t._curriculum_eps_start_abs == pytest.approx(0.02)  # 0.01 * 2.0
    assert t._get_eps_abs(0) == pytest.approx(0.02)
    assert t._get_eps_abs(10) == pytest.approx(t.base_eps_abs)


# ---- the budget is actually respected downstream ----

def test_attack_respects_absolute_budget():
    """PGD's perturbation stays inside the eps_abs ball derived from eps_rel."""
    cbm = _cbm("legendre")
    eps_abs = rel_to_abs(EPS_REL, range_size_of(cbm))  # 0.2

    naturals = torch.zeros(8, DATA_DIM)
    labels = torch.randint(0, NUM_CLASSES, (8,))
    attack = ProjectedGradientDescent(norm="inf", num_steps=5, random_start=True)
    adv = attack.generate(cbm, naturals, labels, eps_abs=eps_abs)

    assert (adv - naturals).abs().max().item() <= eps_abs + 1e-6


def test_likelihood_purification_respects_delta_abs():
    """Purification moves no coordinate further than delta_abs."""
    cbm = _cbm("legendre")
    cbm.eval()
    cbm.cache_log_Z()
    delta_abs = rel_to_abs(0.05, range_size_of(cbm))  # 0.1

    x = torch.zeros(4, DATA_DIM)
    purified, _ = LikelihoodPurification(norm="inf", num_steps=5).purify(
        cbm, x, delta_abs=delta_abs, device=torch.device("cpu")
    )

    assert (purified - x).abs().max().item() <= delta_abs + 1e-6


def test_gibbs_step_is_per_sweep_not_a_global_budget():
    """After k sweeps the envelope is k * step_delta_rel * range_size, not one ball.

    Guards the documented attack-radius-agnostic semantics: the window re-centres
    every sweep, so strength comes from n_sweeps alone.
    """
    cbm = _cbm("legendre")
    cbm.eval()
    cbm.cache_log_Z()
    step_rel = 0.05
    step_abs = rel_to_abs(step_rel, range_size_of(cbm))  # 0.1

    x = torch.zeros(4, DATA_DIM)
    purifier = GibbsPurification(
        num_bins=8, gibbs_batch_size=4, step_delta_rel=step_rel
    )
    snaps = purifier.purify_snapshots(cbm, x, [1, 3], torch.device("cpu"))

    for k, (x_pur, _) in snaps.items():
        moved = (x_pur - x).abs().max().item()
        assert moved <= k * step_abs + 1e-6, f"{k} sweeps exceeded k*step envelope"


# ---- metric-key formatting ----

@pytest.mark.parametrize(
    "value, expected",
    [(0.05, "0.05"), (0.1, "0.1"), (0.15, "0.15"), (0.3, "0.3"), (1.0, "1")],
)
def test_fmt_budget_is_stable(value, expected):
    """Keys must not carry float-repr noise, which would break column joins."""
    assert fmt_budget(value) == expected


def test_fmt_budget_survives_arithmetic():
    """A budget that went through a multiply still formats to a clean key."""
    assert fmt_budget(rel_to_abs(0.15, 2.0)) == "0.3"
    assert fmt_budget(0.1 + 0.05) == "0.15"
