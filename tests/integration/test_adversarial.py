"""End-to-end AdversarialTrainer runs on a tiny real dataset.

Covers the `gen_on_clean` split objective against a real ConditionalBornMachine,
real PGD attacks and a real DataHandler — the unit tests stub all three.
"""

import pytest
import torch

from src.train.adversarial import AdversarialConfig, AdversarialTrainer
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.datahandler import DataHandler, DatasetConfig, DataGenDowConfig
from src.utils.evasion import EvasionConfig
from src.utils.train import NormControlConfig

pytestmark = pytest.mark.slow

DATA_DIM = 2
NUM_CLASSES = 2


def _cbm():
    return ConditionalBornMachine(
        cfg=CBMConfig(
            embedding="fourier",
            init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
        ),
        data_dim=DATA_DIM,
        num_classes=NUM_CLASSES,
    )


@pytest.fixture(scope="module")
def dh():
    handler = DataHandler(DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=64, seed=0, noise=0.1),
        overwrite=True,
    ))
    handler.load()
    handler.split_and_rescale(_cbm())
    handler.get_classification_loaders(batch_size=8)
    return handler


def _cfg(**overrides):
    base = dict(
        max_epoch=6,
        batch_size=8,
        alpha=0.5,
        evasion=EvasionConfig(method="PGD", num_steps=2, strengths=[0.05]),
        stop_crit="rob",
        eval_rob_freq=2,
        patience=250,
        clean_weight=0.3,
        gen_on_clean=True,
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.1, log_target=0.0),
        save=False,
    )
    base.update(overrides)
    return AdversarialConfig(**base)


def test_split_run_completes_and_selects_a_model(dh):
    """gen_on_clean=True trains end to end and restores a selected checkpoint."""
    cbm = _cbm()
    trainer = AdversarialTrainer(cbm=cbm, train_cfg=_cfg(), datahandler=dh,
                                 device=torch.device("cpu"))

    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))

    assert [ep for ep, _ in logged] == [1, 2, 3, 4, 5, 6]
    # Validation only on eval_rob_freq epochs, nothing in between.
    assert [ep for ep, m in logged if "acc/valid" in m] == [2, 4, 6]

    n_valid = len(dh.classification["valid"].dataset)
    assert len(trainer.adv_indices) == round(0.7 * n_valid)
    for _, m in logged:
        if "n_rob_valid" in m:
            assert m["n_rob_valid"] == len(trainer.adv_indices)

    assert trainer.best["rob"] > 0.0
    assert all(torch.isfinite(t).all() for t in cbm.tensors)


def test_split_and_unsplit_agree_at_alpha_zero(dh):
    """At alpha=0 the split is a no-op: same loss, bit-identical model after an epoch.

    Compared over a single epoch on purpose. The two modes run different
    validation code (eval_split over a (1-cw) subset vs. eval_metrics + eval_rob
    over the full set), which consumes different amounts of global RNG and so
    reshuffles the *next* epoch's training batches differently — a divergence
    that says nothing about the objective. random_start=False likewise keeps the
    attack itself deterministic.
    """
    evasion = EvasionConfig(method="PGD", num_steps=2, strengths=[0.05],
                            random_start=False)
    runs = {}
    for gen_on_clean in (False, True):
        torch.manual_seed(0)
        cbm = _cbm()
        cfg = _cfg(alpha=0.0, gen_on_clean=gen_on_clean, max_epoch=1,
                   eval_rob_freq=1, evasion=evasion)
        trainer = AdversarialTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh,
                                     device=torch.device("cpu"))
        seen = []
        trainer.train(on_epoch_end=lambda ep, m: seen.append(m["dis_loss/train"]))
        runs[gen_on_clean] = (seen, [t.detach().clone() for t in cbm.tensors])

    assert runs[False][0] == pytest.approx(runs[True][0], abs=1e-9)
    for unsplit, split in zip(runs[False][1], runs[True][1]):
        assert torch.equal(unsplit, split)


def test_split_rejects_zero_eval_rob_freq(dh):
    """eval_rob_freq is the whole validation cadence in split mode."""
    with pytest.raises(ValueError, match="eval_rob_freq >= 1"):
        AdversarialTrainer(cbm=_cbm(), train_cfg=_cfg(eval_rob_freq=0),
                           datahandler=dh, device=torch.device("cpu"))
