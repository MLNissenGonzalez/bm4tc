"""End-to-end Trainer runs (NAT and AT) on a tiny real dataset, with a real
ConditionalBornMachine, real PGD and a real DataHandler; the unit tests stub them."""

import math

import pytest
import torch

from experiments.metrics import flatten_epoch, key
from src.datahandler import DataHandler, DatasetConfig, DataGenDowConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.train.trainer import Trainer, TrainConfig
from src.utils.evasion import EvasionConfig
from src.utils.train import NormControlConfig, OptimizerConfig

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")
PGD = EvasionConfig(method="PGD", num_steps=2, eps_rel=[0.05])


def _cbm():
    return ConditionalBornMachine(
        cfg=CBMConfig(
            embedding="fourier",
            init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
        ),
        data_dim=2,
        num_classes=2,
    )


@pytest.fixture(scope="module")
def dh():
    handler = DataHandler(DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=64, seed=0, noise=0.1),
        overwrite=True,
    ))
    handler.load()
    handler.split_and_rescale(_cbm().input_range)
    handler.get_classification_loaders(batch_size=8)
    return handler


def _loaders(dh):
    return dh.classification["train"], dh.classification["valid"]


def _train(dh, cbm=None, **cfg):
    cbm = cbm or _cbm()
    trainer = Trainer(cbm, TrainConfig(batch_size=8, patience=999, **cfg), *_loaders(dh), CPU)
    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, flatten_epoch(m))))
    return trainer, logged


# ── NAT ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_nat_logs_every_epoch(dh, alpha):
    expected = {"objective/train", "penalty/train", "objective/valid",
                "loss_dis/valid", "loss_gen/valid", "acc/valid"}
    _, logged = _train(dh, alpha=alpha, max_epoch=2)
    assert [ep for ep, _ in logged] == [1, 2]
    for _, m in logged:
        assert expected <= m.keys(), f"missing keys: {expected - m.keys()}"


def test_early_stopping(dh):
    """patience=0 stops at the first validation without a lower objective.

    lr=0 freezes the model: epoch 1 improves on inf, epoch 2 ties, and a tie is
    not an improvement, so training ends after epoch 2.
    """
    trainer = Trainer(_cbm(), TrainConfig(
        alpha=0.0, max_epoch=20, batch_size=8, patience=0,
        optimizer=OptimizerConfig(kwargs={"lr": 0.0}),
        norm_control=NormControlConfig(soft_strength=0.0),
    ), *_loaders(dh), CPU)
    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append(ep))
    assert logged == [1, 2]


def test_save_creates_file(dh, tmp_path):
    trainer = Trainer(_cbm(), TrainConfig(max_epoch=2, batch_size=8, save=True),
                      *_loaders(dh), CPU)
    trainer.train(output_dir=tmp_path)
    assert (tmp_path / "model").exists()


def test_best_tensors_restored(dh):
    cbm = _cbm()
    trainer, _ = _train(dh, cbm, max_epoch=3)
    for i, (b, c) in enumerate(zip(trainer.best_tensors, cbm.tensors)):
        assert torch.allclose(b, c.cpu(), atol=1e-6), f"tensor {i} mismatch after restore"


# ── AT ──────────────────────────────────────────────────────────────────────

def test_at_run_completes_and_selects_a_model(dh):
    """AT trains end to end, validates every eval_every epochs and restores the best."""
    cbm = _cbm()
    trainer, logged = _train(dh, cbm, alpha=0.5, evasion=PGD, clean_weight=0.3,
                             max_epoch=6, eval_every=2)

    assert [ep for ep, _ in logged] == [1, 2, 3, 4, 5, 6]
    assert [ep for ep, m in logged if "objective/valid" in m] == [2, 4, 6]
    rob = key("rob", "valid", trainer.eps_rel)
    assert [ep for ep, m in logged if rob in m] == [2, 4, 6]

    n_valid = len(dh.classification["valid"].dataset)
    assert len(trainer.adv_indices) == round(0.7 * n_valid)
    assert all(m["n_rob/valid"] == len(trainer.adv_indices)
               for _, m in logged if "n_rob/valid" in m)

    assert math.isfinite(trainer.best["objective"])
    assert trainer.best_epoch in (2, 4, 6)
    assert trainer.best["rob"] > 0.0
    assert all(torch.isfinite(t).all() for t in cbm.tensors)
