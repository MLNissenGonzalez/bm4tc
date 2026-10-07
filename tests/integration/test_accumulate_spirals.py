"""accumulate on vs off on the spirals toy data (legendre d10r6, the spirals arch).

`accumulate` is on by default since Phase 4. On spirals nothing overflows, so the
overflow-safe contraction must give the same numbers as the raw amplitudes path:
evaluation, the loss and its gradients, and a short training run.
"""

import pytest
import torch

from experiments.metrics import flatten_epoch
from src.datahandler import DataHandler, DatasetConfig, DataGenDowConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.train.trainer import Trainer, TrainConfig
from src.utils.train import OptimizerConfig, evaluate

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")


def _cbm(accumulate, tensors=None):
    torch.manual_seed(0)
    cfg = CBMConfig(embedding="legendre", accumulate=accumulate,
                    init_kwargs=MPSInitConfig(in_dim=10, bond_dim=6))
    return ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2, tensors=tensors)


@pytest.fixture(scope="module")
def dh():
    handler = DataHandler(DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=200, seed=25, noise=0.5),
        split=(0.5, 0.25, 0.25), split_seed=11, overwrite=True,
    ))
    handler.load()
    handler.split_and_rescale(_cbm(True).input_range)
    handler.get_classification_loaders(batch_size=64)
    return handler


@pytest.fixture
def pair():
    """The same model twice: accumulate off and on."""
    off = _cbm(False)
    on = _cbm(True, tensors=[t.detach().clone() for t in off.tensors])
    for m in (off, on):
        m.prepare(device=CPU)
    return off, on


def test_evaluate_agrees(dh, pair):
    off, on = pair
    for split in ("valid", "test"):
        a = evaluate(off, dh.classification[split], CPU, alpha=0.5)
        b = evaluate(on, dh.classification[split], CPU, alpha=0.5)
        for k in ("objective", "loss_dis", "loss_gen", "acc"):
            assert b[k] == pytest.approx(a[k], rel=1e-5), (split, k)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_loss_and_gradients_agree(dh, pair, alpha):
    """Matched by name (the two models register parameters in different orders);
    a missing grad (the norm net's boundaries at alpha=0) counts as zero."""
    off, on = pair
    x, y = next(iter(dh.classification["train"]))
    losses, grads = [], []
    for m in (off, on):
        m.zero_grad()
        loss = m.mixed_nll(x, y, alpha=alpha)
        loss.backward()
        losses.append(loss.item())
        grads.append({n: p.grad if p.grad is not None else torch.zeros_like(p)
                      for n, p in m.named_parameters()})
    assert losses[1] == pytest.approx(losses[0], rel=1e-5)
    assert grads[0].keys() == grads[1].keys()
    for name, g_off in grads[0].items():
        scale = max(g_off.abs().max().item(), 1.0)
        assert (grads[1][name] - g_off).abs().max().item() <= 1e-5 * scale, name


def test_training_curves_agree(dh, pair):
    """Five NAT epochs at alpha=0.5 from the same tensors and seed.

    lr=1e-3 on purpose: the boundary nodes' gradients are ~1e-7, the size of the
    float noise between the two contractions, and Adam rescales every element to
    a step of about lr. At lr=1e-2 that noise alone moves objective/valid by ~0.2%
    after one epoch (and ~3% after eight); at 1e-3 it stays below ~3e-4.
    """
    curves = []
    for m in pair:
        torch.manual_seed(1)
        cfg = TrainConfig(alpha=0.5, max_epoch=5, batch_size=64,
                          optimizer=OptimizerConfig(kwargs={"lr": 1e-3}))
        logged = []
        Trainer(m, cfg, dh.classification["train"], dh.classification["valid"], CPU).train(
            on_epoch_end=lambda ep, r: logged.append(flatten_epoch(r)["objective/valid"]))
        curves.append(logged)
    assert curves[1] == pytest.approx(curves[0], rel=1e-3)
