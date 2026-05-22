import pytest
import torch
from pathlib import Path

from src.trainer.nll import NLLConfig, NLLTrainer, NormControlConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.data.handler import DataHandler
from src.data.gen_n_load import DatasetConfig, DataGenDowConfig

pytestmark = pytest.mark.slow

DATA_DIM = 2
NUM_CLASSES = 2


@pytest.fixture(scope="session")
def dh():
    cbm_ref = ConditionalBornMachine(
        cfg=CBMConfig(
            embedding="fourier",
            init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
        ),
        data_dim=DATA_DIM,
        num_classes=NUM_CLASSES,
    )
    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=64, seed=0, noise=0.1),
        overwrite=True,
    )
    handler = DataHandler(ds_cfg)
    handler.load()
    handler.split_and_rescale(cbm_ref)
    handler.get_classification_loaders(batch_size=8)
    return handler


@pytest.fixture
def cbm():
    return ConditionalBornMachine(
        cfg=CBMConfig(
            embedding="fourier",
            init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
        ),
        data_dim=DATA_DIM,
        num_classes=NUM_CLASSES,
    )


# ── Basic training ──────────────────────────────────────────────────────────

def test_nll_alpha0_runs(cbm, dh):
    cfg = NLLConfig(alpha=0.0, max_epoch=3, patience=999)
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))
    assert len(logged) == 3


def test_nll_alpha1_runs(cbm, dh):
    cfg = NLLConfig(alpha=1.0, max_epoch=3, patience=999)
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))
    assert len(logged) == 3


# ── Epoch dict keys ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_nll_epoch_dict_keys(cbm, dh, alpha):
    expected = {"nll/train", "dis_loss/valid", "gen_loss/valid", "acc/valid"}
    cfg = NLLConfig(alpha=alpha, max_epoch=2, patience=999)
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))
    for _, m in logged:
        assert expected <= m.keys(), f"missing keys: {expected - m.keys()}"


# ── Patience / early stopping ───────────────────────────────────────────────

def test_nll_early_stopping(cbm, dh):
    """With patience=0 and stop_crit='acc', stops after 1 epoch (counter > 0)."""
    cfg = NLLConfig(alpha=0.0, max_epoch=20, patience=0, stop_crit="acc")
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))
    assert len(logged) <= 3, f"Expected early stop, but ran {len(logged)} epochs"


# ── Save ────────────────────────────────────────────────────────────────────

def test_nll_save_creates_file(cbm, dh, tmp_path):
    cfg = NLLConfig(alpha=0.0, max_epoch=2, patience=999, save=True)
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    trainer.train(output_dir=tmp_path)
    assert (tmp_path / "model").exists(), "Expected 'model' checkpoint at output_dir/model"


# ── Best tensors restored ───────────────────────────────────────────────────

def test_nll_best_tensors_restored(cbm, dh):
    """After training, cbm.tensors should equal the best tensors seen."""
    cfg = NLLConfig(alpha=0.0, max_epoch=3, patience=999)
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    trainer.train()
    best = trainer.best_tensors
    current = [t.cpu() for t in cbm.tensors]
    for i, (b, c) in enumerate(zip(best, current)):
        assert torch.allclose(b, c, atol=1e-6), f"tensor {i} mismatch after restore"
