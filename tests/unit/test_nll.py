import pytest
import torch
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

from src.train.nll import NLLConfig, NLLTrainer, NormControlConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig


# ── Helpers ────────────────────────────────────────────────────────────────

def _tiny_cbm():
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2)


class _FakeDataHandler:
    """Minimal DataHandler substitute that skips file I/O."""
    data_dim = 2

    def __init__(self, n=16, batch_size=4):
        ds = TensorDataset(torch.rand(n, 2), torch.randint(0, 2, (n,)))
        loader = DataLoader(ds, batch_size=batch_size)
        self.classification = {"train": loader, "valid": loader}

    def get_classification_loaders(self, batch_size=4):
        pass  # already set up


# ── Config defaults ────────────────────────────────────────────────────────

def test_nll_config_defaults():
    cfg = NLLConfig()
    assert cfg.alpha == 0.0
    assert cfg.stop_crit == "acc"
    assert cfg.patience == 250
    assert cfg.max_epoch == 100
    assert cfg.batch_size == 64
    assert cfg.save is False


def test_norm_control_config_defaults():
    nc = NormControlConfig()
    assert nc.target == 1.0
    assert nc.hard_every == 1
    assert nc.soft_strength == 0.0


def test_nll_config_invalid_stop_crit():
    cfg = NLLConfig(stop_crit="bad_metric")
    cbm = _tiny_cbm()
    dh = _FakeDataHandler()
    with pytest.raises(ValueError, match="Invalid stop_crit"):
        NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))


# ── Construction ───────────────────────────────────────────────────────────

def test_nll_trainer_constructs():
    cbm = _tiny_cbm()
    dh = _FakeDataHandler()
    cfg = NLLConfig()
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    assert trainer.norm_regularizer is None
    assert trainer._nc_target is None
    assert len(trainer.best_tensors) == len(cbm.tensors)


def test_nll_trainer_sets_up_classification_loaders():
    """If datahandler.classification is None, get_classification_loaders is called."""
    cbm = _tiny_cbm()
    dh = _FakeDataHandler()
    dh.classification = None
    called = []
    dh.get_classification_loaders = lambda batch_size: called.append(batch_size)
    NLLTrainer(cbm=cbm, train_cfg=NLLConfig(), datahandler=dh, device=torch.device("cpu"))
    assert called == [NLLConfig().batch_size]


# ── Alpha=0 fast path ──────────────────────────────────────────────────────

def test_alpha0_skips_log_partition_function_in_mixed_nll():
    """mixed_nll(alpha=0) must not call log_partition_function (norm control disabled)."""
    cbm = _tiny_cbm()
    dh = _FakeDataHandler()
    # Disable norm control so renormalize_() is never called
    cfg = NLLConfig(
        alpha=0.0, max_epoch=1,
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),
    )
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    trainer.cbm.prepare(device=torch.device("cpu"))
    trainer._nc_target = 1.0
    trainer.optimizer = torch.optim.Adam(cbm.parameters(), lr=1e-3)

    with patch.object(cbm, "log_partition_function", wraps=cbm.log_partition_function) as mock_logZ:
        trainer._train_epoch()

    mock_logZ.assert_not_called()
