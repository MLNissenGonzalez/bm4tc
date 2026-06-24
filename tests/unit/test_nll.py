import math
import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from src.train.nll import NLLConfig, NLLTrainer, NormControlConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.utils.train import NormRegularizer


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
    assert nc.log_target == 0.0
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
    assert trainer._nc_log_target is None
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

# ── NormRegularizer ────────────────────────────────────────────────────────

def test_norm_regularizer_zero_at_target():
    log_target = math.log(2.5)
    reg = NormRegularizer(strength=1.0, log_target=log_target)
    cbm = MagicMock()
    cbm.log_Z.return_value = torch.tensor(log_target)
    penalty = reg(cbm)
    cbm.log_Z.assert_called_once_with(recompute=False)
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_norm_regularizer_nonzero_off_target():
    log_target = 0.0  # Z=1
    strength = 3.0
    reg = NormRegularizer(strength=strength, log_target=log_target)
    log_Z_val = 2.0  # delta = 2.0
    cbm = MagicMock()
    cbm.log_Z.return_value = torch.tensor(log_Z_val)
    penalty = reg(cbm)
    expected = strength * (log_Z_val - log_target) ** 2
    assert penalty.item() == pytest.approx(expected, rel=1e-5)


def test_norm_regularizer_invalid_target():
    with pytest.raises(ValueError, match="log_target must be finite"):
        NormRegularizer(strength=1.0, log_target=float("inf"))


# ── _diagnostics cache routing ─────────────────────────────────────────────

def _diag_trainer():
    cbm = _tiny_cbm()
    trainer = NLLTrainer(cbm=cbm, train_cfg=NLLConfig(alpha=1.0),
                         datahandler=_FakeDataHandler(), device=torch.device("cpu"))
    return trainer, cbm


def test_diagnostics_uses_caches_without_recontracting():
    """When mixed_nll has populated the caches, _diagnostics reads them and does
    not contract the norm again."""
    trainer, cbm = _diag_trainer()
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    cbm.mixed_nll(x, y, alpha=1.0)            # populates both caches
    with patch.object(cbm, "log_partition_function",
                      wraps=cbm.log_partition_function) as mock_logZ:
        diag = trainer._diagnostics(x)
    mock_logZ.assert_not_called()
    assert math.isfinite(diag["log_Z"])
    assert diag["log_Z"] == pytest.approx(cbm._log_Z_cache.detach().item())
    assert {"log_amp_sq_mean", "log_amp_sq_min", "log_amp_sq_max",
            "amp_nonfinite_count"} <= set(diag)


def test_diagnostics_falls_back_when_cache_empty():
    """No cached forward (fresh model) → _diagnostics recomputes and still works."""
    trainer, cbm = _diag_trainer()
    assert cbm._log_Z_cache is None and cbm._amp_diag_cache is None
    diag = trainer._diagnostics(torch.rand(4, 2))
    assert math.isfinite(diag["log_Z"])
    assert math.isfinite(diag["log_amp_sq_mean"])


# ── _format_diagnostics tagging ────────────────────────────────────────────

def test_amp_nonfinite_count_always_tagged_overflow():
    """A non-finite amplitude count is always overflow: 2·log(|amp|.clamp(min=tiny))
    floors underflow, so the count can never be an underflow — including when the
    mean log|amp|² is small (the old `mean_ > 80 else underflow` mislabel)."""
    s = NLLTrainer._format_diagnostics({
        "log_Z": 0.0,
        "log_amp_sq_mean": 1.0, "log_amp_sq_min": 0.0, "log_amp_sq_max": 2.0,
        "amp_nonfinite_count": 3,
    })
    assert "3 non-finite → overflow" in s
    assert "underflow" not in s


def test_format_diagnostics_surfaces_overflow_headroom():
    """A finite log_Z with a headroom value is rendered with the remaining
    overflow headroom; absent the key, the plain log_Z is shown."""
    with_headroom = NLLTrainer._format_diagnostics({"log_Z": 100.0, "log_Z_headroom": 77.45})
    assert "overflow headroom=77.45" in with_headroom

    without = NLLTrainer._format_diagnostics({"log_Z": 100.0})
    assert "headroom" not in without


def test_log_Z_overflow_ceiling_matches_float32():
    """The float32/complex64 overflow ceiling is 2·log(finfo.max) ≈ 177.45."""
    ceiling = 2.0 * math.log(torch.finfo(torch.float32).max)
    assert ceiling == pytest.approx(177.45, abs=0.1)


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


def test_alpha0_soft_norm_control_multistep_backward():
    """Regression: alpha=0 + soft norm control must train across multiple steps.

    The with-grad log_Z cache is only refreshed by mixed_nll when alpha>0; for
    alpha=0 the NormRegularizer reads it via recompute=False. Without per-step
    cache invalidation, the second optimizer step backwards through the freed
    graph of the first step ("Trying to backward through the graph a second
    time"). Guards the _invalidate_log_Z_cache() call in _train_epoch.
    """
    cbm = _tiny_cbm()
    dh = _FakeDataHandler(n=16, batch_size=4)  # 4 batches → 4 steps (>=2 needed)
    cfg = NLLConfig(
        alpha=0.0, max_epoch=1,
        norm_control=NormControlConfig(hard_every=0, soft_strength=1.0, log_target=0.0),
    )
    trainer = NLLTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh, device=torch.device("cpu"))
    trainer.cbm.prepare(device=torch.device("cpu"))
    trainer.step = 0
    trainer._nc_log_target = 0.0
    trainer.norm_regularizer = NormRegularizer(strength=1.0, log_target=0.0)
    trainer.optimizer = torch.optim.Adam(cbm.parameters(), lr=1e-3)

    # Must complete without "Trying to backward through the graph a second time".
    trainer._train_epoch()

    assert not trainer._collapsed
    assert trainer.step >= 2
    assert cbm._log_Z_cache is None  # invalidated after the final step
