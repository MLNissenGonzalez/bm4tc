import pytest
import torch
import numpy as np
from src.analysis.uq import (
    UQConfig,
    UQResults,
    UQEvaluation,
    PurificationMetrics,
    compute_log_px,
    compute_thresholds,
    _batched_forward,
)

pytestmark = pytest.mark.slow


# ---- Unit-level dataclass tests (no model needed) ----

def test_uq_config_defaults():
    cfg = UQConfig()
    assert cfg.norm == "inf"
    assert cfg.num_steps == 20
    assert isinstance(cfg.radii, list)
    assert isinstance(cfg.percentiles, list)


def test_uq_config_gibbs_fields():
    cfg = UQConfig()
    assert cfg.run_gibbs is False
    assert isinstance(cfg.gibbs_n_sweeps, list)
    assert cfg.gibbs_num_bins > 0
    assert cfg.gibbs_batch_size > 0


def test_purification_metrics_construct():
    m = PurificationMetrics(
        accuracy_after_purify=0.8,
        recovery_rate=0.5,
        mean_log_px_before=-2.0,
        mean_log_px_after=-1.0,
        rejection_rate=0.1,
    )
    assert m.accuracy_after_purify == pytest.approx(0.8)


def test_uq_results_gibbs_default_empty():
    results = UQResults(
        clean_log_px=np.array([-1.0, -2.0]),
        clean_accuracy=0.9,
        thresholds={5: -3.0},
        adv_log_px={},
        adv_accuracies={},
        detection_rates={},
        purification_results={},
    )
    assert results.gibbs_purification_results == {}


def test_uq_results_new_det_fields_default_empty():
    results = UQResults(
        clean_log_px=np.array([-1.0]),
        clean_accuracy=0.9,
        thresholds={},
        adv_log_px={},
        adv_accuracies={},
        detection_rates={},
        purification_results={},
    )
    assert results.err_rate_detected == {}
    assert results.err_rate_passed == {}


def test_uq_results_summary_returns_string():
    results = UQResults(
        clean_log_px=np.array([-1.0, -2.0]),
        clean_accuracy=0.9,
        thresholds={5: -3.0},
        adv_log_px={0.1: np.array([-5.0, -6.0])},
        adv_accuracies={0.1: 0.5},
        detection_rates={(5, 0.1): 0.8},
        purification_results={},
    )
    summary = results.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


# ---- Integration tests (need cbm + clean_loader) ----

def test_compute_log_px_shape(cbm, clean_loader):
    log_px, labels = compute_log_px(cbm, clean_loader, device="cpu")
    assert log_px.shape == (32,)
    assert labels.shape == (32,)


def test_compute_log_px_finite(cbm, clean_loader):
    log_px, _ = compute_log_px(cbm, clean_loader, device="cpu")
    assert torch.isfinite(log_px).all()


def test_compute_thresholds_all_percentiles_present(cbm, clean_loader):
    percentiles = [1, 5, 10, 20]
    thresholds, _ = compute_thresholds(cbm, clean_loader, percentiles, device="cpu")
    for p in percentiles:
        assert p in thresholds


def test_compute_thresholds_ordered(cbm, clean_loader):
    percentiles = [5, 20]
    thresholds, _ = compute_thresholds(cbm, clean_loader, percentiles, device="cpu")
    assert thresholds[5] <= thresholds[20]


def test_uq_evaluate_completes(cbm, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(cbm, clean_loader, device="cpu")
    assert isinstance(results, UQResults)


def test_uq_clean_accuracy_range(cbm, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(cbm, clean_loader, device="cpu")
    assert 0.0 <= results.clean_accuracy <= 1.0


def test_uq_detection_rates_range(cbm, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(cbm, clean_loader, device="cpu")
    for rate in results.detection_rates.values():
        assert 0.0 <= rate <= 1.0


def test_uq_purification_results_present(cbm, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(cbm, clean_loader, device="cpu")
    assert len(results.purification_results) > 0


def test_uq_purification_acc_range(cbm, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(cbm, clean_loader, device="cpu")
    for metrics in results.purification_results.values():
        assert 0.0 <= metrics.accuracy_after_purify <= 1.0


def test_uq_gibbs_empty_when_disabled(cbm, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
        run_gibbs=False,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(cbm, clean_loader, device="cpu")
    assert results.gibbs_purification_results == {}


def test_uq_err_rate_detected_range(cbm, clean_loader):
    import math
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    results = UQEvaluation(cfg).evaluate(cbm, clean_loader, device="cpu")
    assert len(results.err_rate_detected) > 0
    for v in results.err_rate_detected.values():
        assert math.isnan(v) or 0.0 <= v <= 1.0


def test_uq_err_rate_passed_range(cbm, clean_loader):
    import math
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    results = UQEvaluation(cfg).evaluate(cbm, clean_loader, device="cpu")
    assert len(results.err_rate_passed) > 0
    for v in results.err_rate_passed.values():
        assert math.isnan(v) or 0.0 <= v <= 1.0


# ---- Memory-control: chunked forwards + fault isolation ----

def test_uq_config_eval_batch_size_default():
    assert UQConfig().eval_batch_size is None


@pytest.mark.parametrize("bs", [1, 7, 20, 100, None])
def test_batched_forward_matches_single_pass(cbm, bs):
    # Chunked forward must equal a single full-batch forward, for batch sizes that
    # divide N (20), don't divide it (7), equal it (20), exceed it (100), or None.
    x = torch.rand(20, cbm._data_dim)
    ref = cbm.class_probabilities(x).detach().cpu()
    out = _batched_forward(cbm.class_probabilities, x, bs, "cpu")
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5)


def test_uq_eval_batch_size_completes(cbm, clean_loader):
    # Re-batching to a small eval_batch_size must not change that evaluation runs.
    cfg = UQConfig(
        attack_strengths=[0.1], radii=[0.1], percentiles=[10],
        attack_num_steps=2, num_steps=2, eval_batch_size=4,
    )
    results = UQEvaluation(cfg).evaluate(cbm, clean_loader, device="cpu")
    assert isinstance(results, UQResults)
    assert len(results.detection_rates) > 0
    assert len(results.purification_results) > 0


def test_uq_fault_isolation_gibbs_failure(cbm, clean_loader, monkeypatch):
    # A failure (e.g. OOM) inside Gibbs purification must not discard the
    # detection and gradient-purification results computed earlier.
    from src.analysis import purification as purif_mod

    def boom(self, *args, **kwargs):
        raise RuntimeError("simulated OOM")

    monkeypatch.setattr(purif_mod.GibbsPurification, "purify_snapshots", boom)
    cfg = UQConfig(
        attack_strengths=[0.1], radii=[0.1], percentiles=[10],
        attack_num_steps=2, num_steps=2, run_gibbs=True, gibbs_n_sweeps=[1],
    )
    results = UQEvaluation(cfg).evaluate(cbm, clean_loader, device="cpu")
    assert len(results.detection_rates) > 0          # detection survived
    assert len(results.purification_results) > 0     # gradient purify survived
    assert results.gibbs_purification_results == {}  # gibbs skipped, not fatal


def test_uq_gibbs_subsample_runs(cbm, clean_loader):
    # Gibbs runs on a fixed subsample (< n_adv) while cheap metrics keep the full set;
    # snapshot sweeps [1,2] both produce valid metrics in a single max-sweep pass.
    import math
    cfg = UQConfig(
        attack_strengths=[0.1], radii=[0.1], percentiles=[10],
        attack_num_steps=2, num_steps=2,
        run_gibbs=True, gibbs_n_sweeps=[1, 2], gibbs_num_bins=8,
        gibbs_batch_size=3,          # forces multiple Gibbs batches on the subsample
        gibbs_subsample=8, gibbs_subsample_seed=0,
    )
    results = UQEvaluation(cfg).evaluate(cbm, clean_loader, device="cpu")
    assert {(0.1, 1), (0.1, 2)} <= set(results.gibbs_purification_results.keys())
    for m in results.gibbs_purification_results.values():
        assert 0.0 <= m.accuracy_after_purify <= 1.0
        assert math.isfinite(m.mean_log_px_after)
    assert set(results.clean_gibbs_purification_results.keys()) == {1, 2}


def test_uq_fault_isolation_one_eps_failure(cbm, clean_loader, monkeypatch):
    # If the attack raises for one eps, the other eps must still produce results.
    from src.utils import evasion as evasion_mod
    real_generate = evasion_mod.RobustnessEvaluation.generate

    def selective(self, born, data, labels, eps, device, *a, **k):
        if eps == 0.2:
            raise RuntimeError("simulated OOM")
        return real_generate(self, born, data, labels, eps, device, *a, **k)

    monkeypatch.setattr(evasion_mod.RobustnessEvaluation, "generate", selective)
    cfg = UQConfig(
        attack_strengths=[0.1, 0.2], radii=[0.1], percentiles=[10],
        attack_num_steps=2, num_steps=2,
    )
    results = UQEvaluation(cfg).evaluate(cbm, clean_loader, device="cpu")
    assert 0.1 in results.adv_accuracies        # good eps survived
    assert 0.2 not in results.adv_accuracies     # failed eps skipped
