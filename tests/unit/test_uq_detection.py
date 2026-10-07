"""Behavioural tests for the likelihood-detection arithmetic.

These exist because the detector once reported exactly 0.0000 across 96 cells of a
committed sweep and nothing failed: the only assertion in the suite was
``0.0 <= rate <= 1.0``, which a constant-zero, constant-one, or sign-inverted detector
all satisfy. Every test here is written to fail for at least one of those mutations.

The key fact these pin: when the threshold is calibrated on the same scores that are
evaluated, a detector with no signal reports **exactly** ``percentile / 100``, not 0.
So ``chance`` -- not zero -- is the null hypothesis.
"""
import math

import numpy as np
import pytest

from bm4tc.analysis.uq import detection_metrics

PCTS = [1, 5, 10, 20]
N = 1000


def _clean_scores(seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).normal(size=N)


def _thresholds(clean: np.ndarray) -> dict:
    return {p: float(np.percentile(clean, p)) for p in PCTS}


def test_chance_floor():
    """No signal must read as the chance line, not as zero."""
    clean = _clean_scores()
    out = detection_metrics(clean.copy(), np.zeros(N, bool), _thresholds(clean))
    for p in PCTS:
        assert out[p].detection_rate == pytest.approx(p / 100, abs=2 / N)
        assert out[p].chance == p / 100
        assert out[p].lift == pytest.approx(0.0, abs=2 / N)


def test_shift_down_detects_all():
    clean = _clean_scores()
    out = detection_metrics(clean - 10.0, np.zeros(N, bool), _thresholds(clean))
    for p in PCTS:
        assert out[p].detection_rate == 1.0
        assert out[p].lift > 0.5


def test_shift_up_detects_none():
    """Regression-pins the spirals-AT signature: det 0.0 with err_detected nan.

    See (tag pre-ousterhout) analysis/outputs/spirals/at/legendre/d10r6/seed_sweep_0206 -- detection was
    0.0000 in all 96 cells while err_detected was nan in all 96, i.e. the flagged set
    was empty because every adversarial score sat ABOVE the threshold.
    """
    clean = _clean_scores()
    out = detection_metrics(clean + 10.0, np.zeros(N, bool), _thresholds(clean))
    for p in PCTS:
        assert out[p].detection_rate == 0.0
        assert math.isnan(out[p].err_rate_detected)
        assert not math.isnan(out[p].err_rate_passed)
        assert out[p].lift < 0.0


def test_monotone_in_threshold():
    """The inversion guard: a flipped comparison reverses this ordering."""
    clean = _clean_scores(1)
    adv = _clean_scores(2) - 0.5
    out = detection_metrics(adv, np.zeros(N, bool), _thresholds(clean))
    rates = [out[p].detection_rate for p in PCTS]
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_lift_positive_for_real_signal():
    clean = _clean_scores(3)
    adv = _clean_scores(4) - 2.0
    out = detection_metrics(adv, np.zeros(N, bool), _thresholds(clean))
    for p in PCTS:
        assert out[p].detection_rate > out[p].chance
        assert out[p].lift > 0.0


def test_constant_scores_flagged_degenerate():
    """A constant score yields 0.0 by the strict-< convention, not by signal."""
    clean = _clean_scores(5)
    out = detection_metrics(np.full(N, 3.0), np.zeros(N, bool), _thresholds(clean))
    for p in PCTS:
        assert out[p].degenerate is True
        assert out[p].detection_rate == 0.0


def test_nonfinite_scores_flagged():
    """Non-finite scores must not read as a confident zero (nan < tau is False)."""
    clean = _clean_scores(6)
    adv = clean.copy()
    adv[:10] = np.nan
    adv[10:15] = np.inf
    out = detection_metrics(adv, np.zeros(N, bool), _thresholds(clean))
    for p in PCTS:
        assert out[p].n_nonfinite == 15
        assert math.isnan(out[p].detection_rate)


def test_err_rates_not_swapped():
    """Errors concentrated on the flagged half must show up as err_detected."""
    clean = _clean_scores(7)
    thresholds = {50: float(np.percentile(clean, 50))}
    misclf = clean < thresholds[50]          # wrong exactly where it gets flagged
    out = detection_metrics(clean, misclf, thresholds)
    assert out[50].err_rate_detected == 1.0
    assert out[50].err_rate_passed == 0.0


def test_empty_masks_are_nan():
    clean = _clean_scores(8)
    out = detection_metrics(clean + 10.0, np.ones(N, bool), _thresholds(clean))
    for p in PCTS:
        assert math.isnan(out[p].err_rate_detected)   # nothing flagged
        assert out[p].err_rate_passed == 1.0
