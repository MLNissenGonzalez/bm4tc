"""Seam test: the whole pipeline on a tiny problem, pinned to fixed numbers.

NAT (beta=0) -> warm-started AT (beta=0.03) -> post-hoc analysis of both, on CPU,
with tiny spirals and legendre d4r3. Every phase of the `ousterhout` refactor keeps
this green, or changes an expected number on purpose and says why in the commit.

Only `run_pipeline` (the harness) is rewritten when entry points change. `EXPECTED`
(end-point metrics, loose) and `CURVES` (per-epoch training curves, tight) stay put.
"""
import csv
import json
from pathlib import Path

import pytest

from tests.e2e.conftest import python

# The two runs are the studies configs/studies/tests/seam_{nat,at}.yaml: tiny
# spirals (400 points: 200 train, 100 valid, 100 test), legendre d4r3, seed 42.
# NAT beta=0, 40 epochs; AT warm from it, beta=0.03, PGD-5 Linf eps_rel
# 0.15, 5 epochs.
STUDIES = {"nat": "tests/seam_nat", "at": "tests/seam_at"}



def _run(seam, study: str) -> tuple[Path, dict]:
    """`run` the study (train, then analyse its single job); returns the run dir
    (from bm4tc.pipeline.runs) and the job's metrics (its row of results.csv)."""
    root = seam(study)
    out = python(["-c", "import sys; from bm4tc.pipeline.runs import Study; "
                   "print(Study(sys.argv[1]).jobs()[0].run_dir)", study], root)
    with open(root / "outputs" / study / "results.csv") as f:
        (row,) = csv.DictReader(f)
    metrics = {k: float(v) for k, v in row.items() if "/" in k}
    return Path(out.strip().splitlines()[-1]), metrics


def _objective(run_dir: Path) -> float:
    """Best validation objective: what selection minimises (D8)."""
    log = json.loads((run_dir / "log.json").read_text())
    return min(r["objective/valid"] for r in log if "objective/valid" in r)


def _curves(run_dir: Path) -> dict:
    """Every logged value of each pinned key, in epoch order."""
    log = json.loads((run_dir / "log.json").read_text())
    keys = {"objective/train", "objective/valid", "rob/valid/0.15"}
    curves = {}
    for record in log:
        for k in keys & record.keys():
            curves.setdefault(k, []).append(record[k])
    return curves


def run_pipeline(seam) -> dict:
    """The harness: returns {"nat": metrics, "at": metrics} with the pinned names,
    each metrics dict holding the per-epoch "curves" too."""
    out = {}
    # AT is warm: it finds the NAT run through its run.json.
    for name in ("nat", "at"):
        run_dir, r = _run(seam, STUDIES[name])
        out[name] = {
            "objective": _objective(run_dir),
            "acc": r["acc/test"],
            "dis_loss": r["loss_dis/test"],
            "rob": r["rob/test/0.1"],
            "detection": r["detect/test/0.1/q5"],
            "clean_flagged": r["detect/test/0/q5"],
            "purified_acc": r["purify/test/0.1/d0.1"],
            "curves": _curves(run_dir),
        }
    return out


# Tolerances: losses relative, rates absolute (the test split has 100 points, so
# 0.02 is two samples).
LOSS_TOL = 1e-3
RATE_TOL = 0.02

# Re-pinned on purpose:
# - `detection`: tau calibrated on valid (D2); was 0.08 / 0.11 calibrated on test.
# - The analysis is the `analyse` parts (Phase 5): `rob` is the accuracy on the UQ
#   attack's PGD examples (random start; was a separate pass without one), every
#   part starts from seed 0, and budgets are the studies' [0.1, 0.15].
# - AT at cw = 0 (D91: clean_weight is gone; the seam ran at 0.5 before).
# - D119: the forward and log Z renormalize every 4 sites. One step agrees to
#   1.2e-7 (float32 rounding); training grows it to 2-3% in the curves and to
#   two test points in AT's rob and purified_acc.
# - D149: PGD is BarePGD (margin loss, best iterate) with 2 restarts at 10 steps:
#   `rob` 0.50 -> 0.39 (NAT), 0.63 -> 0.59 (AT); detection and purified_acc score
#   its examples. Training is untouched (the curves and losses hold).
# - D131, D190: randn_eye's 1/phi_0 rescale reaches the two edge cores and leaves the
#   class core alone (the seam's 3 sites: the edges were unscaled, the class core
#   scaled), so the initial amplitude starts at 1: every number and curve moves from
#   epoch 1 (NAT objective 0.091333 -> 0.089301, AT 0.895261 -> 1.036632, selected at
#   epoch 4 of 5 instead of 5; AT rob 0.59 -> 0.67). A different start, not a fault.
# `clean_flagged`: the clean test flag rate at the 5th-percentile tau from valid.
EXPECTED = {
    "nat": {"objective": 0.089301, "acc": 0.99, "dis_loss": 0.056415, "rob": 0.38,
            "detection": 0.28, "clean_flagged": 0.08, "purified_acc": 0.63},
    "at": {"objective": 1.036632, "acc": 0.74, "dis_loss": 0.428926, "rob": 0.67,
           "detection": 0.06, "clean_flagged": 0.04, "purified_acc": 0.70},
}

# Training curves, pinned tightly (runs are bit-for-bit deterministic on CPU). They
# guard refactors of the training loop (Phase 3): the first epoch that differs shows
# where behaviour changed. A deliberate change re-pins them in its own commit.
CURVE_TOL = 1e-6
CURVES = {
    "nat": {
        "objective/train": [
            10.0035804, 3.23337507, 2.55052137, 1.88806395, 1.33245063, 0.840240339,
            0.57314376, 0.385629207, 0.334134509, 0.303743134, 0.256799082,
            0.236969198, 0.227380579, 0.21966593, 0.209338054, 0.199478994,
            0.186986268, 0.171805054, 0.165553227, 0.159301013, 0.15059338, 0.1449903,
            0.125965819, 0.128798683, 0.126536737, 0.119484002, 0.114872232,
            0.103786208, 0.0993815437, 0.0954800099, 0.0922775442, 0.0857856075,
            0.0837825288, 0.077955503, 0.074574175, 0.0711094911, 0.0681045651,
            0.0644530008, 0.0623777732, 0.0596753607
        ],
        "objective/valid": [
            3.39269691, 2.68831779, 2.13356033, 1.59694263, 1.1055698, 0.753022156,
            0.501086426, 0.377397852, 0.348739595, 0.318443317, 0.295594282,
            0.285446749, 0.277788401, 0.268704796, 0.257500534, 0.24525425,
            0.234156208, 0.222734833, 0.21319663, 0.203476014, 0.195119567,
            0.187702165, 0.180876808, 0.174674644, 0.168802905, 0.1635672, 0.158226414,
            0.151816993, 0.145166125, 0.138444157, 0.131970558, 0.12601665,
            0.120255508, 0.1152318, 0.110347381, 0.105746694, 0.101504102,
            0.0974581695, 0.0932849622, 0.0893014383
        ],
    },
    "at": {
        "objective/train": [
            1.70142015, 1.26390914, 1.13446558, 1.07744547, 1.00748233
        ],
        "objective/valid": [
            1.58772899, 1.30892155, 1.15748009, 1.03663196, 1.04897239
        ],
        "rob/valid/0.15": [
            0.3, 0.33, 0.49, 0.52, 0.5
        ],
    },
}


@pytest.fixture(scope="module")
def results(seam):
    return run_pipeline(seam)


@pytest.mark.slow
@pytest.mark.parametrize(
    "run,metric",
    [(run, metric) for run, metrics in EXPECTED.items() for metric in metrics],
)
def test_pinned(results, run, metric):
    got, want = results[run][metric], EXPECTED[run][metric]
    if metric in ("objective", "dis_loss"):
        assert got == pytest.approx(want, rel=LOSS_TOL)
    else:
        assert got == pytest.approx(want, abs=RATE_TOL)


@pytest.mark.slow
@pytest.mark.parametrize(
    "run,key", [(run, key) for run, curves in CURVES.items() for key in curves]
)
def test_curve(results, run, key):
    got, want = results[run]["curves"][key], CURVES[run][key]
    assert len(got) == len(want), f"{len(got)} epochs logged, {len(want)} expected"
    for epoch, (g, w) in enumerate(zip(got, want), start=1):
        assert g == pytest.approx(w, rel=CURVE_TOL), f"first difference at epoch {epoch}"
