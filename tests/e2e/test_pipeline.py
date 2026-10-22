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
# `clean_flagged`: the clean test flag rate at the 5th-percentile tau from valid.
EXPECTED = {
    "nat": {"objective": 0.091333, "acc": 1.00, "dis_loss": 0.057163, "rob": 0.39,
            "detection": 0.20, "clean_flagged": 0.08, "purified_acc": 0.59},
    "at": {"objective": 0.895261, "acc": 0.71, "dis_loss": 0.522472, "rob": 0.59,
           "detection": 0.11, "clean_flagged": 0.09, "purified_acc": 0.68},
}

# Training curves, pinned tightly (runs are bit-for-bit deterministic on CPU). They
# guard refactors of the training loop (Phase 3): the first epoch that differs shows
# where behaviour changed. A deliberate change re-pins them in its own commit.
CURVE_TOL = 1e-6
CURVES = {
    "nat": {
        "objective/train": [
            10.2421271, 3.49894102, 2.72528831, 1.96175305, 1.32698949, 0.877519071,
            0.616250237, 0.414111594, 0.330580155, 0.262403493, 0.226027717,
            0.216248175, 0.214749555, 0.205876604, 0.193910425, 0.181313848,
            0.163642973, 0.146982665, 0.138750543, 0.131746724, 0.126432079,
            0.120667592, 0.104343971, 0.10866271, 0.106397281, 0.102196343,
            0.0986006781, 0.0906825761, 0.0883177097, 0.0849286417, 0.0829217273,
            0.0789424454, 0.0779143299, 0.073248251, 0.0720853135, 0.0697607771,
            0.0681755804, 0.0657508783, 0.0643409217, 0.0625590595
        ],
        "objective/valid": [
            3.68727249, 2.89790321, 2.25550377, 1.62310848, 1.11296219, 0.778801804,
            0.521056747, 0.390835028, 0.328769484, 0.285055885, 0.271353664,
            0.269600534, 0.267438087, 0.260336647, 0.247811441, 0.231817446,
            0.214199133, 0.196696153, 0.1839115, 0.173981304, 0.16677916, 0.160471339,
            0.154766955, 0.149914117, 0.144761362, 0.139939303, 0.135329075,
            0.130745926, 0.126341004, 0.122175684, 0.118225937, 0.114827385,
            0.111426969, 0.108472209, 0.105451941, 0.102237427, 0.0993902874,
            0.0967045283, 0.0938682079, 0.0913328815
        ],
    },
    "at": {
        "objective/train": [
            1.65380418, 1.25951266, 1.08619157, 1.01425143, 0.878711422
        ],
        "objective/valid": [
            1.61963766, 1.2685713, 1.07247394, 0.978318613, 0.895261117
        ],
        "rob/valid/0.15": [
            0.23, 0.43, 0.48, 0.53, 0.5
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
