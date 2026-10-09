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
# `clean_flagged`: the clean test flag rate at the 5th-percentile tau from valid.
EXPECTED = {
    "nat": {"objective": 0.092497, "acc": 1.00, "dis_loss": 0.057825, "rob": 0.50,
            "detection": 0.17, "clean_flagged": 0.08, "purified_acc": 0.60},
    "at": {"objective": 0.889565, "acc": 0.71, "dis_loss": 0.518483, "rob": 0.65,
           "detection": 0.08, "clean_flagged": 0.09, "purified_acc": 0.67},
}

# Training curves, pinned tightly (runs are bit-for-bit deterministic on CPU). They
# guard refactors of the training loop (Phase 3): the first epoch that differs shows
# where behaviour changed. A deliberate change re-pins them in its own commit.
CURVE_TOL = 1e-6
CURVES = {
    "nat": {
        "objective/train": [
            10.2443228, 3.50001939, 2.7231458, 1.95621745, 1.32596143, 0.895480474,
            0.630250355, 0.42090264, 0.328981022, 0.260149434, 0.22431841, 0.215830386,
            0.216520419, 0.209276453, 0.197578609, 0.185123021, 0.167001297, 0.1497115,
            0.141287987, 0.133960702, 0.128458038, 0.122757589, 0.106495755,
            0.110661174, 0.108264136, 0.103867215, 0.100245282, 0.0923300385,
            0.0898800592, 0.086343276, 0.0843170335, 0.0802549372, 0.0791884412,
            0.0743629746, 0.0732819736, 0.0708469599, 0.0692078856, 0.0667526896,
            0.0652940758, 0.0634679422
        ],
        "objective/valid": [
            3.68945847, 2.89776062, 2.25420418, 1.62023361, 1.12588207, 0.792339134,
            0.527959442, 0.386330404, 0.322004099, 0.282052231, 0.271919317, 0.27343565,
            0.273707218, 0.267721643, 0.255277491, 0.238471165, 0.219686012,
            0.201055045, 0.187432756, 0.176836271, 0.169215145, 0.162739825,
            0.156918807, 0.15203352, 0.146847296, 0.142027431, 0.137428174, 0.13279314,
            0.128302875, 0.124012389, 0.119966688, 0.116477342, 0.112968292,
            0.109913497, 0.106789894, 0.103552046, 0.100684736, 0.0979734182,
            0.0950882792, 0.0924969935
        ],
    },
    "at": {
        "objective/train": [
            1.66091506, 1.24018542, 1.08697391, 1.02466393, 0.880700529
        ],
        "objective/valid": [
            1.62679465, 1.24184342, 1.07933971, 0.984701496, 0.889564552
        ],
        "rob/valid/0.15": [
            0.22, 0.41, 0.48, 0.53, 0.53
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
