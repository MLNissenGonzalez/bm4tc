"""Seam test: the whole pipeline on a tiny problem, pinned to fixed numbers.

NAT (beta=0) -> warm-started AT (beta=0.01) -> post-hoc analysis of both, on CPU,
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
# NAT beta=0, 40 epochs; AT warm from it, beta=0.01, cw=0.5, PGD-5 Linf eps_rel
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
# `clean_flagged`: the clean test flag rate at the 5th-percentile tau from valid.
EXPECTED = {
    "nat": {"objective": 0.093104, "acc": 1.00, "dis_loss": 0.058333, "rob": 0.51,
            "detection": 0.17, "clean_flagged": 0.08, "purified_acc": 0.61},
    "at": {"objective": 0.703004, "acc": 0.87, "dis_loss": 0.321820, "rob": 0.66,
           "detection": 0.04, "clean_flagged": 0.03, "purified_acc": 0.66},
}

# Training curves, pinned tightly (runs are bit-for-bit deterministic on CPU). They
# guard refactors of the training loop (Phase 3): the first epoch that differs shows
# where behaviour changed. A deliberate change re-pins them in its own commit.
CURVE_TOL = 1e-6
CURVES = {
    "nat": {
        "objective/train": [
            10.2443206, 3.49993428, 2.72254268, 1.95532624, 1.32624177, 0.900012016,
            0.640850246, 0.433186998, 0.339725147, 0.269476682, 0.229536946, 0.21611236,
            0.214522799, 0.206922422, 0.197384929, 0.185638542, 0.16865777, 0.151637087,
            0.142864602, 0.135087952, 0.128797275, 0.122721322, 0.106225769,
            0.110900092, 0.108302365, 0.103795963, 0.100063475, 0.0921240722,
            0.0897815575, 0.0861965939, 0.0841273293, 0.0801350921, 0.079054907,
            0.0741878611, 0.0733503252, 0.071075776, 0.0692698459, 0.0668292915,
            0.0655378836, 0.0638445417
        ],
        "objective/valid": [
            3.68946243, 2.89756584, 2.25360619, 1.62035385, 1.12944073, 0.80336689,
            0.546403217, 0.405978451, 0.339158783, 0.290294657, 0.270872631,
            0.267105274, 0.267074165, 0.263591518, 0.253946371, 0.239652071, 0.22177907,
            0.203029909, 0.188856845, 0.177517715, 0.169379268, 0.162785254,
            0.157130151, 0.152414579, 0.147170768, 0.14224577, 0.137582793, 0.132948976,
            0.128456373, 0.124154239, 0.120089774, 0.116531601, 0.113039274,
            0.110017018, 0.106984839, 0.103833795, 0.101015506, 0.0983232474,
            0.0955610228, 0.0931043959
        ],
    },
    "at": {
        "objective/train": [
            0.880207499, 0.701266925, 0.726055364, 0.701490104, 0.67717284
        ],
        "objective/valid": [
            0.80951635, 0.741676149, 0.715386747, 0.703004186, 0.708757467
        ],
        "rob/valid/0.15": [
            0.4, 0.44, 0.5, 0.5, 0.4
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
