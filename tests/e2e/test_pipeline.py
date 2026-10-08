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
# NAT beta=0, 40 epochs; AT warm from it, beta=0.03, cw=0.5, PGD-5 Linf eps_rel
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
    "nat": {"objective": 0.093867, "acc": 1.00, "dis_loss": 0.058943, "rob": 0.52,
            "detection": 0.16, "clean_flagged": 0.08, "purified_acc": 0.60},
    "at": {"objective": 0.691987, "acc": 0.80, "dis_loss": 0.306710, "rob": 0.65,
           "detection": 0.13, "clean_flagged": 0.06, "purified_acc": 0.67},
}

# Training curves, pinned tightly (runs are bit-for-bit deterministic on CPU). They
# guard refactors of the training loop (Phase 3): the first epoch that differs shows
# where behaviour changed. A deliberate change re-pins them in its own commit.
CURVE_TOL = 1e-6
CURVES = {
    "nat": {
        "objective/train": [
            10.2443236, 3.50002265, 2.72325039, 1.95677964, 1.32688936, 0.896406571,
            0.632347385, 0.424031417, 0.332409253, 0.262468015, 0.225417321,
            0.215850944, 0.215988318, 0.208485628, 0.197233955, 0.184709519,
            0.166667228, 0.149365631, 0.140879653, 0.133472594, 0.127848998,
            0.122074974, 0.105696072, 0.110129726, 0.107594579, 0.103330262,
            0.0997187197, 0.0919266244, 0.0896509861, 0.0861525461, 0.0841770545,
            0.0803049207, 0.0793162261, 0.0745204004, 0.073739325, 0.0714517136,
            0.0697799213, 0.067425472, 0.0661192164, 0.0644410104
        ],
        "objective/valid": [
            3.68946388, 2.89777672, 2.25448158, 1.62112316, 1.1268491, 0.794747276,
            0.532569256, 0.392595978, 0.326909237, 0.283774929, 0.271322069,
            0.271670113, 0.271966782, 0.266478157, 0.254426327, 0.237882051,
            0.219044247, 0.200308971, 0.186611085, 0.175973244, 0.168341713, 0.16190032,
            0.156192985, 0.151377225, 0.146243443, 0.141454716, 0.136908555,
            0.132408934, 0.128001165, 0.123805094, 0.119882584, 0.116471682,
            0.113112726, 0.110208778, 0.107270374, 0.104218092, 0.101515996,
            0.098934412, 0.0962615299, 0.093866868
        ],
    },
    "at": {
        "objective/train": [
            0.883839945, 0.717436373, 0.703900933, 0.687736352, 0.688639263
        ],
        "objective/valid": [
            0.733352974, 0.706644136, 0.691986831, 0.694972626, 0.738293909
        ],
        "rob/valid/0.15": [
            0.4, 0.42, 0.52, 0.56, 0.42
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
