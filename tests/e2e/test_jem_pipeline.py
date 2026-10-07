"""The JEM seam: `run` the studies tests/seam_jem_{nat,at} (tiny spirals, JEM sized
to d4r3, alpha 0 and 0.5, AT warm from NAT) and pin their results.csv rows.

Pinned at Phase 6 (D69-D72), when JEM joined the pipeline; later phases keep them
or change them on purpose, as the MPS seam (tests/e2e/test_pipeline.py).
"""
import csv

import pytest

from tests.e2e.test_pipeline import _python

STUDIES = {"nat": "tests/seam_jem_nat", "at": "tests/seam_jem_at"}
LOSS_TOL = 1e-3
RATE_TOL = 0.02

EXPECTED = {
    ("nat", 0.0): {"objective/valid": 0.546472, "acc/test": 0.72, "loss_dis/test": 0.523113,
                   "rob/test/0.1": 0.62, "rob_joint/test/0.1": 0.62, "detect/test/0.1/q5": 0.02,
                   "purify/test/0.1/d0.1": 0.61, "purify_sgld/test/0.1/k2": 0.54,
                   "purify_sgld/test/0/k1": 0.64},
    ("nat", 0.5): {"objective/valid": 0.611876, "acc/test": 0.72, "loss_dis/test": 0.564802,
                   "rob/test/0.1": 0.6, "rob_joint/test/0.1": 0.6, "detect/test/0.1/q5": 0.0,
                   "purify/test/0.1/d0.1": 0.59, "purify_sgld/test/0.1/k2": 0.54,
                   "purify_sgld/test/0/k1": 0.64},
    ("at", 0.0): {"objective/valid": 0.664363, "acc/test": 0.71, "loss_dis/test": 0.548603,
                  "rob/test/0.1": 0.63, "rob_joint/test/0.1": 0.63, "detect/test/0.1/q5": 0.01,
                  "purify/test/0.1/d0.1": 0.56, "purify_sgld/test/0.1/k2": 0.5,
                  "purify_sgld/test/0/k1": 0.62},
    ("at", 0.5): {"objective/valid": 0.631959, "acc/test": 0.71, "loss_dis/test": 0.529748,
                  "rob/test/0.1": 0.6, "rob_joint/test/0.1": 0.6, "detect/test/0.1/q5": 0.01,
                  "purify/test/0.1/d0.1": 0.56, "purify_sgld/test/0.1/k2": 0.52,
                  "purify_sgld/test/0/k1": 0.62},
}


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    root = tmp_path_factory.mktemp("jem_seam")
    out = {}
    for regime, study in STUDIES.items():   # AT is warm: NAT first
        _python(["-m", "bm4tc", "run", study], root)
        with open(root / "outputs" / study / "results.csv") as f:
            for row in csv.DictReader(f):
                out[(regime, float(row["alpha"]))] = row
    return out


@pytest.mark.slow
def test_jem_rows(rows):
    assert set(rows) == set(EXPECTED)
    row = rows[("nat", 0.0)]
    assert row["model"] == "jem" and row["embedding"] == "raw"
    assert "loss_gen/test" not in row          # no exact log Z for JEM
    assert not any(k.startswith("purify_gibbs") for k in row)


@pytest.mark.slow
@pytest.mark.parametrize("run,metric", [(r, m) for r, ms in EXPECTED.items() for m in ms])
def test_jem_pinned(rows, run, metric):
    got, want = float(rows[run][metric]), EXPECTED[run][metric]
    if "loss" in metric or "objective" in metric:
        assert got == pytest.approx(want, rel=LOSS_TOL)
    else:
        assert got == pytest.approx(want, abs=RATE_TOL)
