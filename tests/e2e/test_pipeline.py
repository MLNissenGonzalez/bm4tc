"""Seam test: the whole pipeline on a tiny problem, pinned to fixed numbers.

NAT (alpha=0) -> warm-started AT (alpha=0.01) -> post-hoc analysis of both, on CPU,
with tiny spirals and legendre d4r3. Every phase of the `ousterhout` refactor keeps
this green, or changes an expected number on purpose and says why in the commit.

Only `run_pipeline` (the harness) is rewritten when entry points change. `EXPECTED`
(end-point metrics, loose) and `CURVES` (per-epoch training curves, tight) stay put.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

DATA = [
    "dataset=2Dtoy/spirals",
    "dataset.gen_dow_kwargs.size=200",  # 400 points: 200 train, 100 valid, 100 test
    "born=legendre/d4r3c64",
]
NAT = [
    "trainer=nat/test",
    "trainer.alpha=0.0",
    "trainer.max_epoch=40",
    "trainer.batch_size=64",
    "trainer.optimizer.kwargs.lr=1e-2",
    "trainer.save=true",
]
AT = [
    "trainer=at/test",  # PGD-5, Linf, eps_rel 0.15
    "trainer.alpha=0.01",
    "trainer.clean_weight=0.5",  # exercises every term of the AT objective
    "trainer.max_epoch=5",
    "trainer.batch_size=64",
    "trainer.optimizer.kwargs.lr=1e-2",
    "trainer.save=true",
]

ANALYSE = """
import json, sys, torch
from pathlib import Path
from analysis.run import AnalysisConfig, analyze_run
torch.manual_seed(0)
cfg = AnalysisConfig(
    compute_acc=True, compute_dis_loss=True, compute_rob=True,
    compute_uq=True, compute_rob_ceiling=False, device="cpu",
    evasion_override={"method": "PGD", "norm": "inf", "num_steps": 10,
                      "random_start": False, "eps_rel": [0.1]},
    uq_config={"eps_rel": [0.1], "delta_rel": [0.1], "percentiles": [5],
               "attack_num_steps": 10, "num_steps": 10},
)
print("RESULTS=" + json.dumps(analyze_run(Path(sys.argv[1]), cfg)))
"""


def _python(args, root: Path) -> str:
    env = {**os.environ, "BM4TC_DATA_ROOT": str(root), "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def _train(root: Path, name: str, overrides: list[str]) -> Path:
    run_dir = root / "outputs" / name  # the W&B group needs an `outputs/` ancestor
    _python(["-m", "experiments.train", *overrides, f"hydra.run.dir={run_dir}"], root)
    return run_dir


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


def _analyse(run_dir: Path, root: Path) -> dict:
    out = _python(["-c", textwrap.dedent(ANALYSE), str(run_dir)], root)
    line = next(l for l in out.splitlines() if l.startswith("RESULTS="))
    return json.loads(line.removeprefix("RESULTS="))


def run_pipeline(root: Path) -> dict:
    """The harness: returns {"nat": metrics, "at": metrics} with the pinned names,
    each metrics dict holding the per-epoch "curves" too."""
    nat = _train(root, "nat", DATA + NAT)
    at = _train(root, "at", DATA + AT + [f"model_path={nat / 'models' / 'model'}"])
    out = {}
    for name, run_dir in [("nat", nat), ("at", at)]:
        r = _analyse(run_dir, root)
        out[name] = {
            "objective": _objective(run_dir),
            "acc": r["acc"],
            "dis_loss": r["dis_loss"],
            "rob": r["rob/0.1"],
            "detection": r["uq_detection/5pct/0.1"],
            "purified_acc": r["uq_purify_acc/0.1/0.1"],
            "curves": _curves(run_dir),
        }
    return out


# Tolerances: losses relative, rates absolute (the test split has 100 points, so
# 0.02 is two samples).
LOSS_TOL = 1e-3
RATE_TOL = 0.02

# Expected to change on purpose:
# - `detection`: tau is calibrated on the test split today; D2 moves it to valid.
EXPECTED = {
    "nat": {"objective": 0.093104, "acc": 1.00, "dis_loss": 0.058333,
            "rob": 0.56, "detection": 0.08, "purified_acc": 0.57},
    "at": {"objective": 0.703004, "acc": 0.87, "dis_loss": 0.321820,
           "rob": 0.66, "detection": 0.11, "purified_acc": 0.65},
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
def results(tmp_path_factory):
    return run_pipeline(tmp_path_factory.mktemp("seam"))


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
