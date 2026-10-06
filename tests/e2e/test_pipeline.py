"""Seam test: the whole pipeline on a tiny problem, pinned to fixed numbers.

NAT (alpha=0) -> warm-started AT (alpha=0.01) -> post-hoc analysis of both, on CPU,
with tiny spirals and legendre d4r3. Every phase of the `ousterhout` refactor keeps
this green, or changes an expected number on purpose and says why in the commit.

Only `run_pipeline` (the harness) is rewritten when entry points change. `EXPECTED`
stays put.
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
    "trainer/nll=test",
    "trainer.nll.alpha=0.0",
    "trainer.nll.max_epoch=40",
    "trainer.nll.batch_size=64",
    "trainer.nll.optimizer.kwargs.lr=1e-2",
    "trainer.nll.save=true",
]
AT = [
    "~trainer/nll",
    "trainer/adversarial=test",  # PGD-5, Linf, eps_rel 0.15, stop_crit at_loss
    "trainer.adversarial.alpha=0.01",
    "trainer.adversarial.max_epoch=5",
    "trainer.adversarial.batch_size=64",
    "trainer.adversarial.optimizer.kwargs.lr=1e-2",
    "trainer.adversarial.save=true",
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


def _objective(run_dir: Path, key: str) -> float:
    """Best validation value of the stopping criterion: what selection is based on."""
    log = json.loads((run_dir / "log.json").read_text())
    return min(r[key] for r in log if key in r)


def _analyse(run_dir: Path, root: Path) -> dict:
    out = _python(["-c", textwrap.dedent(ANALYSE), str(run_dir)], root)
    line = next(l for l in out.splitlines() if l.startswith("RESULTS="))
    return json.loads(line.removeprefix("RESULTS="))


def run_pipeline(root: Path) -> dict:
    """The harness: returns {"nat": metrics, "at": metrics} with the pinned names."""
    nat = _train(root, "nat", DATA + NAT)
    at = _train(root, "at", DATA + AT + [f"+model_path={nat / 'models' / 'model'}"])
    out = {}
    for name, run_dir, crit in [("nat", nat, "dis_loss/valid"), ("at", at, "at_loss/valid")]:
        r = _analyse(run_dir, root)
        out[name] = {
            "objective": _objective(run_dir, crit),
            "acc": r["acc"],
            "dis_loss": r["dis_loss"],
            "rob": r["rob/0.1"],
            "detection": r["uq_detection/5pct/0.1"],
            "purified_acc": r["uq_purify_acc/0.1/0.1"],
        }
    return out


# Tolerances: losses relative, rates absolute (the test split has 100 points, so
# 0.02 is two samples).
LOSS_TOL = 1e-3
RATE_TOL = 0.02

# Expected to change on purpose:
# - `detection`: tau is calibrated on the test split today; D2 moves it to valid.
EXPECTED = {
    "nat": {"objective": 0.094781, "acc": 1.00, "dis_loss": 0.061173,
            "rob": 0.56, "detection": 0.08, "purified_acc": 0.57},
    "at": {"objective": 1.043574, "acc": 0.71, "dis_loss": 0.551664,
           "rob": 0.64, "detection": 0.08, "purified_acc": 0.65},
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
