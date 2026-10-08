"""The seam studies (configs/studies/tests/seam_*) are run once per session, into
one data root, and shared by the seam tests and the figures test."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def python(args, root: Path) -> str:
    """``python <args>`` in the repo, on CPU, with ``root`` as the data root.

    The CPU kernels are fixed to AVX2 (torch and MKL): AVX-512 kernels round
    differently, and PGD's sign() steps turn that into different AT curves, so the
    pinned numbers would hold only on the CPU they were pinned on (D85)."""
    env = {**os.environ, "BM4TC_DATA_ROOT": str(root), "CUDA_VISIBLE_DEVICES": "",
           "ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "AVX2"}
    proc = subprocess.run(
        [sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


@pytest.fixture(scope="session")
def seam(tmp_path_factory):
    """``seam(study)`` runs the study (once per session) and returns the data root.
    Warm studies need their source run first (seam_nat before seam_at)."""
    root = tmp_path_factory.mktemp("seam")
    done = set()

    def run(study: str) -> Path:
        if study not in done:
            python(["-m", "bm4tc", "run", study], root)
            done.add(study)
        return root
    return run
