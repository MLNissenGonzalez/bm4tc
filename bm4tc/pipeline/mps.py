"""NVIDIA MPS daemons for ``--mps`` launches (D90).

A captured training step saturates the GPU only through concurrency: under the
Multi-Process Service, the units on one GPU run their kernels at the same time
instead of time-slicing. One daemon serves one physical GPU; it is found through
its pipe directory, ``/tmp/bm4tc-mps-<user>/<gpu>``, local to the node.

A launch starts the daemons its GPUs lack and leaves them running: other studies
on the node may be using them. Stop one with
``echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/bm4tc-mps-$USER/<gpu> nvidia-cuda-mps-control``.
Eager units run slower under MPS: launch them without ``--mps``.
"""
import getpass
import os
import subprocess
from pathlib import Path
from typing import Dict

CONTROL = "nvidia-cuda-mps-control"


def pipe_directory(gpu: str) -> Path:
    return Path("/tmp") / f"bm4tc-mps-{getpass.getuser()}" / str(gpu)


def _daemon_env(gpu: str) -> Dict[str, str]:
    directory = pipe_directory(gpu)
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_MPS_PIPE_DIRECTORY": str(directory),
            "CUDA_MPS_LOG_DIRECTORY": str(directory / "log")}


def is_running(gpu: str) -> bool:
    """Whether a daemon answers on this GPU's pipe directory."""
    if not (pipe_directory(gpu) / "control").exists():
        return False
    answer = subprocess.run([CONTROL], input="get_server_list\n", env=_daemon_env(gpu),
                            capture_output=True, text=True, timeout=10)
    return answer.returncode == 0


def ensure_running(gpu: str) -> bool:
    """Start this GPU's daemon unless one is running; True if it was started."""
    if is_running(gpu):
        return False
    (pipe_directory(gpu) / "log").mkdir(parents=True, exist_ok=True)
    subprocess.run([CONTROL, "-d"], env=_daemon_env(gpu), check=True, timeout=30)
    return True


def client_env(gpu: str) -> Dict[str, str]:
    """The environment entries of a unit served by this GPU's daemon. Device
    indices are relative to the daemon, which sees only its GPU: always 0."""
    return {"CUDA_VISIBLE_DEVICES": "0", "CUDA_MPS_PIPE_DIRECTORY": str(pipe_directory(gpu))}
