"""A small job pool for a DAG of units (D37, D38).

Each unit is a subprocess with its own log file. The pool has one slot per
(GPU, k) for ``--gpus 0,1 --per-gpu k``; a unit's ``CUDA_VISIBLE_DEVICES`` is
its slot's GPU. Without GPUs there are ``per_gpu`` unpinned slots. A unit
starts when every unit it depends on succeeded; when a unit fails, everything
downstream of it is skipped and the rest goes on. The state of every unit is
written to a JSON file as it changes, for ``status``.

The pool knows nothing about studies: :mod:`experiments.stages` builds the units.
"""
import datetime
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"


@dataclass
class Unit:
    name: str                       # unique, path-like: train/legendre/d3r40/a0/s1
    argv: List[str]
    log: Path
    deps: List[str] = field(default_factory=list)


def slots(gpus: Optional[Sequence[str]], per_gpu: int) -> List[Optional[str]]:
    """One entry per slot: the GPU it pins, or None (not pinned)."""
    if not gpus:
        return [None] * per_gpu
    return [gpu for _ in range(per_gpu) for gpu in gpus]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def execute(units: List[Unit], gpus: Optional[Sequence[str]] = None, per_gpu: int = 1,
            state_file: Optional[Path] = None, poll: float = 0.5) -> Dict[str, str]:
    """Run the units; returns each unit's final state (done, failed or skipped)."""
    by_name = {u.name: u for u in units}
    for u in units:
        unknown = set(u.deps) - set(by_name)
        if unknown:
            raise ValueError(f"{u.name} depends on unknown units {sorted(unknown)}")
    state = {u.name: {"state": PENDING, "log": str(u.log)} for u in units}
    free = slots(gpus, per_gpu)
    running: Dict[str, tuple] = {}  # name -> (process, slot gpu, log handle)

    def save():
        if state_file is not None:
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"pid": os.getpid(), "units": state}, indent=2))
            tmp.replace(state_file)

    def set_state(name, value, **extra):
        state[name].update(state=value, **extra)
        save()

    save()
    try:
        while True:
            # Skip everything downstream of a failure.
            changed = True
            while changed:
                changed = False
                for u in units:
                    if state[u.name]["state"] == PENDING and any(
                            state[d]["state"] in (FAILED, SKIPPED) for d in u.deps):
                        set_state(u.name, SKIPPED)
                        changed = True
            ready = [u for u in units if state[u.name]["state"] == PENDING
                     and all(state[d]["state"] == DONE for d in u.deps)]
            while ready and free:
                u, gpu = ready.pop(0), free.pop(0)
                u.log.parent.mkdir(parents=True, exist_ok=True)
                handle = open(u.log, "w")
                env = dict(os.environ)
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                proc = subprocess.Popen(u.argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
                running[u.name] = (proc, gpu, handle)
                set_state(u.name, RUNNING, started=_now(), gpu=gpu)
                logger.info(f"start {u.name}" + (f" on GPU {gpu}" if gpu is not None else ""))
            if not running:
                break
            time.sleep(poll)
            for name, (proc, gpu, handle) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                del running[name]
                free.append(gpu)
                if code == 0:
                    set_state(name, DONE, finished=_now())
                    logger.info(f"done  {name}")
                else:
                    set_state(name, FAILED, finished=_now(), exit=code)
                    logger.error(f"FAILED {name} (exit {code}); log: {by_name[name].log}")
    finally:
        for name, (proc, _, handle) in running.items():
            proc.terminate()
            proc.wait()
            handle.close()
            set_state(name, FAILED, finished=_now(), exit="interrupted")
    return {name: s["state"] for name, s in state.items()}
