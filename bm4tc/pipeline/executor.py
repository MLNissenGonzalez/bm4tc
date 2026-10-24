"""A small job pool for a DAG of units (D37, D38).

Each unit is a subprocess with its own log file. The pool has one slot per
(GPU, k) for ``--gpus 0,1 --per-gpu k``; a unit's ``CUDA_VISIBLE_DEVICES`` is
its slot's GPU, or with ``mps`` that GPU's MPS daemon (:mod:`bm4tc.pipeline.mps`,
started if it is not running). Without GPUs there are ``per_gpu`` unpinned slots. A unit
starts when every unit in its ``deps`` succeeded and every unit in its ``after``
ended, in any state; when a unit fails, everything that depends on it is skipped
and the rest goes on. Units start on one GPU at least ``stagger`` seconds apart
(D130), so their warm-up peaks do not coincide and its MPS daemon is not asked to
take several clients at once. The state of every unit is written to a JSON file as it
changes, for ``status``. Neither that file nor a unit's log can end a launch: a write
the filesystem refuses is logged and the pool goes on (D159), since the units' own
outputs, not these, hold the results.

The pool knows nothing about studies: :mod:`bm4tc.pipeline.stages` builds the units.
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
STAGGER = 10.0  # seconds between two starts on one GPU (D130)


@dataclass
class Unit:
    name: str                       # unique, path-like: train/legendre/d3r40/a0/s1
    argv: List[str]
    log: Path
    deps: List[str] = field(default_factory=list)   # must succeed first
    after: List[str] = field(default_factory=list)  # must end first, in any state


def slots(gpus: Optional[Sequence[str]], per_gpu: int) -> List[Optional[str]]:
    """One entry per slot: the GPU it pins, or None (not pinned)."""
    if not gpus:
        return [None] * per_gpu
    return [gpu for _ in range(per_gpu) for gpu in gpus]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _close(handle) -> None:
    """Close a unit's log. On the lab's full ceph export a close has failed with
    EACCES (2026-10-24), which ended the launch and killed every running unit; the
    log may lose its tail, the unit's state still comes from its exit code (D159)."""
    try:
        handle.close()
    except OSError as e:
        logger.warning(f"closing the log {handle.name} failed ({e}); it may be cut short")


def execute(units: List[Unit], gpus: Optional[Sequence[str]] = None, per_gpu: int = 1,
            state_file: Optional[Path] = None, poll: float = 0.5,
            mps: bool = False, stagger: float = STAGGER) -> Dict[str, str]:
    """Run the units; returns each unit's final state (done, failed or skipped)."""
    if mps:
        if not gpus:
            raise ValueError("--mps needs --gpus")
        from bm4tc.pipeline import mps as mps_daemons
        for gpu in dict.fromkeys(gpus):
            started = mps_daemons.ensure_running(gpu)
            logger.info(f"MPS daemon for GPU {gpu}: {'started' if started else 'running'} "
                        f"({mps_daemons.pipe_directory(gpu)})")
    by_name = {u.name: u for u in units}
    for u in units:
        unknown = set(u.deps + u.after) - set(by_name)
        if unknown:
            raise ValueError(f"{u.name} depends on unknown units {sorted(unknown)}")
    state = {u.name: {"state": PENDING, "log": str(u.log)} for u in units}
    free = slots(gpus, per_gpu)
    running: Dict[str, tuple] = {}  # name -> (process, slot gpu, log handle)
    last_start: Dict[str, float] = {}  # gpu -> time of its last start

    def save():
        if state_file is None:
            return
        tmp = state_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"pid": os.getpid(), "mps": mps, "units": state},
                                      indent=2))
            tmp.replace(state_file)
        except OSError as e:        # the next change writes it again (D159)
            logger.warning(f"writing {state_file} failed ({e}); `status` may lag")

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
                     and all(state[d]["state"] == DONE for d in u.deps)
                     and all(state[d]["state"] in (DONE, FAILED, SKIPPED) for d in u.after)]
            while ready and free:
                now = time.monotonic()
                slot = next((i for i, g in enumerate(free) if g is None
                             or now - last_start.get(g, -stagger) >= stagger), None)
                if slot is None:
                    break                   # every free slot's GPU started a unit just now
                u, gpu = ready.pop(0), free.pop(slot)
                if gpu is not None:
                    last_start[gpu] = now
                u.log.parent.mkdir(parents=True, exist_ok=True)
                handle = open(u.log, "w")
                env = dict(os.environ)
                if gpu is not None and mps:
                    env.update(mps_daemons.client_env(gpu))
                elif gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                proc = subprocess.Popen(u.argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
                running[u.name] = (proc, gpu, handle)
                set_state(u.name, RUNNING, started=_now(), gpu=gpu)
                logger.info(f"start {u.name}" + (f" on GPU {gpu}" if gpu is not None else ""))
            if not running and not (ready and free):
                break
            time.sleep(poll)
            for name, (proc, gpu, handle) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                _close(handle)
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
            _close(handle)
            set_state(name, FAILED, finished=_now(), exit="interrupted")
    return {name: s["state"] for name, s in state.items()}
