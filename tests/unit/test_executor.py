"""bm4tc/pipeline/executor.py: the job pool (D38), with trivial subprocesses."""
import json
import sys

from bm4tc.pipeline.executor import Unit, execute, slots


def _unit(tmp_path, name, code, deps=(), after=()):
    return Unit(name, [sys.executable, "-c", code], tmp_path / f"{name}.log", list(deps),
                list(after))


def test_slots():
    assert slots(None, 2) == [None, None]
    assert slots(["0", "1"], 2) == ["0", "1", "0", "1"]


def test_dependencies_order_and_failures_skip_downstream(tmp_path):
    order = tmp_path / "order"
    write = lambda tag: f"open({str(order)!r}, 'a').write({tag!r})"
    units = [
        _unit(tmp_path, "b", write("b"), deps=["a"]),
        _unit(tmp_path, "a", write("a")),
        _unit(tmp_path, "bad", "import sys; print('boom'); sys.exit(3)"),
        _unit(tmp_path, "after_bad", write("x"), deps=["bad"]),
        _unit(tmp_path, "after_after", write("y"), deps=["after_bad"]),
    ]
    state_file = tmp_path / "status.json"
    states = execute(units, per_gpu=2, state_file=state_file, poll=0.05)
    assert states == {"b": "done", "a": "done", "bad": "failed",
                      "after_bad": "skipped", "after_after": "skipped"}
    assert order.read_text() == "ab"
    assert (tmp_path / "bad.log").read_text().strip() == "boom"
    saved = json.loads(state_file.read_text())["units"]
    assert saved["bad"]["exit"] == 3


def test_after_waits_for_the_end_in_any_state(tmp_path):
    """A unit with ``after`` starts once those units ended, failed or skipped too,
    and not before (D99: select after the HPO workers)."""
    order = tmp_path / "order"
    write = lambda tag: f"open({str(order)!r}, 'a').write({tag!r})"
    units = [
        _unit(tmp_path, "last", write("z"), after=["bad", "skipped", "slow"]),
        _unit(tmp_path, "bad", "import sys; sys.exit(3)"),
        _unit(tmp_path, "skipped", write("x"), deps=["bad"]),
        _unit(tmp_path, "slow", "import time; time.sleep(0.5); " + write("s")),
    ]
    states = execute(units, per_gpu=4, poll=0.05)
    assert states == {"last": "done", "bad": "failed", "skipped": "skipped", "slow": "done"}
    assert order.read_text() == "sz"


def test_each_slot_pins_its_gpu(tmp_path):
    code = "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"
    units = [_unit(tmp_path, f"u{i}", code) for i in range(4)]
    execute(units, gpus=["3", "5"], per_gpu=1, poll=0.05, stagger=0)
    seen = {(tmp_path / f"u{i}.log").read_text().strip() for i in range(4)}
    assert seen == {"3", "5"}


def test_starts_on_one_gpu_are_staggered(tmp_path):
    """D130: units start on one GPU at least ``stagger`` seconds apart, on
    different GPUs at once."""
    code = "import time; print(time.monotonic())"
    units = [_unit(tmp_path, f"u{i}", code) for i in range(4)]
    execute(units, gpus=["3", "5"], per_gpu=2, poll=0.05, stagger=0.5)
    starts = sorted(float((tmp_path / f"u{i}.log").read_text()) for i in range(4))
    assert starts[1] - starts[0] < 0.4                 # one unit on each GPU at once
    assert starts[2] - starts[0] >= 0.45               # then the second on each


def test_mps_units_use_their_gpus_daemon(tmp_path, monkeypatch):
    """--mps: each GPU's daemon is ensured once, and a unit sees device 0 of the
    daemon serving its slot's GPU (D90)."""
    from bm4tc.pipeline import mps
    ensured = []
    monkeypatch.setattr(mps, "ensure_running", lambda gpu: ensured.append(gpu) or True)
    code = ("import os; print(os.environ['CUDA_VISIBLE_DEVICES'], "
            "os.environ['CUDA_MPS_PIPE_DIRECTORY'])")
    units = [_unit(tmp_path, f"u{i}", code) for i in range(4)]
    state_file = tmp_path / "status.json"
    execute(units, gpus=["3", "5"], per_gpu=2, state_file=state_file, poll=0.05, mps=True,
            stagger=0)
    assert ensured == ["3", "5"]
    seen = {tuple((tmp_path / f"u{i}.log").read_text().split()) for i in range(4)}
    assert seen == {("0", str(mps.pipe_directory("3"))), ("0", str(mps.pipe_directory("5")))}
    assert json.loads(state_file.read_text())["mps"] is True


def test_mps_needs_gpus(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="--gpus"):
        execute([_unit(tmp_path, "u", "pass")], mps=True)


def test_a_unit_given_a_gpu_fails_without_cuda(monkeypatch):
    """A broken MPS server leaves CUDA unavailable: the unit raises, not run on the
    CPU (D150); a unit given no GPU, or hidden from them (""), runs on the CPU."""
    import pytest
    import torch
    from bm4tc.pipeline import stages  # imports torch; only here
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        stages._device()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert stages._device() == torch.device("cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert stages._device() == torch.device("cpu")
