"""bm4tc/pipeline/executor.py: the job pool (D38), with trivial subprocesses."""
import json
import sys

from bm4tc.pipeline.executor import Unit, execute, slots


def _unit(tmp_path, name, code, deps=()):
    return Unit(name, [sys.executable, "-c", code], tmp_path / f"{name}.log", list(deps))


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


def test_each_slot_pins_its_gpu(tmp_path):
    code = "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"
    units = [_unit(tmp_path, f"u{i}", code) for i in range(4)]
    execute(units, gpus=["3", "5"], per_gpu=1, poll=0.05)
    seen = {(tmp_path / f"u{i}.log").read_text().strip() for i in range(4)}
    assert seen == {"3", "5"}
