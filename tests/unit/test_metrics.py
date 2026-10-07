"""Metric keys (bm4tc/pipeline/metrics.py, D48)."""
from bm4tc.pipeline.metrics import SELECTION, flatten, flatten_epoch, key


def test_key_rule():
    assert key("objective", "valid") == "objective/valid"
    assert key("rob", "valid", 0.15) == "rob/valid/0.15"
    assert key("rob", "test", 0.1) == "rob/test/0.1"
    assert SELECTION == key("objective", "valid")


def test_flatten_budgeted_values():
    out = flatten("valid", {"objective": 1.0, "rob": {0.1: 0.5, 0.15: 0.4}})
    assert out == {"objective/valid": 1.0, "rob/valid/0.1": 0.5, "rob/valid/0.15": 0.4}


def test_flatten_epoch_passes_norm_through():
    record = {
        "train": {"objective": 2.0, "penalty": 0.1},
        "valid": {"acc": 0.9},
        "diagnostics": {"norm/log_Z_mean": 0.3},
    }
    assert flatten_epoch(record) == {
        "objective/train": 2.0, "penalty/train": 0.1,
        "acc/valid": 0.9, "norm/log_Z_mean": 0.3,
    }
