"""init_wandb: an unreachable W&B makes the run log offline, not fail (D146)."""
import pytest
import wandb
from omegaconf import OmegaConf

from bm4tc.pipeline import tracking


def _cfg(mode):
    return OmegaConf.create({"tracking": {"project": "p", "entity": "", "mode": mode}})


@pytest.mark.parametrize("error", [wandb.errors.CommError("Timed out initializing run"),
                                   TimeoutError("Timed out waiting for response")])
def test_unreachable_wandb_logs_offline(monkeypatch, tmp_path, error):
    modes = []

    def init(mode, **kwargs):
        modes.append(mode)
        if mode == "online":
            raise error
        return "run"

    monkeypatch.setattr(tracking.wandb, "init", init)
    assert tracking.init_wandb(_cfg("online"), tmp_path, {"name": "s1"}) == "run"
    assert modes == ["online", "offline"]


def test_other_modes_raise(monkeypatch, tmp_path):
    def init(mode, **kwargs):
        raise wandb.errors.CommError("x")

    monkeypatch.setattr(tracking.wandb, "init", init)
    with pytest.raises(wandb.errors.CommError):
        tracking.init_wandb(_cfg("offline"), tmp_path, {"name": "s1"})
