"""Every config composes under the structured schema, and the schema rejects unknown keys (D25)."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from experiments.config import register
from experiments.resolvers import register_resolvers

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
GROUPS = ["born", "dataset", "trainer/nat", "trainer/at", "tracking"]

register()
register_resolvers()


def _cases():
    yield []
    for test in sorted((CONFIGS / "experiments" / "tests").glob("*.yaml")):
        yield [f"+experiments=tests/{test.stem}"]
    for group in GROUPS:
        for f in sorted((CONFIGS / group).rglob("*.yaml")):
            option = f.relative_to(CONFIGS / group).with_suffix("").as_posix()
            if group == "trainer/at":
                yield ["~trainer/nat", f"trainer/at={option}"]
            else:
                yield [f"{group}={option}"]


def _compose(overrides):
    with initialize_config_dir(str(CONFIGS), version_base=None):
        return compose("config", overrides=overrides)


@pytest.mark.parametrize("overrides", list(_cases()), ids=lambda o: " ".join(o) or "default")
def test_config_composes_under_schema(overrides):
    OmegaConf.to_container(_compose(overrides), resolve=False)


def test_unknown_key_is_rejected():
    """A key outside the schema fails, here a stale one. (An explicit `+key=` append
    is Hydra's force-add and bypasses the schema by design.)"""
    with pytest.raises(ConfigCompositionException):
        _compose(["trainer.nat.stop_crit=acc"])
