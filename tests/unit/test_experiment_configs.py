"""Every 2-D toy experiment config must compose, and the Phase 2 tree must be uniform.

Two levels of check:

* **composition** — every config under ``configs/experiments/{spirals,circles,
  moons}`` composes and resolves. Catches dead ``override`` targets, missing
  resolvers, and typos in a group name, none of which surface until a run is
  launched — and in particular catches a config left pointing at a born config
  that no longer exists, which is exactly what the move to complex64 risked.
  Mandatory (``???``) values are allowed: they are how an unfilled HPO slot is
  meant to fail, loudly, at launch rather than silently with a stale number.

* **uniformity** — the Phase 2 ladder (``{nat,at}/legendre/d*c64``) must use one
  selection rule and one search space across every alpha and every arch. That
  uniformity *is* the deliverable of Phase 2 (issue B in the consistency
  register: the previous spirals sweeps used three stop_crits and five lr
  intervals, which is a sufficient mechanism for "interpolation stopped looking
  beneficial"). A future edit that tunes one arm individually should fail here.
"""
from functools import lru_cache
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from experiments.config import register
from experiments.resolvers import register_resolvers

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "configs"
_EXPERIMENTS = _CONFIG_DIR / "experiments"
_SPIRALS = _EXPERIMENTS / "spirals"
_TOYS = [_EXPERIMENTS / d for d in ("spirals", "circles", "moons")]


def _experiment_names(root: Path):
    """Hydra `+experiments=` names (config-relative, no suffix) under `root`."""
    return sorted(
        str(p.relative_to(_EXPERIMENTS).with_suffix("")) for p in root.rglob("*.yaml")
    )


@lru_cache(maxsize=None)
def _compose(name: str, with_hydra: bool = False):
    """Compose one experiment config. Cached: initialize_config_dir dominates the
    runtime and several checks read the same config. Callers must not mutate."""
    GlobalHydra.instance().clear()
    register()
    register_resolvers()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(
            config_name="config",
            overrides=[f"+experiments={name}"],
            return_hydra_config=with_hydra,
        )


# Phase 2 tree: the complex64 ladder, both regimes.
_PHASE2 = sorted(
    n for n in _experiment_names(_SPIRALS)
    if "legendre/d" in n and "c64" in n
)
_PHASE2_HPO = sorted(
    n for n in _experiment_names(_SPIRALS)
    if "legendre/hpo/" in n
)


_ALL_TOY = sorted(n for root in _TOYS for n in _experiment_names(root))


@pytest.mark.parametrize("name", _ALL_TOY)
def test_toy_experiment_config_composes(name):
    cfg = _compose(name)
    # resolve everything; ??? is legitimate (unfilled HPO slot), unresolvable is not
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


@pytest.mark.parametrize("name", _ALL_TOY)
def test_toy_configs_are_all_complex64(name):
    """The 2-D toys moved to complex64 wholesale; no real-dtype born config
    survives for them, so a config still pointing at one would fail to compose
    above. This pins the intent rather than the accident: every toy arch is c64,
    and c64 implies the overflow-safe contraction."""
    cfg = _compose(name)
    assert cfg.born.init_kwargs.dtype == "complex64", f"{name}: toys are complex64"
    assert cfg.born.accumulate is True, f"{name}: complex64 implies accumulate"


@pytest.mark.parametrize("name", _PHASE2)
def test_phase2_seed_sweep_is_uniform(name):
    cfg = _compose(name)
    is_at = "/at/" in name
    # `override /trainer/x: null` deletes the key rather than setting it to None,
    # so select() rather than attribute access.
    nll = OmegaConf.select(cfg, "trainer.nll")
    adv = OmegaConf.select(cfg, "trainer.adversarial")

    if is_at:
        t = adv
        assert nll is None, f"{name}: AT config must not carry an nll node"
        assert t.stop_crit == "rob", f"{name}: AT selects on rob"
        assert t.gen_on_clean is True, f"{name}: split objective is on for both AT alphas"
        # gen_on_clean raises at alpha=1; the AT arm must stay away from it.
        assert t.alpha < 1.0, f"{name}: gen_on_clean=True is invalid at alpha=1"
        assert cfg.descriptor == "at_warm", f"{name}: AT always warm-starts"
    else:
        t = nll
        assert adv is None, f"{name}: NAT config must not carry an AT node"
        assert t.stop_crit == "mixed_loss", (
            f"{name}: every NAT arm selects on mixed_loss (Phase 2, issue B) -- it "
            f"reduces to dis_loss at alpha=0 and gen_loss at alpha=1"
        )
        assert cfg.descriptor == "nat_cold", f"{name}: the whole ladder is cold-start"
        assert cfg.get("model_path") is None, f"{name}: cold start means no model_path"

    assert t.optimizer.kwargs.weight_decay == 0.0, (
        f"{name}: weight_decay is fixed at 0 for every arm, not tuned per arm"
    )
    assert t.batch_size == 256, f"{name}: batch size is shared across the ladder"


@pytest.mark.parametrize("name", _PHASE2)
def test_phase2_seed_sweep_has_five_seeds(name):
    cfg = _compose(name, with_hydra=True)
    if cfg.stage != "seed_sweep":
        pytest.skip("not a seed sweep")
    assert cfg.hydra.sweeper.params["tracking.seed"] == "range(1, 6)"


@pytest.mark.parametrize("name", _PHASE2_HPO)
def test_phase2_hpo_search_space_is_identical(name):
    """One lr interval for every arm. Narrowing it per alpha is issue C1."""
    cfg = _compose(name, with_hydra=True)
    params = cfg.hydra.sweeper.params
    lr_key = next(k for k in params if k.endswith("optimizer.kwargs.lr"))
    assert params[lr_key] == "tag(log, interval(1e-6, 1e-1))", (
        f"{name}: the lr search space must be the same for every alpha and arch"
    )
    if "/at/" in name:
        assert cfg.hydra.sweeper.direction == "maximize"   # rob
        assert params["trainer.adversarial.clean_weight"] == "interval(0.0, 0.5)"
    else:
        assert cfg.hydra.sweeper.direction == "minimize"   # mixed_loss
        assert len(params) == 1, f"{name}: NAT HPO sweeps lr only"


def test_phase2_tree_is_complete():
    """3 arches x (6 NAT alphas + 2 AT alphas) seed sweeps, plus 3 HPO configs."""
    sweeps = [n for n in _PHASE2 if "seed_sweep" in n]
    assert len(sweeps) == 24, f"expected 24 seed sweeps, found {len(sweeps)}"
    for arch in ("d4r3c64", "d6r4c64", "d10r6c64"):
        nat = [n for n in sweeps if f"spirals/nat/legendre/{arch}" in n]
        at = [n for n in sweeps if f"spirals/at/legendre/{arch}" in n]
        assert len(nat) == 6, f"{arch}: expected 6 NAT alphas, found {len(nat)}"
        assert len(at) == 2, f"{arch}: expected 2 AT alphas, found {len(at)}"
    assert len(_PHASE2_HPO) == 3, f"expected 3 HPO configs, found {len(_PHASE2_HPO)}"


def test_phase2_alpha_grid_is_the_agreed_one():
    """0, 1e-3, 1e-2, 1e-1, 0.5, 1 -- log-spaced decades plus 0.5."""
    alphas = sorted(
        _compose(n).trainer.nll.alpha
        for n in _PHASE2 if "nat/legendre/d10r6c64/seed_sweep" in n
    )
    assert alphas == [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
