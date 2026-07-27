from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from baselines.jem.hpo_export import patch_seed_sweep_config, target_for_alpha


def test_alpha_targets_match_seed_sweep_names():
    assert target_for_alpha(0.01) == "a001"
    assert target_for_alpha(0.1) == "a01"
    assert target_for_alpha(1.0) == "a1"


def test_patch_seed_sweep_config_preserves_unrelated_yaml():
    original = """hydra:
  sweeper:
    params:
      tracking.seed: range(1, 6)
      trainer.optimizer.kwargs.lr: 0.001  # selected by HPO
      sampler.num_steps: 20
"""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "a01.yaml"
        path.write_text(original)
        updated = patch_seed_sweep_config(
            path,
            {
                "trainer.optimizer.kwargs.lr": 2.5e-4,
                "sampler.num_steps": 40,
            },
        )
        parsed = yaml.safe_load(path.read_text())
        params = parsed["hydra"]["sweeper"]["params"]
        assert updated == ["trainer.optimizer.kwargs.lr", "sampler.num_steps"]
        assert params["tracking.seed"] == "range(1, 6)"
        assert params["trainer.optimizer.kwargs.lr"] == 2.5e-4
        assert params["sampler.num_steps"] == 40
        assert "# selected by HPO" in path.read_text()


def test_every_hpo_parameter_has_a_seed_sweep_destination():
    config_root = (
        Path(__file__).resolve().parents[3]
        / "baselines"
        / "jem"
        / "configs"
        / "experiment"
    )
    pairings = {
        "a0": ("a0",),
        "pretrained": ("a001", "a01", "a02", "a05", "a1"),
        "at": ("at",),
    }
    for hpo_name, seed_names in pairings.items():
        hpo = yaml.safe_load((config_root / "hpo" / f"{hpo_name}.yaml").read_text())
        optimized = set(hpo["hydra"]["sweeper"]["params"])
        for seed_name in seed_names:
            seed = yaml.safe_load(
                (config_root / "seed_sweep" / f"{seed_name}.yaml").read_text()
            )
            destinations = set(seed["hydra"]["sweeper"]["params"])
            assert optimized <= destinations
