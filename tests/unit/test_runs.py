"""bm4tc/pipeline/runs.py: names (D14), grids, hparams, run.json (D17), warm starts (D19)."""
import json

import pytest

from bm4tc.pipeline import runs
from bm4tc.pipeline.runs import Cell, Job, RunConflict, Study, find_runs, parse_arch


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
    return tmp_path


# ── names ───────────────────────────────────────────────────────────────────

def test_parse_arch():
    assert parse_arch("d3r40") == (3, 40)
    with pytest.raises(ValueError):
        parse_arch("d3r40c64")


@pytest.mark.parametrize("cell, name", [
    (Cell("legendre", "d3r40", 0.0), "legendre/d3r40/a0"),
    (Cell("legendre", "d3r40", 0.001), "legendre/d3r40/a0.001"),
    (Cell("fourier", "d10r6", 1.0), "fourier/d10r6/a1"),
    (Cell("legendre", "d3r40", 0.01, 0.1), "legendre/d3r40/a0.01/eps0.1"),
])
def test_cell_names(cell, name):
    assert cell.name == name


def test_run_dir_is_study_rooted(data_root):
    job = Study("spirals_capacity").jobs()[0]
    assert job.run_dir == data_root / "outputs/spirals_capacity/legendre/d4r3/a0/s1"
    assert job.wandb() == {"group": "spirals_capacity/legendre/d4r3/a0", "name": "s1",
                           "job_type": "train"}


# ── grids ───────────────────────────────────────────────────────────────────

def test_defaults_fill_the_grid():
    study = Study("spirals_nat")
    assert study.seeds() == [1, 2, 3, 4, 5]                      # D29
    assert [c.alpha for c in study.cells()] == [0, 1e-3, 1e-2, 1e-1, 0.5, 1]  # D45
    assert all(c.eps is None for c in study.cells())             # NAT has no radius


def test_at_cells_carry_the_radius():
    cells = Study("spirals_at").cells()
    assert [(c.alpha, c.eps) for c in cells] == [(0.0, 0.1), (0.01, 0.1)]


def test_embedding_settings_apply_per_cell():
    study = Study("spirals_embedding")
    hermite = next(c for c in study.cells() if c.embedding == "hermite")
    legendre = next(c for c in study.cells() if c.embedding == "legendre")
    assert Job(study, hermite, 1).compose(hparams={}).born.init_kwargs.init_method == "canonical"
    assert Job(study, legendre, 1).compose(hparams={}).born.init_kwargs.init_method == "randn_eye"


def test_cell_sets_the_model_and_alpha():
    study = Study("mnist12_at")
    cfg = Job(study, Cell("legendre", "d3r20", 0.01, 0.1), 2).compose(hparams={})
    assert (cfg.born.init_kwargs.in_dim, cfg.born.init_kwargs.bond_dim) == (3, 20)
    assert cfg.trainer.alpha == 0.01
    assert list(cfg.trainer.evasion.eps_rel) == [0.1]
    assert cfg.tracking.seed == 2
    assert cfg.dataset.name == "mnist_full_r12"


def test_at_must_be_warm(tmp_path, monkeypatch):
    (tmp_path / "studies").mkdir()
    (tmp_path / "studies" / "bad.yaml").write_text("dataset: spirals\nregime: at\ngrid: {arch: [d4r3]}\n")
    (tmp_path / "defaults.yaml").write_text((runs.CONFIGS / "defaults.yaml").read_text())
    monkeypatch.setattr(runs, "CONFIGS", tmp_path)
    with pytest.raises(ValueError, match="always warm"):
        Study("bad")


# ── hparams (D21, D34) ──────────────────────────────────────────────────────

def test_missing_hparams_fail_loudly():
    study = Study("spirals_at")
    with pytest.raises(LookupError, match="select spirals_at"):
        study.jobs()[0].compose()


def test_hparams_are_read_per_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "HPARAMS", tmp_path)
    study = Study("spirals_at")
    cell = study.cells()[1]
    (tmp_path / "spirals_at.yaml").write_text(  # as `select spirals_at` writes it
        f"{cell.name}:\n  trainer.optimizer.kwargs.lr: 3.0e-3\n"
        "  trainer.clean_weight: 0.4\n  extra: 1\n")
    assert study.hparams(cell) == {"trainer.clean_weight": 0.4,
                                   "trainer.optimizer.kwargs.lr": 3e-3}
    cfg = Job(study, cell, 1).compose()
    assert cfg.trainer.optimizer.kwargs.lr == 3e-3 and cfg.trainer.clean_weight == 0.4


def test_hparams_from_inherits_the_matching_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "HPARAMS", tmp_path)
    study = Study("jem_mnist12_at")
    cell = next(c for c in study.cells() if c.alpha == 1e-2 and c.arch == "d3r20")
    (tmp_path / "jem_mnist12_nat.yaml").write_text(
        "raw/d3r20/a0.01:\n  trainer.optimizer.kwargs.lr: 1.0e-4\n  jem.energy_l2: 1.0e-5\n"
        "  jem.sampler.step_size: 0.02\n  jem.sampler.noise_std: 0.003\n"
        "  jem.sampler.num_steps: 40\n")
    (tmp_path / "jem_mnist12_at.yaml").write_text(
        f"{cell.name}:\n  trainer.optimizer.kwargs.lr: 3.0e-4\n  trainer.clean_weight: 0.5\n")
    cfg = Job(study, cell, 1).compose()
    assert (cfg.jem.sampler.step_size, cfg.jem.sampler.noise_std, cfg.jem.sampler.num_steps) \
        == (0.02, 0.003, 40)
    assert cfg.trainer.optimizer.kwargs.lr == 3e-4      # its own, not NAT's
    assert cfg.jem.energy_l2 == 1e-4                    # not named in hparams_from
    assert cfg.model == "jem" and cfg.jem.model.match_bond_dim == 20


def test_jem_study_needs_the_raw_embedding(tmp_path, monkeypatch):
    (tmp_path / "studies").mkdir()
    (tmp_path / "defaults.yaml").write_text((runs.CONFIGS / "defaults.yaml").read_text())
    (tmp_path / "studies" / "x.yaml").write_text("dataset: mnist12\nmodel: jem\nregime: nat\n"
                                                 "grid: {arch: [d3r20]}\n")
    monkeypatch.setattr(runs, "CONFIGS", tmp_path)
    with pytest.raises(ValueError, match="raw"):
        Study("x")


def test_study_without_hpo_needs_no_hparams():
    assert Study("tests/seam_nat").hparams(Cell("legendre", "d4r3", 0.0)) == {}


# ── run.json, skip / refuse / archive ───────────────────────────────────────

def _finish(job, config_hash="h1"):
    assert job.claim(config_hash)
    (job.run_dir / "models").mkdir()
    (job.run_dir / "models" / "model").write_text("checkpoint")
    job.write_manifest(job.compose(), None, config_hash, "2026-10-06T00:00:00", {"objective": 1.0})


def test_unfinished_run_is_restarted():
    job = Study("tests/seam_nat").jobs()[0]
    job.run_dir.mkdir(parents=True)
    (job.run_dir / "partial").write_text("x")
    assert job.claim("h1")
    assert not (job.run_dir / "partial").exists()


def test_finished_run_with_same_config_is_skipped():
    job = Study("tests/seam_nat").jobs()[0]
    _finish(job)
    assert job.claim("h1") is False
    assert (job.run_dir / "run.json").exists()


def test_finished_run_with_other_config_is_never_overwritten():
    job = Study("tests/seam_nat").jobs()[0]
    _finish(job)
    with pytest.raises(RunConflict, match="--replace"):
        job.claim("h2")
    assert json.loads((job.run_dir / "run.json").read_text())["config_hash"] == "h1"


def test_replace_archives_the_old_run(data_root):
    job = Study("tests/seam_nat").jobs()[0]
    _finish(job)
    assert job.claim("h2", replace=True)
    archived = list((data_root / "outputs/tests/seam_nat/.replaced").rglob("run.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["config_hash"] == "h1"
    assert not (job.run_dir / "run.json").exists()


def test_config_hash_follows_config_and_warm_start():
    job = Study("tests/seam_nat").jobs()[0]
    cfg = job.compose()
    h = job.config_hash(cfg, None)
    assert job.config_hash(cfg, None) == h
    assert job.config_hash(cfg, {"run": "x", "hash": "abc"}) != h
    cfg.trainer.max_epoch += 1
    assert job.config_hash(cfg, None) != h
    cfg.trainer.max_epoch -= 1
    cfg.tracking.mode = "online"  # W&B settings do not make a different run
    assert job.config_hash(cfg, None) == h


# ── warm starts (D19) ───────────────────────────────────────────────────────

def test_warm_start_finds_the_nat_run_by_manifest():
    nat = Study("tests/seam_nat").jobs()[0]
    at = Study("tests/seam_at").jobs()[0]
    with pytest.raises(LookupError, match="train tests/seam_nat first"):
        at.warm_source()
    _finish(nat)
    assert at.warm_source() == {"run": str(nat.run_dir), "hash": "h1"}
    assert [d for d, _ in find_runs(regime="nat", seed=42)] == [nat.run_dir]


def test_archived_runs_are_not_found():
    nat = Study("tests/seam_nat").jobs()[0]
    _finish(nat)
    nat.claim("h2", replace=True)
    assert find_runs(study="tests/seam_nat") == []


# ── prune (D22) ─────────────────────────────────────────────────────────────

def _finished_study(name, seeds):
    from bm4tc.pipeline import stages  # noqa: F401  (imports torch; only here)
    study = Study(name)
    study.cfg.seeds = seeds
    for job in study.jobs():
        _finish(job)
    return study


def test_prune_keeps_results_and_one_seed_per_cell(data_root):
    from bm4tc.pipeline import stages
    study = _finished_study("tests/seam_nat", [1, 2, 3])
    plan = stages.prune_plan(study, "keep-one")
    assert [p.parent.name for p in plan] == ["s2", "s3"]
    stages.prune(study, "keep-one", yes=True)
    s1, s2, _ = study.jobs()
    assert s1.checkpoint.exists() and not s1.pruned
    assert s2.pruned and (s2.run_dir / "run.json").exists()
    assert stages.prune_plan(study, "all") == [s1.run_dir / "models"]


def test_pruned_run_is_skipped_unless_replaced(data_root):
    from bm4tc.pipeline import stages
    study = _finished_study("tests/seam_nat", [42])
    stages.prune(study, "all", yes=True)
    job = study.jobs()[0]
    assert job.claim("h1") is False                               # results are kept
    with pytest.raises(LookupError, match="was pruned"):
        Study("tests/seam_at").jobs()[0].warm_source()
    assert job.claim("h1", replace=True)                          # retrain it
    assert len(list((data_root / "outputs/tests/seam_nat/.replaced").rglob("run.json"))) == 1


def test_prune_old_deletes_the_archive(data_root):
    from bm4tc.pipeline import stages
    study = _finished_study("tests/seam_nat", [42])
    study.jobs()[0].claim("h2", replace=True)
    assert stages.prune_plan(study, "old") == [data_root / "outputs/tests/seam_nat/.replaced"]
    stages.prune(study, "old", yes=True)
    assert stages.prune_plan(study, "old") == []
