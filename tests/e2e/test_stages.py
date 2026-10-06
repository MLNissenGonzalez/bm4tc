"""The stages on tiny studies: hpo -> select -> train, NAT then warm AT (D19, D21, D22).

configs/studies/tests/stages_{nat,at}.yaml; outputs and the hparams file go to a
scratch dir.
"""
import json

import optuna
import pytest
import yaml

from experiments import runs, stages
from experiments.runs import RunConflict, Study


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(runs, "HPARAMS", tmp_path / "hparams")
    return tmp_path


def _states(study, cell):
    return [t.state for t in stages._optuna_study(study, cell).get_trials()]


@pytest.mark.slow
def test_hpo_select_train(scratch):
    nat = Study("tests/stages_nat")
    with pytest.raises(LookupError, match="select tests/stages_nat"):
        nat.jobs()[0].compose()
    with pytest.raises(LookupError, match="run `hpo"):
        stages.select(nat)

    stages.hpo(nat, nat.cells())
    for cell in nat.cells():
        hpo_dir = nat.hpo_dir(cell)
        assert _states(nat, cell) == [optuna.trial.TrialState.COMPLETE] * 3
        assert sorted(p.name for p in hpo_dir.glob("t*")) == ["t0", "t1", "t2"]
        assert not list(hpo_dir.rglob("models/model"))           # D22: no checkpoints
        assert (hpo_dir / "t0" / "log.json").exists()

    # Relaunching finds every trial done.
    stages.hpo(nat, nat.cells())
    assert all(len(_states(nat, c)) == 3 for c in nat.cells())

    path = stages.select(nat)
    selected = yaml.safe_load(path.read_text())
    assert set(selected) == {c.name for c in nat.cells()}
    for cell in nat.cells():
        opt = stages._optuna_study(nat, cell)
        assert selected[cell.name] == opt.best_params
        assert 1e-3 <= selected[cell.name]["trainer.optimizer.kwargs.lr"] <= 1e-1

    for job in nat.jobs():
        stages.train(job)
        manifest = json.loads((job.run_dir / "run.json").read_text())
        assert manifest["config"]["trainer"]["optimizer"]["kwargs"]["lr"] == \
            selected[job.cell.name]["trainer.optimizer.kwargs.lr"]

    # AT HPO warm-starts every trial from the seed's alpha=0 NAT run.
    at = Study("tests/stages_at")
    stages.hpo(at, at.cells())
    stages.select(at)
    job = at.jobs()[0]
    stages.train(job)
    manifest = json.loads((job.run_dir / "run.json").read_text())
    assert manifest["init"]["run"] == str(nat.jobs()[0].run_dir)


@pytest.mark.slow
def test_changed_space_is_refused(scratch, monkeypatch):
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    monkeypatch.setattr(nat.cfg.hpo, "n_trials", 1)
    stages.hpo(nat, [cell])
    nat.cfg.hpo.space["trainer.optimizer.kwargs.lr"] = {"log": [1e-4, 1e-1]}
    with pytest.raises(RunConflict, match="--replace"):
        stages.hpo(nat, [cell])
    stages.hpo(nat, [cell], replace=True)
    assert len(list((scratch / "outputs/tests/stages_nat/.replaced").rglob("journal.log"))) == 1
    assert len(_states(nat, cell)) == 1


def test_stale_running_trials_fail(scratch):
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    stages.prepare_hpo(nat, cell)
    stages._optuna_study(nat, cell).ask()                         # a crashed worker's trial
    stages.prepare_hpo(nat, cell)
    assert _states(nat, cell) == [optuna.trial.TrialState.FAIL]


def test_suggest_parses_the_space():
    trial = optuna.trial.FixedTrial({"lr": 1e-3, "cw": 0.5, "n": 3, "opt": "adam"})
    assert stages.suggest(trial, "lr", {"log": [1e-5, 1e-1]}) == 1e-3
    assert stages.suggest(trial, "cw", [0.2, 1.0]) == 0.5
    assert stages.suggest(trial, "n", [1, 5]) == 3
    assert stages.suggest(trial, "opt", {"choice": ["adam", "sgd"]}) == "adam"
    with pytest.raises(ValueError):
        stages.suggest(trial, "lr", {"uniform": [0, 1]})
