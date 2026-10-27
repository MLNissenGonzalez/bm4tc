"""The stages on tiny studies: hpo -> select -> train, NAT then warm AT (D19, D21, D22).

configs/studies/tests/stages_{nat,at}.yaml; outputs, hparams files included, go to a
scratch dir.
"""
import csv
import json
import logging

import optuna
import pytest
import yaml

from bm4tc.pipeline import runs, stages
from bm4tc.pipeline.runs import RunConflict, Study


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
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
        assert _states(nat, cell) == [optuna.trial.TrialState.COMPLETE] * nat.n_trials
        assert sorted(p.name for p in hpo_dir.glob("t*")) == [f"t{i}" for i in range(nat.n_trials)]
        assert not list(hpo_dir.rglob("models/model"))           # D22: no checkpoints
        assert (hpo_dir / "t0" / "log.json").exists()

    # Relaunching finds every trial done.
    stages.hpo(nat, nat.cells())
    assert all(len(_states(nat, c)) == nat.n_trials for c in nat.cells())

    # One cell at a time keeps the other cells' entries (D130).
    first, second = nat.cells()
    assert set(yaml.safe_load(stages.select(nat, [second]).read_text())) == {second.name}
    one_by_one = yaml.safe_load(stages.select(nat, [first]).read_text())
    path = stages.select(nat)
    selected = yaml.safe_load(path.read_text())
    assert selected == one_by_one
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

    # AT HPO warm-starts every trial from the seed's beta=0 NAT run.
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
    monkeypatch.setattr(nat.cfg.hpo, "trials_per_param", 1)
    stages.hpo(nat, [cell])
    nat.cfg.hpo.space["trainer.optimizer.kwargs.lr"] = {"log": [1e-4, 1e-1]}
    with pytest.raises(RunConflict, match="--replace"):
        stages.hpo(nat, [cell])
    stages.hpo(nat, [cell], replace=True)
    assert len(list((scratch / "outputs/tests/stages_nat/.replaced").rglob("journal.log"))) == 1
    assert len(_states(nat, cell)) == 1


def test_stale_running_trials_fail_and_are_run_again(scratch):
    """A trial left running by a stopped launch (Ctrl+C) is marked FAIL and
    interrupted, and does not count towards the budget (D95)."""
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    stages.prepare_hpo(nat, cell)
    stages._optuna_study(nat, cell).ask()                         # a stopped worker's trial
    stages.prepare_hpo(nat, cell)
    (trial,) = stages._optuna_study(nat, cell).get_trials()
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs[stages.INTERRUPTED]
    assert stages._counted([trial]) == []
    assert f"0/{nat.n_trials}" in stages.status(nat)


@pytest.mark.slow
def test_replace_starts_an_unchanged_hpo_over(scratch, monkeypatch):
    """--replace archives a journal even when its config is unchanged: one of
    crashed launches would otherwise be kept, its failures counted (D95)."""
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    monkeypatch.setattr(nat.cfg.hpo, "trials_per_param", 1)
    stages.hpo(nat, [cell])
    stages.hpo(nat, [cell])                                       # resumes: nothing to do
    assert len(_states(nat, cell)) == 1
    stages.hpo(nat, [cell], replace=True)
    assert len(list((scratch / "outputs/tests/stages_nat/.replaced").rglob("journal.log"))) == 1
    assert len(_states(nat, cell)) == 1


def test_suggest_parses_the_space():
    trial = optuna.trial.FixedTrial({"lr": 1e-3, "cw": 0.5, "n": 3, "opt": "adam"})
    assert stages.suggest(trial, "lr", {"log": [1e-5, 1e-1]}) == 1e-3
    assert stages.suggest(trial, "cw", [0.2, 1.0]) == 0.5
    assert stages.suggest(trial, "n", [1, 5]) == 3
    assert stages.suggest(trial, "opt", {"choice": ["adam", "sgd"]}) == "adam"
    with pytest.raises(ValueError):
        stages.suggest(trial, "lr", {"uniform": [0, 1]})


@pytest.mark.slow
def test_analyse_caches_each_part(scratch):
    nat = Study("tests/stages_nat")
    job = nat.jobs()[0]
    with pytest.raises(LookupError, match="train tests/stages_nat"):
        stages.analyse(job)
    nat.hparams_path.parent.mkdir(parents=True, exist_ok=True)
    nat.hparams_path.write_text(f"{job.cell.name}:\n  trainer.optimizer.kwargs.lr: 0.01\n")
    stages.train(job)
    metrics = stages.analyse(job)
    for k in ("acc/test", "rob/test/0.1", "detect/test/0/q10", "detect/test/0.1/q10",
              "purify/test/0.1/d0.1", "purify_gibbs/test/0.1/k1",
              "purify_gibbs/test/0/k1",
              "rob_ceiling/test/0.1", "rob_ceiling_sel/test/0.1", "rob_ceiling/pop/0.1",
              "rob_ceiling_sel/pop/0.1", "purify_adapt/test/0.1/d0.1", "rob_sel/test/0.1/q10",
              "rob_sel_adapt/test/0.1/q10", "purify_gibbs_adapt/test/0.1/k1",
              "margin/test", "gain/test", "gain_ends/test", "eps_star/test",
              "eps_star_below/test/0.1"):                          # D169
        assert k in metrics, k
    assert not any(k.startswith("purify_sel") for k in metrics)    # D122
    assert not any("_joint" in k or k.startswith("rob_adapt") for k in metrics)  # D149

    path = job.run_dir / "analysis.json"
    before = json.loads(path.read_text())
    assert stages.analyse(job) == metrics                          # all cached
    nat.cfg.analysis.purify_steps = 3                              # uq, adaptive
    stages.analyse(job)
    after = json.loads(path.read_text())
    assert after["clean"] == before["clean"] and after["gibbs"] == before["gibbs"]
    assert after["adaptive_sweep"] == before["adaptive_sweep"]     # D132: its own part
    assert after["adaptive_det"] == before["adaptive_det"]         # D156: likewise
    assert after["uq"]["hash"] != before["uq"]["hash"]
    nat.cfg.analysis.adaptive.seeds = [job.seed]                   # D129: it picks runs,
    nat.cfg.analysis.sweep_purify.adaptive_seeds = [job.seed]      # D132: so does this,
    assert stages.analysed(job) is not None                        # the hashes stay
    nat.cfg.analysis.sweep_purify.adaptive_seeds = []
    assert "purify_gibbs_adapt/test/0.1/k1" not in stages.analyse(job)
    assert "purify_adapt/test/0.1/d0.1" in stages.analyse(job)
    nat.cfg.analysis.adaptive.seeds = [job.seed + 1]
    assert not any("_adapt" in k for k in stages.analyse(job))
    nat.cfg.analysis.adaptive.seeds = None
    nat.cfg.analysis.sweep_purify.adaptive_seeds = None
    stages.analyse(job)
    before = json.loads(path.read_text())
    restarts = nat.cfg.analysis.attack.restarts                    # D135: the sweep
    nat.cfg.analysis.sweep_purify.adaptive_restarts = restarts     # purifier's attack
    nat.cfg.analysis.attack.restarts = restarts + 1                # keeps its count, so
    stages.analyse(job)                                            # its part stays cached;
    after = json.loads(path.read_text())                           # PGD changes (D149)
    assert after["adaptive_sweep"] == before["adaptive_sweep"]
    for part in ("uq", "gibbs", "adaptive", "adaptive_det"):
        assert after[part]["hash"] != before[part]["hash"], part
    before = after
    nat.cfg.analysis.attack.subsample = 50                         # the sweep parts attack
    stages.analyse(job)                                            # their own subsample
    after = json.loads(path.read_text())
    assert after["gibbs"] == before["gibbs"] and after["adaptive_sweep"] == before["adaptive_sweep"]
    assert after["uq"]["hash"] != before["uq"]["hash"]

    stages.collect(nat)                                            # one run analysed
    with open(scratch / "outputs/tests/stages_nat/results.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["beta"] == "0.0"
    assert "trainer.optimizer.kwargs.lr" in rows[0] and "rob/test/0.1" in rows[0]


@pytest.mark.slow
def test_run_in_parallel_then_status(scratch):
    """`run` with parallel units: two HPO workers share each cell's journal."""
    import os
    import subprocess
    import sys
    env = {**os.environ, "BM4TC_DATA_ROOT": str(scratch), "CUDA_VISIBLE_DEVICES": ""}
    nat = Study("tests/stages_nat")
    proc = subprocess.run([sys.executable, "-m", "bm4tc", "run", "tests/stages_nat",
                           "--per-gpu", "2"], cwd=runs.REPO, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert yaml.safe_load(nat.hparams_path.read_text()).keys() == {c.name for c in nat.cells()}
    for cell in nat.cells():
        assert len(_states(nat, cell)) == nat.n_trials              # not n_trials per worker
    with open(scratch / "outputs/tests/stages_nat/results.csv") as f:
        assert len(list(csv.DictReader(f))) == 2
    report = stages.status(nat)
    assert "legendre/d4r3/b0" in report and "6/6" in report and "failed" not in report


def test_select_runs_after_failed_hpo_workers():
    """A cell's select waits for its HPO workers to end, not to succeed: a worker
    stopped by a CUDA error leaves its trial to the others (D93), and select checks
    the budget itself (D99). train still needs its cell's select to succeed."""
    nat = Study("tests/stages_nat")
    todo = {u.name: u for u in stages.units(nat, ["hpo", "select", "train"], nat.cells(),
                                            workers=8)}
    hpo_names = [n for n in todo if n.startswith("hpo/")]
    assert len(hpo_names) == 3 * len(nat.cells())      # at most 3 workers (D157)
    for cell in nat.cells():                           # a select per cell (D130)
        select = todo[f"select/{cell.name}"]
        assert select.deps == [] and select.after == [f"hpo/{cell.name}/w{i}"
                                                      for i in range(3)]
    assert all(u.deps == [f"select/{n.removeprefix('train/').rsplit('/', 1)[0]}"]
               for n, u in todo.items() if n.startswith("train/"))


def test_pruner_follows_the_config():
    nat = Study("tests/stages_nat")                    # max_epoch 3, warmup 0.5 -> 2 epochs
    pruner = stages._pruner(nat, nat.cells()[0])
    assert isinstance(pruner, optuna.pruners.MedianPruner)
    assert (pruner._n_startup_trials, pruner._n_warmup_steps) == (6, 2)
    nat.cfg.hpo.pruning.enabled = False
    assert isinstance(stages._pruner(nat, nat.cells()[0]), optuna.pruners.NopPruner)
    assert nat.n_trials == 6                           # trials_per_param 6 x one hparam


@pytest.mark.slow
def test_pruned_trials_count_as_finished(scratch, monkeypatch):
    """A pruned trial stops at the validation the pruner rejects, counts towards
    n_trials, and is never selected (D81)."""
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    monkeypatch.setattr(stages, "_pruner",
                        lambda study, cell: optuna.pruners.ThresholdPruner(upper=-1e30))
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.INFO)                        # as the CLI runs
    try:
        stages.hpo(nat, [cell])
    finally:
        root.setLevel(level)
    assert _states(nat, cell) == [optuna.trial.TrialState.PRUNED] * nat.n_trials
    log = (nat.hpo_dir(cell) / "t0" / "train.log").read_text()
    assert "pruned at epoch 1" in log and "failed" not in log
    with pytest.raises(LookupError, match="no trial reached a finite"):
        stages.select(nat)


@pytest.mark.slow
def test_changed_pruning_is_refused(scratch, monkeypatch):
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    monkeypatch.setattr(nat.cfg.hpo, "trials_per_param", 1)
    stages.hpo(nat, [cell])
    nat.cfg.hpo.pruning.warmup = 0.3
    with pytest.raises(RunConflict, match="--replace"):
        stages.hpo(nat, [cell])


@pytest.mark.slow
@pytest.mark.parametrize("error", ["out of memory", "illegal memory access"])
def test_cuda_failures_are_retried(scratch, monkeypatch, error):
    """A trial that ends in a CUDA error is marked, not counted, and stops its
    worker; the next launch runs the budget to the end (D93)."""
    import torch
    nat = Study("tests/stages_nat")
    cell = nat.cells()[0]
    fit = stages._fit
    calls = []

    def first_runs_out_of_memory(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            if error == "out of memory":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20 MiB.")
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return fit(*args, **kwargs)

    monkeypatch.setattr(stages, "_fit", first_runs_out_of_memory)
    with pytest.raises(stages.CUDAFailure, match="trial 0 ended in a CUDA error"):
        stages.hpo(nat, [cell])
    assert f"0/{nat.n_trials}" in stages.status(nat)

    stages.hpo(nat, [cell])                            # the relaunch
    trials = stages._optuna_study(nat, cell).get_trials()
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert trials[0].user_attrs[stages.CUDA_FAILURE]
    assert [t.state for t in trials[1:]] == [optuna.trial.TrialState.COMPLETE] * nat.n_trials
    assert f"{nat.n_trials}/{nat.n_trials}" in stages.status(nat)
    stages.hpo(nat, nat.cells()[1:])                   # select of every cell needs each HPO
    selected = yaml.safe_load(stages.select(nat).read_text())
    assert cell.name in selected


@pytest.mark.slow
def test_cv_hpo_averages_the_folds_and_seeds_are_the_folds(scratch, monkeypatch):
    """use_ucr_split: cv (D148): a trial fits every fold, each as the seed run that
    validates on it, and returns the mean; seed s's run is then that fold's fit."""
    from bm4tc.pipeline.data import DataHandler
    load = DataHandler.load
    def ucr_load(self):                 # the first 150 spirals points play the UCR train split
        load(self)
        self.ucr_train_size = 150
    monkeypatch.setattr(DataHandler, "load", ucr_load)

    cv = Study("tests/stages_cv")
    cell, = cv.cells()
    stages.hpo(cv, [cell])
    best = lambda d: min(e["objective/valid"] for e in json.loads((d / "log.json").read_text())
                         if "objective/valid" in e)
    trials = stages._optuna_study(cv, cell).get_trials()
    assert len(trials) == cv.n_trials == 2
    for t in trials:
        folds = [cv.hpo_dir(cell) / f"t{t.number}" / f"f{s}" for s in range(1, 6)]
        assert t.value == pytest.approx(sum(map(best, folds)) / 5)
        assert len(t.user_attrs["best_epoch"]) == 5
    assert len({best(cv.hpo_dir(cell) / "t0" / f"f{s}") for s in range(1, 6)}) == 5   # five splits

    stages.select(cv)
    job = next(j for j in cv.jobs() if j.seed == 3)
    stages.train(job)
    chosen = min(trials, key=lambda t: t.value).number
    assert best(job.run_dir) == pytest.approx(best(cv.hpo_dir(cell) / f"t{chosen}" / "f3"), rel=1e-5)
