"""The pipeline stages of a study: hpo, select, train (D21, D22, D23).

    python -m experiments hpo    <study> [--cell C] [--replace]
    python -m experiments select <study>
    python -m experiments train  <study> [--cell C] [--seed S] [--replace]
    python -m experiments analyse <study> [--cell C] [--seed S]

``hpo`` runs one Optuna study per grid cell (TPE, no pruning) whose trials save
no checkpoint; ``select`` writes each cell's best trial (argmin
``objective/valid``, D8) to ``configs/hparams/<study>.yaml``; ``train`` runs the
seed runs with those hparams; ``analyse`` evaluates each finished run on test
(:mod:`experiments.analyse`) into its ``analysis.json`` and collects the study's
``results.csv``. Every stage resumes: finished trials and runs are
kept, and a changed config is refused unless ``--replace`` archives the old
results (:meth:`experiments.runs.Job.claim`). Warm studies need their
``warm_from`` study trained first.
"""
import contextlib
import datetime
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import optuna
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from experiments.runs import Cell, Job, RunConflict, Study, _git_version, archive
from experiments.tracking import init_wandb, log_dataset_viz, make_logger
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine
from src.train import Trainer
from src.utils import set_seed

logger = logging.getLogger(__name__)
_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


@contextlib.contextmanager
def _log_to(path: Path) -> Iterator[None]:
    """Copy every log record to ``path`` while the block runs."""
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(handler)
    try:
        yield
    except Exception:
        logger.exception("failed")
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _fit(cfg: DictConfig, init: Optional[Dict[str, str]], run_dir: Path,
         names: Dict[str, str]) -> Trainer:
    """Train one model in ``run_dir`` (log.json; the checkpoint in models/ if
    ``trainer.save``); cold, or warm from the run ``init`` names."""
    run = init_wandb(cfg, run_dir, names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    set_seed(cfg.tracking.seed)

    if init is not None:
        source = f"{init['run']}/models/model"
        logger.info(f"Warm start from {source}")
        cbm = ConditionalBornMachine.load(source, accumulate=cfg.born.accumulate)
        cbm.to(device)
    else:
        cbm = ConditionalBornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(cbm)
    log_dataset_viz(datahandler)

    trainer = Trainer(cbm, cfg.trainer, datahandler, device)
    trainer.train(on_epoch_end=make_logger(run_dir, wandb_run=run),
                  output_dir=run_dir / "models")
    run.finish()
    return trainer


# ── train ───────────────────────────────────────────────────────────────────

def train(job: Job, replace: bool = False) -> Optional[float]:
    """Train one seed run; returns its best validation objective, None if skipped."""
    cfg = job.compose()
    init = job.warm_source()
    config_hash = job.config_hash(cfg, init)
    if not job.claim(config_hash, replace):
        logger.info(f"{job.name}: finished with the same config, skipping.")
        return None

    launched = datetime.datetime.now().isoformat(timespec="seconds")
    with _log_to(job.run_dir / "train.log"):
        logger.info(f"{job.name}: training (config {config_hash}).")
        trainer = _fit(cfg, init, job.run_dir, job.wandb())
        if cfg.trainer.save and not (job.run_dir / "models" / "model").exists():
            raise RuntimeError(f"{job.name}: no checkpoint saved (no finite validation "
                               "objective); the run stays unfinished.")
        job.write_manifest(cfg, init, config_hash, launched, {
            "objective": trainer.best["objective"], "best_epoch": trainer.best_epoch,
        })
    return trainer.best["objective"]


# ── hpo ─────────────────────────────────────────────────────────────────────

optuna.logging.set_verbosity(optuna.logging.WARNING)  # trials log their own results
TPE_SEED = 42
TPE_STARTUP_TRIALS = 6
_COUNTED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.RUNNING)


def suggest(trial: optuna.Trial, key: str, spec: Any) -> Any:
    """Sample one value of the HPO space (configs/defaults.yaml ``hpo.space``):
    ``{log: [lo, hi]}`` log-uniform, ``[lo, hi]`` uniform (integers if both are),
    ``{choice: [...]}`` categorical."""
    if isinstance(spec, dict) and set(spec) == {"log"}:
        lo, hi = spec["log"]
        return trial.suggest_float(key, float(lo), float(hi), log=True)
    if isinstance(spec, dict) and set(spec) == {"choice"}:
        return trial.suggest_categorical(key, list(spec["choice"]))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            return trial.suggest_int(key, lo, hi)
        return trial.suggest_float(key, float(lo), float(hi))
    raise ValueError(f"HPO space {key}: {spec!r} is not {{log: [lo, hi]}}, [lo, hi] "
                     "or {choice: [...]}")


def _space(study: Study) -> Dict[str, Any]:
    return OmegaConf.to_container(study.cfg.hpo.space, resolve=True)


def _optuna_study(study: Study, cell: Cell, worker: int = 0) -> optuna.Study:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    path = study.hpo_dir(cell) / "journal.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=cell.name,
        storage=JournalStorage(JournalFileBackend(str(path))),
        # Workers share the journal, so each samples from its own seed.
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED + worker,
                                           n_startup_trials=TPE_STARTUP_TRIALS),
        direction="minimize",
        load_if_exists=True,
    )


def _hpo_hash(study: Study, cell: Cell) -> str:
    """What makes two HPOs of a cell the same: the trial config without hparams,
    the warm start and the search space."""
    job = study.hpo_job(cell)
    base = job.config_hash(job.compose(hparams={}), job.warm_source())
    space = json.dumps(_space(study), sort_keys=True)
    return hashlib.sha256(f"{base}{space}".encode()).hexdigest()[:16]


def prepare_hpo(study: Study, cell: Cell, replace: bool = False) -> None:
    """Make the cell's HPO ready to (re)start, before any worker runs.

    A journal started under another config or space raises :class:`RunConflict`,
    unless ``replace`` archives it. Trials left RUNNING by a crashed launch are
    marked FAIL (so: one launch per study at a time).
    """
    hpo_dir = study.hpo_dir(cell)
    if (hpo_dir / "journal.log").exists():
        stored = _optuna_study(study, cell).user_attrs.get("config_hash")
        if stored != _hpo_hash(study, cell):
            if not replace:
                raise RunConflict(
                    f"{hpo_dir} holds an HPO under another config or search space. "
                    f"Pass --replace to move it to .replaced/ and start over.")
            archive(study.name, hpo_dir)
    opt = _optuna_study(study, cell)
    opt.set_user_attr("config_hash", _hpo_hash(study, cell))
    for t in opt.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        logger.warning(f"{study.name}/{cell.name}: trial {t.number} was left running; "
                       "marking it failed.")
        opt.tell(t.number, state=optuna.trial.TrialState.FAIL)


def hpo_worker(study: Study, cell: Cell, worker: int = 0) -> None:
    """Run trials of the cell until ``hpo.n_trials`` are finished or running.
    Several workers may run at once on the same journal."""
    job = study.hpo_job(cell)
    init = job.warm_source()
    space = _space(study)
    opt = _optuna_study(study, cell, worker)

    def objective(trial: optuna.Trial) -> float:
        hparams = {key: suggest(trial, key, spec) for key, spec in space.items()}
        cfg = job.compose(hparams=hparams)
        cfg.trainer.save = False  # D22: trials keep curves only
        trial_dir = study.hpo_dir(cell) / f"t{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with _log_to(trial_dir / "train.log"):
            logger.info(f"{job.study.name}/{cell.name}: trial {trial.number} {hparams}")
            trainer = _fit(cfg, init, trial_dir, job.wandb(trial=trial.number))
        trial.set_user_attr("best_epoch", trainer.best_epoch)
        logger.info(f"{job.study.name}/{cell.name}: trial {trial.number} objective/valid "
                    f"{trainer.best['objective']:.6g} (epoch {trainer.best_epoch})")
        return trainer.best["objective"]  # inf if no validation was finite

    while len(opt.get_trials(deepcopy=False, states=_COUNTED)) < study.cfg.hpo.n_trials:
        opt.optimize(objective, n_trials=1, catch=(Exception,))


def hpo(study: Study, cells: List[Cell], replace: bool = False) -> None:
    if study.cfg.hpo is None:
        raise ValueError(f"{study.name} has no HPO (hpo: null): it fixes every hparam")
    for cell in cells:
        prepare_hpo(study, cell, replace)
        hpo_worker(study, cell)


# ── select ──────────────────────────────────────────────────────────────────

def select(study: Study) -> Path:
    """Write every cell's best trial to ``configs/hparams/<study>.yaml`` (D21, D34).
    Refuses while a cell's HPO is unfinished or has no finite trial."""
    if study.cfg.hpo is None:
        raise ValueError(f"{study.name} has no HPO (hpo: null): nothing to select")
    n_trials = study.cfg.hpo.n_trials
    lines = [f"# Selected by `select {study.name}`: per cell, the HPO trial with the lowest",
             "# objective/valid (D8, D21). Written by select only; do not edit.",
             f"# {datetime.datetime.now().isoformat(timespec='seconds')}, git {_git_version()}"]
    for cell in study.cells():
        if not (study.hpo_dir(cell) / "journal.log").exists():
            raise LookupError(f"{study.name}/{cell.name}: no HPO; run `hpo {study.name}` first")
        trials = _optuna_study(study, cell).get_trials(deepcopy=False)
        states = [t.state for t in trials]
        finished = sum(s in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
                       for s in states)
        if finished < n_trials or optuna.trial.TrialState.RUNNING in states:
            raise LookupError(f"{study.name}/{cell.name}: HPO unfinished ({finished} of "
                              f"{n_trials} trials); run `hpo {study.name}`")
        complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
                    and math.isfinite(t.value)]
        if not complete:
            raise LookupError(f"{study.name}/{cell.name}: no trial reached a finite "
                              f"objective/valid; see {study.hpo_dir(cell)}/t*/train.log")
        best = min(complete, key=lambda t: t.value)
        lines.append(f"{cell.name}:  # trial {best.number} of {len(trials)}, "
                     f"objective/valid {best.value:.6g}")
        for key in sorted(best.params):
            value = yaml.safe_dump(best.params[key], default_flow_style=True).strip()
            lines.append(f"  {key}: {value.removesuffix('...').strip()}")
    path = study.hparams_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")
    return path


# ── analyse ─────────────────────────────────────────────────────────────────

def analyse(job: Job) -> Dict[str, float]:
    """Analyse one finished run; returns its metrics.

    ``{run}/analysis.json`` holds each part's results under the hash of what they
    depend on (:func:`experiments.analyse.part_hash`). A part with a matching hash
    is kept, so a failed or extended analysis resumes; a part the settings no
    longer ask for stays in the file (Gibbs is expensive) but is not returned.
    """
    from experiments import analyse as parts_

    manifest_path = job.run_dir / "run.json"
    if not manifest_path.exists():
        raise LookupError(f"{job.name}: not trained; run `train {job.study.name}` first")
    run_hash = json.loads(manifest_path.read_text())["config_hash"]
    analysis, budgets = job.study.cfg.analysis, list(job.study.cfg.budgets)
    path = job.run_dir / "analysis.json"
    stored = json.loads(path.read_text()) if path.exists() else {}

    todo = {name: part for name, part in parts_.parts(analysis).items()
            if stored.get(name, {}).get("hash") != parts_.part_hash(name, analysis, budgets, run_hash)}
    if todo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with _log_to(job.run_dir / "analyse.log"):
            cbm, datahandler = parts_.load(job.run_dir, analysis.batch_size, device)
            for name, part in todo.items():
                logger.info(f"{job.name}: analysis part {name}")
                results = parts_.run_part(part, cbm, datahandler, analysis, budgets, device)
                stored[name] = {"hash": parts_.part_hash(name, analysis, budgets, run_hash),
                                "results": results}
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(stored, indent=2))
                tmp.replace(path)  # a part is written whole or not at all
            del cbm
    else:
        logger.info(f"{job.name}: analysis up to date, skipping.")
    return analysed(job)


def analysed(job: Job) -> Optional[Dict[str, float]]:
    """The run's metrics if every part the study asks for is up to date, else None."""
    from experiments import analyse as parts_

    manifest, path = job.run_dir / "run.json", job.run_dir / "analysis.json"
    if not (manifest.exists() and path.exists()):
        return None
    run_hash = json.loads(manifest.read_text())["config_hash"]
    analysis, budgets = job.study.cfg.analysis, list(job.study.cfg.budgets)
    stored = json.loads(path.read_text())
    out = {}
    for name in parts_.parts(analysis):
        part = stored.get(name, {})
        if part.get("hash") != parts_.part_hash(name, analysis, budgets, run_hash):
            return None
        out.update(part["results"])
    return out


def collect(study: Study) -> Path:
    """Write ``outputs/{study}/results.csv``: one row per run whose analysis is up
    to date (identity, selected hparams, best objective/valid, metrics), plus the
    analysis settings next to it (D20)."""
    import pandas as pd
    from experiments.runs import outputs_root

    space = sorted(study.cfg.hpo.space) if study.cfg.hpo is not None else []
    rows, missing = [], []
    for job in study.jobs():
        metrics = analysed(job)
        if metrics is None:
            missing.append(job.name)
            continue
        manifest = json.loads((job.run_dir / "run.json").read_text())
        config = OmegaConf.create(manifest["config"])
        rows.append({**manifest["identity"],
                     **{k: OmegaConf.select(config, k) for k in space},
                     "objective/valid": manifest["result"]["objective"],
                     **metrics})
    root = outputs_root() / study.name
    path = root / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    settings = {"budgets": list(study.cfg.budgets),
                "analysis": OmegaConf.to_container(study.cfg.analysis, resolve=True)}
    (root / "analysis.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    logger.info(f"Wrote {path} ({len(rows)} runs)")
    if missing:
        logger.warning(f"{len(missing)} runs not (fully) analysed, left out: {missing}")
    return path


# ── launching: units, lock, status (D38) ────────────────────────────────────

STAGES = ("hpo", "select", "train", "analyse")


def logs_dir(study: Study) -> Path:
    from experiments.runs import outputs_root
    return outputs_root() / study.name / ".logs"


def units(study: Study, stages: Sequence[str], cells: List[Cell],
          seeds: Optional[List[int]] = None, workers: int = 1,
          replace: bool = False) -> List["Unit"]:
    """The units of the given stages of a study, as subprocesses of ``_unit``.

    select waits for every HPO worker, train for select, analyse for its train,
    each only when that stage is part of the launch. Studies without HPO have no
    hpo or select units.
    """
    from experiments.executor import Unit

    def unit(name, *args, deps=()):
        argv = [sys.executable, "-m", "experiments", "_unit", *args]
        return Unit(name, argv, logs_dir(study) / f"{name}.log", list(deps))

    has_hpo = study.cfg.hpo is not None
    jobs = [j for j in study.jobs() if j.cell in cells and (not seeds or j.seed in seeds)]
    out, hpo_names = [], []
    if "hpo" in stages and has_hpo:
        n = min(workers, study.cfg.hpo.n_trials)
        for cell in cells:
            for w in range(n):
                hpo_names.append(f"hpo/{cell.name}/w{w}")
                out.append(unit(hpo_names[-1], "hpo", study.name, "--cell", cell.name,
                                "--worker", str(w)))
    select_deps = []
    if "select" in stages and has_hpo:
        out.append(unit("select", "select", study.name, deps=hpo_names))
        select_deps = ["select"]
    for job in jobs:
        name = f"{job.cell.name}/s{job.seed}"
        cell_args = ("--cell", job.cell.name, "--seed", str(job.seed))
        if "train" in stages:
            out.append(unit(f"train/{name}", "train", study.name, *cell_args,
                            *(["--replace"] if replace else []), deps=select_deps))
        if "analyse" in stages:
            deps = [f"train/{name}"] if "train" in stages else []
            out.append(unit(f"analyse/{name}", "analyse", study.name, *cell_args, deps=deps))
    return out


def run_unit(study: Study, kind: str, cell: Optional[Cell], seed: Optional[int],
             worker: int = 0, replace: bool = False) -> None:
    """One unit, in this process (``python -m experiments _unit ...``)."""
    if kind == "hpo":
        hpo_worker(study, cell, worker)
    elif kind == "select":
        select(study)
    elif kind in ("train", "analyse"):
        job = Job(study, cell, seed)
        train(job, replace) if kind == "train" else analyse(job)
    else:
        raise ValueError(f"unknown unit kind {kind!r}")


@contextlib.contextmanager
def study_lock(study: Study) -> Iterator[None]:
    """One launch per study at a time (HPO marks stale trials failed on start)."""
    path = logs_dir(study) / "lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        pid = int(path.read_text() or 0)
        if _alive(pid):
            raise RuntimeError(f"{study.name} is already being run by process {pid} "
                               f"(lock {path}); see `status {study.name}`")
    path.write_text(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return pid > 0


def launch(study: Study, stages: Sequence[str], cells: List[Cell],
           seeds: Optional[List[int]] = None, gpus: Optional[List[str]] = None,
           per_gpu: int = 1, replace: bool = False) -> Dict[str, str]:
    """Run the stages of a study through the executor; returns each unit's state.
    HPO journals are prepared here first, and results.csv is collected last."""
    from experiments.executor import execute

    with study_lock(study):
        if "hpo" in stages and study.cfg.hpo is not None:
            for cell in cells:
                prepare_hpo(study, cell, replace)
        todo = units(study, stages, cells, seeds, workers=len(gpus or [None]) * per_gpu,
                     replace=replace)
        states = execute(todo, gpus, per_gpu, state_file=logs_dir(study) / "status.json")
        if "analyse" in stages:
            collect(study)
    return states


def status(study: Study) -> str:
    """Per cell: HPO trials, selection, trained and analysed seeds; then every
    running, failed or skipped unit of the last launch with its log."""
    seeds = study.seeds()
    has_hpo = study.cfg.hpo is not None
    lines = [f"{study.name}: {len(study.cells())} cells x {len(seeds)} seeds"]
    header = f"{'cell':32} {'hpo':>7} {'sel':>4} {'trained':>8} {'analysed':>9}"
    lines.append(header)
    selected = {}
    if has_hpo and study.hparams_path.exists():
        selected = OmegaConf.to_container(OmegaConf.load(study.hparams_path))
    for cell in study.cells():
        hpo_col, sel_col = "-", "-"
        if has_hpo:
            journal = study.hpo_dir(cell) / "journal.log"
            n = 0
            if journal.exists():
                n = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
                        for t in _optuna_study(study, cell).get_trials(deepcopy=False))
            hpo_col = f"{n}/{study.cfg.hpo.n_trials}"
            sel_col = "yes" if cell.name in selected else "no"
        jobs = [Job(study, cell, s) for s in seeds]
        trained = sum((j.run_dir / "run.json").exists() for j in jobs)
        done = sum(analysed(j) is not None for j in jobs)
        lines.append(f"{cell.name:32} {hpo_col:>7} {sel_col:>4} "
                     f"{trained:>5}/{len(seeds):<2} {done:>6}/{len(seeds):<2}")

    state_file = logs_dir(study) / "status.json"
    if state_file.exists():
        last = json.loads(state_file.read_text())
        live = _alive(last["pid"])
        counts: Dict[str, int] = {}
        for s in last["units"].values():
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        lines.append("")
        lines.append(f"last launch (process {last['pid']}, "
                     f"{'running' if live else 'ended'}): "
                     + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
        for name, s in last["units"].items():
            state = s["state"]
            if state == "running" and not live:
                state = "interrupted"
            if state in ("running", "failed", "interrupted"):
                lines.append(f"  {state:11} {name}  {s['log']}")
    return "\n".join(lines)
