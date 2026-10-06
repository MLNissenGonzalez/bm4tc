"""The pipeline stages of a study: hpo, select, train (D21, D22, D23).

    python -m experiments hpo    <study> [--cell C] [--replace]
    python -m experiments select <study>
    python -m experiments train  <study> [--cell C] [--seed S] [--replace]

``hpo`` runs one Optuna study per grid cell (TPE, no pruning) whose trials save
no checkpoint; ``select`` writes each cell's best trial (argmin
``objective/valid``, D8) to ``configs/hparams/<study>.yaml``; ``train`` runs the
seed runs with those hparams. Every stage resumes: finished trials and runs are
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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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
