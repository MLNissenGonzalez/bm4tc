"""Train the runs of a study (one Trainer for NAT and AT).

    python -m experiments.train <study> [--cell legendre/d3r40/a0.01] [--seed 3] [--replace]

Runs every (cell, seed) job of the study in order, or the selected ones. A
finished run with the same config is skipped, so a relaunch resumes; a finished
run with a different config stops the launch unless ``--replace`` (see
:meth:`experiments.runs.Job.claim`). Warm studies need their ``warm_from``
study trained first. Phase 5 adds the other verbs and parallel execution.
"""
import argparse
import datetime
import logging
from typing import Optional

import torch

from experiments.runs import Job, Study
from experiments.tracking import init_wandb, log_dataset_viz, make_logger
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine
from src.train import Trainer
from src.utils import set_seed

logger = logging.getLogger(__name__)
_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def train(job: Job, replace: bool = False) -> Optional[float]:
    """Train one job; returns its best validation objective, None if skipped."""
    cfg = job.compose()
    init = job.warm_source()
    config_hash = job.config_hash(cfg, init)
    if not job.claim(config_hash, replace):
        logger.info(f"{job.name}: finished with the same config, skipping.")
        return None

    launched = datetime.datetime.now().isoformat(timespec="seconds")
    log_file = logging.FileHandler(job.run_dir / "train.log")
    log_file.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(log_file)
    try:
        logger.info(f"{job.name}: training (config {config_hash}).")
        run = init_wandb(cfg, job)
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
        trainer.train(on_epoch_end=make_logger(job.run_dir, wandb_run=run),
                      output_dir=job.run_dir / "models")
        run.finish()

        if cfg.trainer.save and not (job.run_dir / "models" / "model").exists():
            raise RuntimeError(f"{job.name}: no checkpoint saved (no finite validation "
                               "objective); the run stays unfinished.")
        job.write_manifest(cfg, init, config_hash, launched, {
            "objective": trainer.best["objective"], "best_epoch": trainer.best_epoch,
        })
        return trainer.best["objective"]
    finally:
        logging.getLogger().removeHandler(log_file)
        log_file.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="a study under configs/studies/, e.g. spirals_nat")
    parser.add_argument("--cell", action="append",
                        help="only this cell (repeatable), e.g. legendre/d3r40/a0.01")
    parser.add_argument("--seed", type=int, action="append", help="only this seed (repeatable)")
    parser.add_argument("--replace", action="store_true",
                        help="archive finished runs whose config changed, then retrain them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=_FORMAT)

    study = Study(args.study)
    jobs = study.jobs()
    if args.cell:
        unknown = set(args.cell) - {c.name for c in study.cells()}
        if unknown:
            parser.error(f"no such cell(s) in {study.name}: {sorted(unknown)}")
        jobs = [j for j in jobs if j.cell.name in args.cell]
    if args.seed:
        jobs = [j for j in jobs if j.seed in args.seed]
    for job in jobs:
        train(job, replace=args.replace)


if __name__ == "__main__":
    main()
