"""The pipeline CLI: ``python -m experiments <verb> <study>`` (see experiments.stages)."""
import argparse
import logging

from experiments import stages
from experiments.runs import Study

VERBS = ("hpo", "select", "train")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m experiments",
                                     description=stages.__doc__.splitlines()[0])
    parser.add_argument("verb", choices=VERBS)
    parser.add_argument("study", help="a study under configs/studies/, e.g. spirals_nat")
    parser.add_argument("--cell", action="append",
                        help="only this cell (repeatable), e.g. legendre/d3r40/a0.01")
    parser.add_argument("--seed", type=int, action="append",
                        help="train: only this seed (repeatable)")
    parser.add_argument("--replace", action="store_true",
                        help="archive finished results whose config changed, then redo them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=stages._FORMAT)

    study = Study(args.study)
    cells = study.cells()
    if args.cell:
        unknown = set(args.cell) - {c.name for c in cells}
        if unknown:
            parser.error(f"no such cell(s) in {study.name}: {sorted(unknown)}")
        cells = [c for c in cells if c.name in args.cell]

    if args.verb == "hpo":
        stages.hpo(study, cells, replace=args.replace)
    elif args.verb == "select":
        stages.select(study)
    elif args.verb == "train":
        for job in study.jobs():
            if job.cell in cells and (not args.seed or job.seed in args.seed):
                stages.train(job, replace=args.replace)


if __name__ == "__main__":
    main()
