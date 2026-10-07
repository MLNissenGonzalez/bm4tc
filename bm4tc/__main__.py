"""The pipeline CLI: ``python -m bm4tc <verb> <study>`` (see bm4tc.pipeline.stages).

    hpo | train | analyse   that stage, through the job pool
    run                     hpo -> select -> train -> analyse, skipping what is done
    select                  write configs/hparams/<study>.yaml
    figures <paper>         every figure and table of configs/papers/<paper>.yaml
                            (--item NAME: only that one, repeatable)
    status                  progress per cell; running and failed units with logs
    prune --keep-one|--all|--old [--yes]
                            delete checkpoints (all but seed 1 per cell, or all) or
                            archived old runs; asks first unless --yes

    --gpus 0,1 --per-gpu 2  four parallel units, two per GPU (default: one unit)
"""
import argparse
import logging
import sys
from pathlib import Path

from bm4tc.pipeline import stages
from bm4tc.pipeline.runs import Study

VERBS = ("hpo", "select", "train", "analyse", "run", "status", "prune", "figures")
POOLED = {"hpo": ("hpo",), "train": ("train",), "analyse": ("analyse",), "run": stages.STAGES}


def _unit(argv) -> None:
    """``_unit <kind> <study> [--cell C] [--seed S] [--worker W] [--replace]``:
    one unit of the job pool, run in this process."""
    parser = argparse.ArgumentParser(prog="python -m bm4tc _unit")
    parser.add_argument("kind", choices=("hpo", "select", "train", "analyse"))
    parser.add_argument("study")
    parser.add_argument("--cell")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    study = Study(args.study)
    cell = next((c for c in study.cells() if c.name == args.cell), None)
    stages.run_unit(study, args.kind, cell, args.seed, args.worker, args.replace)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format=stages._FORMAT)
    if argv[:1] == ["_unit"]:
        return _unit(argv[1:])

    parser = argparse.ArgumentParser(prog="python -m bm4tc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("verb", choices=VERBS)
    parser.add_argument("study", help="a study under configs/studies/, e.g. spirals_nat "
                                      "(figures: a paper under configs/papers/)")
    parser.add_argument("--cell", action="append",
                        help="only this cell (repeatable), e.g. legendre/d3r40/a0.01")
    parser.add_argument("--seed", type=int, action="append",
                        help="train, analyse: only this seed (repeatable)")
    parser.add_argument("--gpus", help="comma-separated GPU ids, e.g. 0,1")
    parser.add_argument("--per-gpu", type=int, default=1,
                        help="parallel units per GPU (or in total without --gpus)")
    parser.add_argument("--replace", action="store_true",
                        help="archive finished results whose config changed (or whose checkpoint "
                             "was pruned), then redo them")
    prune = parser.add_mutually_exclusive_group()
    for mode in stages.PRUNE_MODES:
        prune.add_argument(f"--{mode}", dest="prune", action="store_const", const=mode,
                           help="prune: " + {"keep-one": "keep seed 1's checkpoint per cell",
                                             "all": "every checkpoint",
                                             "old": "the archived runs in .replaced/"}[mode])
    parser.add_argument("--yes", action="store_true", help="prune: do not ask")
    parser.add_argument("--item", action="append", help="figures: only this item (repeatable)")
    args = parser.parse_args(argv)
    if (args.verb == "prune") != (args.prune is not None):
        parser.error("prune takes exactly one of --keep-one, --all, --old (and only prune does)")
    if args.verb == "figures":
        from bm4tc.pipeline import figures
        done = figures.make(args.study, args.item)
        failed = sorted(n for n, paths in done.items() if paths is None)
        print(f"{len(done) - len(failed)} items drawn, {len(failed)} failed"
              + (f": {failed}" if failed else ""))
        sys.exit(1 if failed else 0)

    study = Study(args.study)
    cells = study.cells()
    if args.cell:
        unknown = set(args.cell) - {c.name for c in cells}
        if unknown:
            parser.error(f"no such cell(s) in {study.name}: {sorted(unknown)}")
        cells = [c for c in cells if c.name in args.cell]

    if args.verb == "select":
        stages.select(study)
    elif args.verb == "status":
        print(stages.status(study))
    elif args.verb == "prune":
        stages.prune(study, args.prune, args.yes)
    else:
        gpus = args.gpus.split(",") if args.gpus else None
        states = stages.launch(study, POOLED[args.verb], cells, args.seed, gpus,
                               args.per_gpu, args.replace)
        failed = [n for n, s in states.items() if s == "failed"]
        for name in failed:
            log = stages.logs_dir(study) / f"{name}.log"
            tail = Path(log).read_text().splitlines()[-15:] if Path(log).exists() else []
            print(f"\n── FAILED {name} ({log}) ──\n" + "\n".join(tail), file=sys.stderr)
        skipped = sum(s == "skipped" for s in states.values())
        print(f"{len(states) - len(failed) - skipped} done, {len(failed)} failed, "
              f"{skipped} skipped. `python -m bm4tc status {study.name}`")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
