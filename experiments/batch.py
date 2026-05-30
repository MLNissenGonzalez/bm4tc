#!/usr/bin/env python
"""
Discover and run experiment configs from configs/experiments/.

Directory layout:
  configs/experiments/{dataset}/{nat|at}/{embedding}/{arch}/{kind}.yaml

where {kind} encodes the experiment type and alpha (if applicable):
  hpo_a0, hpo_a1, hpo_a05, hpo        (HPO variants)
  seed_sweep_a0, seed_sweep_a1, …      (seed sweep variants)
  alpha_curve, grid_sweep, cls_reg_a1  (special sweep types)

Usage
-----
    python -m experiments.batch --list
    python -m experiments.batch --dry-run
    python -m experiments.batch --trainer nat --embedding legendre --dry-run
    python -m experiments.batch --trainer nat --kind seed_sweep_a1
    python -m experiments.batch --dataset moons --force --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT         = Path(__file__).parent.parent
CONFIGS_ROOT = ROOT / "configs" / "experiments"

VALID_TRAINERS = ["nat", "at"]

BASE_KINDS = ["seed_sweep", "hpo", "grid_sweep", "cls_reg", "alpha_curve"]


def parse_dataset_name(config_path):
    """Extract dataset name from yaml defaults list; fall back to dataset folder."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        for entry in data.get("defaults", []):
            if isinstance(entry, dict):
                val = entry.get("override /dataset", "")
                if val:
                    return val.split("/")[-1]
    except Exception:
        pass
    # Fallback: dataset folder name (grandparent^3 of the yaml)
    return config_path.parts[-5] if len(config_path.parts) >= 5 else config_path.stem


def get_experiment_field(config_path, kind_stem):
    """Read 'experiment:' from yaml; derive from kind stem if absent."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        exp = data.get("experiment")
        if exp:
            return exp
    except Exception:
        pass
    # Derive from kind stem: strip alpha suffixes to get canonical kind name
    base = kind_stem
    for suffix in ("_a0", "_a1", "_a05", "_a01", "_linf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def is_already_run(dataset, trainer, embedding, arch, kind_stem):
    """Return True if an output directory exists for this config."""
    pattern = f"outputs/{dataset}/{trainer}/{embedding}/{arch}/{kind_stem}_*"
    return any(ROOT.glob(pattern))


def discover_configs():
    """Yield config dicts for every yaml under CONFIGS_ROOT/{dataset}/{nat|at}/...

    Layout: {dataset}/{trainer}/{embedding}/{arch}/{kind}.yaml
    """
    configs = []

    for trainer_dir in CONFIGS_ROOT.iterdir():
        if not trainer_dir.is_dir() or trainer_dir.name == "tests":
            continue
        dataset = trainer_dir.name

        for trainer_type_dir in trainer_dir.iterdir():
            if not trainer_type_dir.is_dir():
                continue
            trainer = trainer_type_dir.name
            if trainer not in VALID_TRAINERS:
                continue

            for config_path in sorted(trainer_type_dir.rglob("*.yaml")):
                rel = config_path.relative_to(trainer_type_dir)
                parts = rel.parts  # (embedding, arch, kind.yaml)
                if len(parts) != 3:
                    continue
                embedding, arch, kind_yaml = parts
                kind_stem = Path(kind_yaml).stem

                dataset_name  = parse_dataset_name(config_path)
                experiment    = get_experiment_field(config_path, kind_stem)
                experiment_key = str(
                    Path(dataset) / trainer / embedding / arch / kind_stem
                )

                configs.append({
                    "dataset":        dataset,
                    "trainer":        trainer,
                    "embedding":      embedding,
                    "arch":           arch,
                    "kind":           kind_stem,
                    "dataset_name":   dataset_name,
                    "experiment":     experiment,
                    "config_path":    config_path,
                    "experiment_key": experiment_key,
                })

    return configs


def build_cmd(c):
    return [
        "python", "-m", "experiments.train", "--multirun",
        f"+experiments={c['experiment_key']}",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Discover and run experiment configs.",
    )
    parser.add_argument("--list", action="store_true",
                        help="Print all discovered configs with [ran]/[   ] status.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if output already exists.")
    parser.add_argument("--trainer", metavar="TRAINER",
                        help="nat | at")
    parser.add_argument("--embedding", metavar="EMB",
                        help="fourier | legendre | hermite")
    parser.add_argument("--arch", metavar="ARCH",
                        help="e.g. d4r3, d30r18")
    parser.add_argument("--dataset", metavar="DS",
                        help="Substring match on dataset name (e.g. moons).")
    parser.add_argument("--kind", metavar="KIND",
                        help="Substring match on kind (e.g. seed_sweep, hpo_a1).")
    args = parser.parse_args()

    configs = discover_configs()
    if not configs:
        print("No experiment configs found under configs/experiments/.")
        return

    if args.trainer:
        configs = [c for c in configs if c["trainer"] == args.trainer]
    if args.embedding:
        configs = [c for c in configs if c["embedding"] == args.embedding]
    if args.arch:
        configs = [c for c in configs if c["arch"] == args.arch]
    if args.dataset:
        configs = [c for c in configs if args.dataset in c["dataset_name"]]
    if args.kind:
        configs = [c for c in configs if args.kind in c["kind"]]

    if args.list:
        for c in configs:
            ran = is_already_run(
                c["dataset"], c["trainer"], c["embedding"],
                c["arch"], c["kind"],
            )
            status = "ran" if ran else "   "
            print(f"[{status}] {c['experiment_key']}")
        return

    todo    = [c for c in configs
               if args.force or not is_already_run(
                   c["dataset"], c["trainer"], c["embedding"],
                   c["arch"], c["kind"])]
    skipped = len(configs) - len(todo)

    if skipped:
        print(f"Skipping {skipped} already-run experiment(s).")
    if not todo:
        print("Nothing to do. Use --force to re-run experiments that already have outputs.")
        return

    label = "[dry-run] " if args.dry_run else ""
    print(f"{label}Running {len(todo)} experiment(s):\n")

    for c in todo:
        cmd = build_cmd(c)
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"ERROR: {c['experiment_key']} failed", file=sys.stderr)
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
