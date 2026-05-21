#!/usr/bin/env python
"""
Discover and run seed_sweep / hpo / grid_sweep experiment configs from configs/experiments/.

Directory layout:
  configs/experiments/nll/{dis,gen}/{embedding}/{arch}/{kind}/{dataset}.yaml
  configs/experiments/adversarial/{embedding}/{arch}/{kind}/{dataset}.yaml

where {kind} is one of: hpo, hpo_<variant>, seed_sweep, seed_sweep_<variant>,
grid_sweep, grid_sweep_<variant>, cls_reg.

Usage
-----
    python -m experiments.queue_experiments --list
    python -m experiments.queue_experiments --dry-run
    python -m experiments.queue_experiments --filter-type gen --filter-embedding legendre --dry-run
    python -m experiments.queue_experiments --filter-type gen --filter-kind seed_sweep
    python -m experiments.queue_experiments --filter-type dis --filter-dataset moons --force --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT         = Path(__file__).parent.parent
CONFIGS_ROOT = ROOT / "configs" / "experiments"

VALID_TYPES = ["dis", "gen", "mixed", "adversarial"]
BASE_KINDS = ["seed_sweep", "hpo", "grid_sweep", "cls_reg", "alpha_curve"]

TYPE_TO_MODULE = {
    "dis":         "experiments.nll",
    "gen":         "experiments.nll",
    "mixed":       "experiments.nll",
    "adversarial": "experiments.adversarial",
}

TYPE_SHORT = {
    "dis":   "dis",
    "gen":   "gen",
    "mix":   "mixed",
    "adv":   "adversarial",
}

TYPE_ABBREV = {v: k for k, v in TYPE_SHORT.items()}

KIND_SHORT = {
    "seed": "seed_sweep",
}


def parse_dataset_name(config_path, fallback):
    """Extract dataset name from yaml defaults list; fall back to config stem."""
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
    return fallback


def get_experiment_field(config_path, kind):
    """Read 'experiment:' from yaml; derive from kind if absent."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        exp = data.get("experiment")
        if exp:
            return exp
    except Exception:
        pass
    return "seed_sweep" if kind.startswith("seed_sweep") else kind


def is_already_run(experiment, typ, embedding, arch, dataset_name):
    """Return True if an output directory exists for this config."""
    regime = "nll" if typ in ("dis", "gen", "mixed") else "adv"
    pattern = f"outputs/{experiment}/{regime}/{embedding}/{arch}/{dataset_name}_*"
    return any(ROOT.glob(pattern))


def discover_configs():
    """Yield config dicts for every seed_sweep / hpo / grid_sweep yaml under CONFIGS_ROOT.

    NLL layout:     nll/{dis,gen}/{embedding}/{arch}/{kind}/{dataset}.yaml
    Adversarial:    adversarial/{embedding}/{arch}/{kind}/{dataset}.yaml
    """
    configs = []

    # ── NLL (dis + gen + mixed) ────────────────────────────────────────────
    for sub in ("dis", "gen", "mixed"):
        type_dir = CONFIGS_ROOT / "nll" / sub
        if not type_dir.is_dir():
            continue
        for config_path in sorted(type_dir.rglob("*.yaml")):
            rel   = config_path.relative_to(type_dir)
            parts = rel.parts  # (embedding, arch, kind, name)

            kind_idx = next(
                (i for i, p in enumerate(parts[:-1])
                 if any(p == k or p.startswith(k + "_") for k in BASE_KINDS)), None
            )
            if kind_idx is None:
                continue
            kind   = parts[kind_idx]
            prefix = parts[:kind_idx]
            if len(prefix) != 2:
                continue
            embedding, arch = prefix
            name          = config_path.stem
            dataset_name  = parse_dataset_name(config_path, name)
            experiment    = get_experiment_field(config_path, kind)
            experiment_key = str(Path("nll") / sub / rel.with_suffix(""))
            configs.append({
                "type":           sub,
                "embedding":      embedding,
                "arch":           arch,
                "kind":           kind,
                "name":           name,
                "dataset_name":   dataset_name,
                "experiment":     experiment,
                "config_path":    config_path,
                "experiment_key": experiment_key,
            })

    # ── Adversarial ─────────────────────────────────────────────────────────
    type_dir = CONFIGS_ROOT / "adversarial"
    if type_dir.is_dir():
        for config_path in sorted(type_dir.rglob("*.yaml")):
            rel   = config_path.relative_to(type_dir)
            parts = rel.parts

            kind_idx = next(
                (i for i, p in enumerate(parts[:-1])
                 if any(p == k or p.startswith(k + "_") for k in BASE_KINDS)), None
            )
            if kind_idx is None:
                continue
            kind   = parts[kind_idx]
            prefix = parts[:kind_idx]
            if len(prefix) != 2:
                continue
            embedding, arch = prefix
            name          = config_path.stem
            dataset_name  = parse_dataset_name(config_path, name)
            experiment    = get_experiment_field(config_path, kind)
            experiment_key = str(Path("adversarial") / rel.with_suffix(""))
            configs.append({
                "type":           "adversarial",
                "embedding":      embedding,
                "arch":           arch,
                "kind":           kind,
                "name":           name,
                "dataset_name":   dataset_name,
                "experiment":     experiment,
                "config_path":    config_path,
                "experiment_key": experiment_key,
            })

    return configs


def build_cmd(c):
    module = TYPE_TO_MODULE[c["type"]]
    return ["python", "-m", module, "--multirun",
            f"+experiments={c['experiment_key']}"]


def main():
    parser = argparse.ArgumentParser(
        description="Discover and run seed_sweep / hpo / grid_sweep experiment configs."
    )
    parser.add_argument("--list", action="store_true",
                        help="Print all discovered configs with [ran]/[   ] status.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if output already exists.")
    parser.add_argument("--filter-type", metavar="TYPE",
                        help="dis | gen | adv (or full names).")
    parser.add_argument("--filter-embedding", metavar="EMB",
                        help="fourier | legendre | hermite")
    parser.add_argument("--filter-arch", metavar="ARCH",
                        help="e.g. d4r3, d30r18")
    parser.add_argument("--filter-dataset", metavar="DS",
                        help="Substring match on dataset name (e.g. moons).")
    parser.add_argument("--filter-kind", metavar="KIND",
                        help="seed_sweep | hpo | grid_sweep | …")
    args = parser.parse_args()

    configs = discover_configs()
    if not configs:
        print("No experiment configs found under configs/experiments/.")
        return

    if args.filter_type:
        typ = TYPE_SHORT.get(args.filter_type, args.filter_type)
        configs = [c for c in configs if c["type"] == typ]
    if args.filter_embedding:
        configs = [c for c in configs if c["embedding"] == args.filter_embedding]
    if args.filter_arch:
        configs = [c for c in configs if c["arch"] == args.filter_arch]
    if args.filter_dataset:
        configs = [c for c in configs if args.filter_dataset in c["dataset_name"]]
    if args.filter_kind:
        kind = KIND_SHORT.get(args.filter_kind, args.filter_kind)
        configs = [c for c in configs if c["kind"] == kind]

    if args.list:
        for c in configs:
            ran = is_already_run(c["experiment"], c["type"], c["embedding"],
                                 c["arch"], c["dataset_name"])
            status = "ran" if ran else "   "
            print(f"[{status}] {c['experiment_key']}")
        return

    todo    = [c for c in configs
               if args.force or not is_already_run(
                   c["experiment"], c["type"], c["embedding"],
                   c["arch"], c["dataset_name"])]
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
