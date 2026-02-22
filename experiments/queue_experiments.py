#!/usr/bin/env python
"""
Discover and run seed_sweep / hpo experiment configs from configs/experiments/.

Usage
-----
    python -m experiments.queue_experiments --list
    python -m experiments.queue_experiments --dry-run
    python -m experiments.queue_experiments --filter-type gen --filter-embedding hermite --dry-run
    python -m experiments.queue_experiments --filter-type gen --filter-embedding hermite --filter-kind seed_sweep
    python -m experiments.queue_experiments --filter-type cls --filter-dataset moons --force --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT         = Path(__file__).parent.parent
CONFIGS_ROOT = ROOT / "configs" / "experiments"

VALID_TYPES = ["classification", "adversarial", "generative"]
VALID_KINDS = ["seed_sweeps", "hpo"]

TYPE_TO_MODULE = {
    "classification": "experiments.classification",
    "adversarial":    "experiments.adversarial",
    "generative":     "experiments.generative",
}

TYPE_SHORT = {
    "cls": "classification",
    "adv": "adversarial",
    "gen": "generative",
}

KIND_SHORT = {
    "seed_sweep": "seed_sweeps",
    "hpo":        "hpo",
}


def parse_arch(arch):
    """Parse 'd4D3' → (in_dim=4, bond_dim=3)."""
    m = re.match(r"d(\d+)D(\d+)", arch)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


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
    return "seed_sweep" if kind == "seed_sweeps" else "hpo"


def is_already_run(experiment, embedding, in_dim, bond_dim, dataset_name):
    """Return True if an output directory exists for this config."""
    pattern = (
        f"outputs/{experiment}/*/{embedding}"
        f"/d{in_dim}D{bond_dim}/{dataset_name}_*"
    )
    return any(ROOT.glob(pattern))


def discover_configs():
    """Yield config dicts for every seed_sweeps / hpo yaml under CONFIGS_ROOT."""
    configs = []
    for typ in VALID_TYPES:
        type_dir = CONFIGS_ROOT / typ
        if not type_dir.is_dir():
            continue
        for embedding_dir in sorted(type_dir.iterdir()):
            if not embedding_dir.is_dir():
                continue
            embedding = embedding_dir.name
            for arch_dir in sorted(embedding_dir.iterdir()):
                if not arch_dir.is_dir():
                    continue
                arch = arch_dir.name
                in_dim, bond_dim = parse_arch(arch)
                for kind in VALID_KINDS:
                    kind_dir = arch_dir / kind
                    if not kind_dir.is_dir():
                        continue
                    for config_path in sorted(kind_dir.glob("*.yaml")):
                        name         = config_path.stem
                        dataset_name = parse_dataset_name(config_path, name)
                        experiment   = get_experiment_field(config_path, kind)
                        experiment_key = (
                            f"{typ}/{embedding}/{arch}/{kind}/{name}"
                        )
                        configs.append({
                            "type":           typ,
                            "embedding":      embedding,
                            "arch":           arch,
                            "in_dim":         in_dim,
                            "bond_dim":       bond_dim,
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
        description="Discover and run seed_sweep / hpo experiment configs."
    )
    parser.add_argument("--list", action="store_true",
                        help="Print all discovered configs with [ran]/[   ] status.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if output already exists.")
    parser.add_argument("--filter-type", metavar="TYPE",
                        help="cls | adv | gen (or full names).")
    parser.add_argument("--filter-embedding", metavar="EMB",
                        help="fourier | legendre | hermite")
    parser.add_argument("--filter-arch", metavar="ARCH",
                        help="e.g. d4D3, d30D18")
    parser.add_argument("--filter-dataset", metavar="DS",
                        help="Substring match on dataset name (e.g. moons).")
    parser.add_argument("--filter-kind", metavar="KIND",
                        help="seed_sweep | hpo")
    args = parser.parse_args()

    configs = discover_configs()
    if not configs:
        print("No experiment configs found under configs/experiments/.")
        return

    if args.list:
        for c in configs:
            ran = is_already_run(c["experiment"], c["embedding"],
                                 c["in_dim"], c["bond_dim"], c["dataset_name"])
            status = "ran" if ran else "   "
            print(f"[{status}] {c['experiment_key']}")
        return

    # Apply filters
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

    todo    = [c for c in configs
               if args.force or not is_already_run(
                   c["experiment"], c["embedding"],
                   c["in_dim"], c["bond_dim"], c["dataset_name"])]
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
