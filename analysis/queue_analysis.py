#!/usr/bin/env python
"""
Run sweep_analysis.py for completed but unanalyzed seed sweeps.

Usage
-----
    python analysis/queue_analysis.py                              # all unanalyzed
    python analysis/queue_analysis.py --dry-run                   # print commands only
    python analysis/queue_analysis.py --filter-embedding hermite
    python analysis/queue_analysis.py --filter-dataset  circles   # substring match
    python analysis/queue_analysis.py --filter-embedding fourier --filter-dataset moons_4k
    python analysis/queue_analysis.py --force                     # re-run already analyzed
    python analysis/queue_analysis.py --list                      # show status of all sweeps
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT          = Path(__file__).parent.parent   # project root (analysis/../)
SWEEP_ROOT    = ROOT / "outputs" / "seed_sweep"
ANALYSIS_ROOT = ROOT / "analysis" / "outputs"


def find_sweep_dirs(root: Path):
    """Yield every sweep dir that contains a multirun.yaml."""
    for marker in sorted(root.rglob("multirun.yaml")):
        yield marker.parent


def analysis_output_dir(sweep_dir: Path) -> Path:
    """Mirror sweep path under analysis/outputs/ (matches sweep_analysis.py logic)."""
    rel = sweep_dir.relative_to(ROOT / "outputs")   # strip leading 'outputs/'
    return ANALYSIS_ROOT / rel


def is_analyzed(sweep_dir: Path) -> bool:
    return (analysis_output_dir(sweep_dir) / "evaluation_data.csv").exists()


def get_embedding(sweep_dir: Path) -> str:
    parts = sweep_dir.relative_to(SWEEP_ROOT).parts
    return parts[1] if len(parts) >= 2 else ""


def get_dataset_base(sweep_dir: Path) -> str:
    """Strip 4-digit date suffix: circles_4k_2102 → circles_4k, circles_2102 → circles."""
    return re.sub(r"_\d{4}$", "", sweep_dir.name)


def main():
    parser = argparse.ArgumentParser(
        description="Run sweep_analysis.py for unanalyzed seed sweeps."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if evaluation_data.csv already exists.")
    parser.add_argument("--filter-embedding", metavar="EMB",
                        help="Only process sweeps with this embedding (e.g. hermite).")
    parser.add_argument("--filter-dataset", metavar="DS",
                        help="Only process sweeps with this base dataset name (e.g. circles_4k).")
    parser.add_argument("--list", action="store_true",
                        help="Print status of all discovered sweeps and exit.")
    args = parser.parse_args()

    sweeps = list(find_sweep_dirs(SWEEP_ROOT))
    if not sweeps:
        print("No completed sweeps found under outputs/seed_sweep/.")
        return

    if args.list:
        for s in sweeps:
            status = "OK " if is_analyzed(s) else "   "
            print(f"[{status}] {s.relative_to(ROOT)}")
        return

    # Apply filters
    if args.filter_embedding:
        sweeps = [s for s in sweeps if get_embedding(s) == args.filter_embedding]
    if args.filter_dataset:
        sweeps = [s for s in sweeps if args.filter_dataset in get_dataset_base(s)]

    todo    = [s for s in sweeps if args.force or not is_analyzed(s)]
    skipped = len(sweeps) - len(todo)

    if skipped:
        print(f"Skipping {skipped} already-analyzed sweep(s).")
    if not todo:
        print("Nothing to do. Use --force to re-run analyzed sweeps.")
        return

    label = "[dry-run] " if args.dry_run else ""
    print(f"{label}Processing {len(todo)} sweep(s):\n")

    for sweep_dir in todo:
        rel = sweep_dir.relative_to(ROOT)
        cmd = ["python", "analysis/sweep_analysis.py", str(rel)]
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"ERROR: analysis failed for {rel}", file=sys.stderr)
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
