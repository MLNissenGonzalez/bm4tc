#!/usr/bin/env python
"""
Run sweep.py for completed but unanalyzed seed sweeps.

Scans both outputs/seed_sweep/ and outputs/alpha_curve/ for multirun.yaml
markers.  Distribution plots are intentionally skipped (--no-viz is always
passed); use python -m analysis.visualize.batch to regenerate them separately.

Usage
-----
    python -m analysis.batch                                    # all unanalyzed
    python -m analysis.batch --dry-run                          # print commands only
    python -m analysis.batch --experiment alpha_curve           # alpha_curve sweeps only
    python -m analysis.batch --embedding hermite
    python -m analysis.batch --dataset  circles                 # substring match
    python -m analysis.batch --arch     d4r3
    python -m analysis.batch --type     gen                     # gen | cls | adv
    python -m analysis.batch --embedding fourier --dataset moons
    python -m analysis.batch --force                            # re-run already analyzed
    python -m analysis.batch --list                             # show status of all sweeps
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT          = Path(__file__).parent.parent   # project root (analysis/../)
sys.path.insert(0, str(ROOT))

from src.utils.paths import data_root as _data_root

ANALYSIS_ROOT = _data_root() / "analysis" / "outputs"

# All experiment families to scan.  Each maps a short name to its outputs/ sub-dir.
SWEEP_ROOTS = {
    "seed_sweep":  _data_root() / "outputs" / "seed_sweep",
    "alpha_curve": _data_root() / "outputs" / "alpha_curve",
}


def find_sweep_dirs(root: Path):
    """Yield every sweep dir that contains a multirun.yaml."""
    for marker in sorted(root.rglob("multirun.yaml")):
        yield marker.parent


def analysis_output_dir(sweep_dir: Path) -> Path:
    """Mirror sweep path under analysis/outputs/ (matches sweep.py logic)."""
    rel = sweep_dir.relative_to(_data_root() / "outputs")   # strip leading 'outputs/'
    return ANALYSIS_ROOT / rel


def is_analyzed(sweep_dir: Path) -> bool:
    return (analysis_output_dir(sweep_dir) / "evaluation_data.csv").exists()


def get_training_type(sweep_dir: Path, root: Path) -> str:
    """First path component under root: gen | cls | adv."""
    parts = sweep_dir.relative_to(root).parts
    return parts[0] if len(parts) >= 1 else ""


def get_embedding(sweep_dir: Path, root: Path) -> str:
    parts = sweep_dir.relative_to(root).parts
    return parts[1] if len(parts) >= 2 else ""


def get_arch(sweep_dir: Path, root: Path) -> str:
    parts = sweep_dir.relative_to(root).parts
    return parts[2] if len(parts) >= 3 else ""


def get_dataset_base(sweep_dir: Path) -> str:
    """Strip 4-digit date suffix: circles_4k_2102 → circles_4k, circles_2102 → circles."""
    return re.sub(r"_\d{4}$", "", sweep_dir.name)


def main():
    parser = argparse.ArgumentParser(
        description="Run sweep.py for unanalyzed sweeps (seed_sweep + alpha_curve)."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if evaluation_data.csv already exists.")
    parser.add_argument("--experiment", metavar="EXP",
                        choices=list(SWEEP_ROOTS.keys()),
                        help="Only process sweeps from this experiment family "
                             f"({' | '.join(SWEEP_ROOTS.keys())}).")
    parser.add_argument("--embedding", metavar="EMB",
                        help="Only process sweeps with this embedding (e.g. hermite).")
    parser.add_argument("--dataset", metavar="DS",
                        help="Only process sweeps with this base dataset name (e.g. circles).")
    parser.add_argument("--arch", metavar="ARCH",
                        help="Only process sweeps with this architecture (e.g. d4r3).")
    parser.add_argument("--type", metavar="TYPE",
                        help="Only process sweeps with this training type (gen, cls, adv).")
    parser.add_argument("--list", action="store_true",
                        help="Print status of all discovered sweeps and exit.")
    args = parser.parse_args()

    # Collect (sweep_dir, experiment_name, root) for all roots
    roots_to_scan = (
        {args.experiment: SWEEP_ROOTS[args.experiment]}
        if args.experiment
        else SWEEP_ROOTS
    )

    all_sweeps = []  # list of (sweep_dir, exp_name, root)
    for exp_name, root in roots_to_scan.items():
        if not root.exists():
            continue
        for sweep_dir in find_sweep_dirs(root):
            all_sweeps.append((sweep_dir, exp_name, root))

    if not all_sweeps:
        roots_str = ", ".join(f"outputs/{k}/" for k in roots_to_scan)
        print(f"No completed sweeps found under {roots_str}.")
        return

    if args.list:
        for sweep_dir, exp_name, root in all_sweeps:
            status = "OK " if is_analyzed(sweep_dir) else "   "
            print(f"[{status}] [{exp_name}] {sweep_dir.relative_to(_data_root())}")
        return

    # Apply filters
    filtered = all_sweeps
    if args.type:
        filtered = [(s, e, r) for s, e, r in filtered if get_training_type(s, r) == args.type]
    if args.embedding:
        filtered = [(s, e, r) for s, e, r in filtered if get_embedding(s, r) == args.embedding]
    if args.arch:
        filtered = [(s, e, r) for s, e, r in filtered if get_arch(s, r) == args.arch]
    if args.dataset:
        filtered = [(s, e, r) for s, e, r in filtered if args.dataset in get_dataset_base(s)]

    todo    = [(s, e, r) for s, e, r in filtered if args.force or not is_analyzed(s)]
    skipped = len(filtered) - len(todo)

    if skipped:
        print(f"Skipping {skipped} already-analyzed sweep(s).")
    if not todo:
        print("Nothing to do. Use --force to re-run analyzed sweeps.")
        return

    label = "[dry-run] " if args.dry_run else ""
    print(f"{label}Processing {len(todo)} sweep(s):\n")

    for sweep_dir, exp_name, root in todo:
        rel = sweep_dir.relative_to(_data_root())
        # --no-viz: distribution plots are handled separately by analysis/visualize/batch.py
        cmd = ["python", "analysis/sweep.py", str(rel)]
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"ERROR: analysis failed for {rel}", file=sys.stderr)
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
