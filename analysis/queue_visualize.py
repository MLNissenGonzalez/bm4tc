#!/usr/bin/env python
"""
Regenerate distribution visualizations for already-analyzed seed sweeps.

Produces best_class_dist.png and best_joint.png in each analysis output dir,
replacing the old best_run_distributions.png if present.

Usage
-----
    python analysis/queue_visualize.py                              # all unvisualized
    python analysis/queue_visualize.py --dry-run                   # print commands only
    python analysis/queue_visualize.py --list                      # show status
    python analysis/queue_visualize.py --force                     # re-run already done
    python analysis/queue_visualize.py --filter-embedding hermite
    python analysis/queue_visualize.py --filter-dataset  circles   # substring match
    python analysis/queue_visualize.py --filter-arch     d4D3
    python analysis/queue_visualize.py --filter-type     gen       # gen | cls | adv
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT          = Path(__file__).parent.parent   # project root
SWEEP_ROOT    = ROOT / "outputs" / "seed_sweep"
ANALYSIS_ROOT = ROOT / "analysis" / "outputs"


def find_sweep_dirs(root: Path):
    """Yield every sweep dir that contains a multirun.yaml."""
    for marker in sorted(root.rglob("multirun.yaml")):
        yield marker.parent


def analysis_output_dir(sweep_dir: Path) -> Path:
    """Mirror sweep path under analysis/outputs/ (matches seed_sweep_analysis.py logic)."""
    rel = sweep_dir.relative_to(ROOT / "outputs")   # strip leading 'outputs/'
    return ANALYSIS_ROOT / rel


_UCR_NAMES = {
    "ecg200", "italypowerdemand", "chlorineconcentration",
    "syntheticcontrol", "cricketx", "crickety", "cricketz",
}


def get_viz_type(sweep_dir: Path) -> str:
    """Return 'mnist', 'ts', or '2d' based on the dataset name in the sweep path."""
    base = get_dataset_base(sweep_dir).lower()
    if "mnist" in base:
        return "mnist"
    for name in _UCR_NAMES:
        if name in base:
            return "ts"
    return "2d"


def is_analyzed(sweep_dir: Path) -> bool:
    return (analysis_output_dir(sweep_dir) / "evaluation_data.csv").exists()


def is_visualized(sweep_dir: Path) -> bool:
    ana_dir = analysis_output_dir(sweep_dir)
    return (
        (ana_dir / "best_class_dist.png").exists()
        or (ana_dir / "mnist_samples.png").exists()
        or (ana_dir / "ts_samples.png").exists()
    )


def best_run_from_csv(ana_dir: Path) -> str | None:
    """Read evaluation_data.csv; return run_path with highest eval/valid/acc."""
    csv = ana_dir / "evaluation_data.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if "run_path" not in df.columns:
        return None
    df["_idx"] = pd.to_numeric(df["run_path"].apply(lambda p: Path(p).name), errors="coerce")
    row = df.loc[df["_idx"].idxmin()]
    return row["run_path"]


def get_training_type(sweep_dir: Path) -> str:
    parts = sweep_dir.relative_to(SWEEP_ROOT).parts
    return parts[0] if len(parts) >= 1 else ""


def get_embedding(sweep_dir: Path) -> str:
    parts = sweep_dir.relative_to(SWEEP_ROOT).parts
    return parts[1] if len(parts) >= 2 else ""


def get_arch(sweep_dir: Path) -> str:
    parts = sweep_dir.relative_to(SWEEP_ROOT).parts
    return parts[2] if len(parts) >= 3 else ""


def get_dataset_base(sweep_dir: Path) -> str:
    """Strip 4-digit date suffix: circles_4k_2102 → circles_4k."""
    return re.sub(r"_\d{4}$", "", sweep_dir.name)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate distribution visualizations for analyzed seed sweeps."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if best_class_dist.png already exists.")
    parser.add_argument("--filter-embedding", metavar="EMB",
                        help="Only process sweeps with this embedding (e.g. hermite).")
    parser.add_argument("--filter-dataset", metavar="DS",
                        help="Only process sweeps with this base dataset name (e.g. circles_4k).")
    parser.add_argument("--filter-arch", metavar="ARCH",
                        help="Only process sweeps with this architecture (e.g. d4D3).")
    parser.add_argument("--filter-type", metavar="TYPE",
                        help="Only process sweeps with this training type (gen, cls, adv).")
    parser.add_argument("--list", action="store_true",
                        help="Print status of all discovered sweeps and exit.")
    args = parser.parse_args()

    sweeps = list(find_sweep_dirs(SWEEP_ROOT))
    if not sweeps:
        print("No completed sweeps found under outputs/seed_sweep/.")
        return

    # Only consider sweeps that have been analyzed (have evaluation_data.csv)
    analyzed = [s for s in sweeps if is_analyzed(s)]

    if args.list:
        for s in sweeps:
            status = "OK " if is_visualized(s) else "   "
            print(f"[{status}] {s.relative_to(ROOT)}")
        return

    # Apply filters
    if args.filter_type:
        analyzed = [s for s in analyzed if get_training_type(s) == args.filter_type]
    if args.filter_embedding:
        analyzed = [s for s in analyzed if get_embedding(s) == args.filter_embedding]
    if args.filter_arch:
        analyzed = [s for s in analyzed if get_arch(s) == args.filter_arch]
    if args.filter_dataset:
        analyzed = [s for s in analyzed if args.filter_dataset in get_dataset_base(s)]

    todo    = [s for s in analyzed if args.force or not is_visualized(s)]
    skipped = len(analyzed) - len(todo)

    if skipped:
        print(f"Skipping {skipped} already-visualized sweep(s).")
    if not todo:
        print("Nothing to do. Use --force to re-run visualized sweeps.")
        return

    label = "[dry-run] " if args.dry_run else ""
    print(f"{label}Processing {len(todo)} sweep(s):\n")

    for sweep_dir in todo:
        ana_dir = analysis_output_dir(sweep_dir)
        rel = sweep_dir.relative_to(ROOT)

        # Delete old distributions file if present
        old_dist = ana_dir / "best_run_distributions.png"
        if old_dist.exists() and not args.dry_run:
            old_dist.unlink()
            print(f"  Deleted {old_dist.relative_to(ROOT)}")

        run_path = best_run_from_csv(ana_dir)
        if run_path is None:
            print(f"  WARNING: Could not determine best run for {rel}, skipping.")
            continue

        viz_type = get_viz_type(sweep_dir)
        if viz_type == "mnist":
            module = "analysis.visualize.mnist_samples"
        elif viz_type == "ts":
            module = "analysis.visualize.ts_samples"
        else:
            module = "analysis.visualize.distributions"
        cmd = [
            "python", "-m", module,
            "--run", run_path,
            "--save-dir", str(ana_dir),
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"ERROR: visualization failed for {rel}", file=sys.stderr)
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
