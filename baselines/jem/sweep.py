"""Evaluate every numbered run in a JEM seed sweep and write MPS-style CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import data_root

from .analysis import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ood", action="store_true")
    parser.add_argument("--defense-subsample", type=int, default=None)
    parser.add_argument("--threshold-split", choices=("test", "valid"), default="test")
    parser.add_argument("--adaptive-score-weight", type=float, default=1.0)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    runs = sorted(
        (
            p for p in sweep_dir.iterdir()
            if p.is_dir() and p.name.isdigit() and (p / ".hydra/config.yaml").exists()
        ),
        key=lambda p: int(p.name),
    )
    if not runs:
        raise FileNotFoundError(f"No numbered Hydra runs below {sweep_dir}")

    rows = []
    for run in runs:
        result = analyze_run(
            run,
            device=args.device,
            defense_subsample=args.defense_subsample,
            compute_ood=not args.no_ood,
            threshold_split=args.threshold_split,
            adaptive_score_weight=args.adaptive_score_weight,
        )
        rows.append({"run_name": run.name, "run_path": str(run.resolve()), **result})

    df = pd.DataFrame(rows)
    try:
        sweep_suffix = sweep_dir.resolve().relative_to(
            (data_root() / "outputs/baselines/jem").resolve()
        )
    except ValueError:
        sweep_suffix = Path(sweep_dir.name)
    output_dir = data_root() / "analysis/outputs/baselines/jem" / sweep_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "evaluation_data.csv", index=False)

    numeric = df.select_dtypes(include="number")
    summary = numeric.agg(["mean", "std"]).T
    summary.to_csv(output_dir / "evaluation_summary.csv")
    (output_dir / "evaluation_summary.txt").write_text(summary.to_string())
    print(f"Saved {output_dir / 'evaluation_data.csv'}")


if __name__ == "__main__":
    main()
