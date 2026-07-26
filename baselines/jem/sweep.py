"""Evaluate every numbered run in a JEM seed sweep and write MPS-style CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.utils.paths import data_root

from .analysis import analyze_run
from .report import write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--device", default="auto")
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
        run_cfg = OmegaConf.load(run / ".hydra/config.yaml")
        config_values = {
            "config/dataset.name": OmegaConf.select(run_cfg, "dataset.name"),
            "config/tracking.seed": OmegaConf.select(run_cfg, "tracking.seed"),
            "config/trainer.alpha": OmegaConf.select(run_cfg, "trainer.alpha"),
            "config/trainer.stop_crit": OmegaConf.select(run_cfg, "trainer.stop_crit"),
            "config/sampler.num_steps": OmegaConf.select(run_cfg, "sampler.num_steps"),
            "config/sampler.step_size": OmegaConf.select(run_cfg, "sampler.step_size"),
            "config/sampler.noise_std": OmegaConf.select(run_cfg, "sampler.noise_std"),
        }
        result = analyze_run(
            run,
            device=args.device,
            defense_subsample=args.defense_subsample,
            compute_ood=not args.no_ood,
            threshold_split=args.threshold_split,
            adaptive_score_weight=args.adaptive_score_weight,
        )
        rows.append(
            {
                "run_name": run.name,
                "run_path": str(run.resolve()),
                **config_values,
                **result,
            }
        )

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

    write_summary(
        df,
        output_dir,
        sweep_name=str(sweep_suffix),
        device=args.device,
    )
    print(f"Saved {output_dir / 'evaluation_data.csv'}")


if __name__ == "__main__":
    main()
