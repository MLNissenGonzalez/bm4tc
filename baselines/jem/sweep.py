"""Evaluate every numbered run in a JEM seed sweep and write MPS-style CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.utils.paths import data_root

from .analysis import analyze_run, load_run_config
from .report import write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--defense-subsample", type=int, default=None)
    parser.add_argument("--sampling-radius", type=float, default=0.2)
    parser.add_argument("--sampling-sweeps", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--sampling-steps-per-sweep", type=int, default=20)
    parser.add_argument("--sampling-step-size", type=float, default=0.01)
    parser.add_argument("--sampling-noise-std", type=float, default=0.005)
    parser.add_argument("--sampling-subsample", type=int, default=1000)
    parser.add_argument("--threshold-split", choices=("test", "valid"), default="test")
    parser.add_argument("--adaptive-score-weight", type=float, default=1.0)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    runs = sorted(
        (
            p for p in sweep_dir.iterdir()
            if p.is_dir()
            and p.name.isdigit()
            and (
                (p / ".hydra/config.yaml").exists()
                or (p / "models/model.pt").exists()
                or (p / "models/model").exists()
            )
        ),
        key=lambda p: int(p.name),
    )
    if not runs:
        raise FileNotFoundError(f"No numbered JEM runs below {sweep_dir}")

    rows = []
    for run in runs:
        run_cfg = load_run_config(run)
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
            sampling_radius=args.sampling_radius,
            sampling_sweeps=tuple(args.sampling_sweeps),
            sampling_steps_per_sweep=args.sampling_steps_per_sweep,
            sampling_step_size=args.sampling_step_size,
            sampling_noise_std=args.sampling_noise_std,
            sampling_subsample=args.sampling_subsample,
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
