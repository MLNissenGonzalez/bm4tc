from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import torch

from baselines.jem.generate import sample_and_plot
from baselines.jem.plots import (
    plot_alpha_curves,
    plot_defense_comparison,
    plot_detection_thresholds,
    plot_purification_radii,
)
from baselines.jem.tables import write_mnist_tables


def _evaluation_frame() -> pd.DataFrame:
    rows = []
    for seed in range(2):
        row = {
            "acc": 0.95 - 0.01 * seed,
            "dis_loss": 0.1 + 0.01 * seed,
            "gen_loss": 0.2 + 0.01 * seed,
        }
        for epsilon in (0.1, 0.2, 0.3):
            row[f"rob/{epsilon}"] = 0.8
            row[f"uq_adv_acc/{epsilon}"] = 0.8
            row[f"uq_joint_adv_acc/{epsilon}"] = 0.75
            for radius in (0.2, 0.3):
                row[f"uq_purify_acc/{epsilon}/{radius}"] = 0.85
                row[f"uq_joint_purify_acc/{epsilon}/{radius}"] = 0.8
            for sweep in (1, 3, 5):
                row[f"sgld_purify_acc/{epsilon}/{sweep}"] = 0.84
                row[f"sgld_joint_purify_acc/{epsilon}/{sweep}"] = 0.79
            for percentile in (1, 5, 10, 20):
                key = f"{percentile}pct/{epsilon}"
                row[f"uq_detection/{key}"] = 0.5
                row[f"uq_joint_detection/{key}"] = 0.4
                row[f"uq_det_err_passed/{key}"] = 0.2
                row[f"uq_joint_det_err_passed/{key}"] = 0.25
        rows.append(row)
    return pd.DataFrame(rows)


def test_notebook_outputs_use_the_mps_filenames_and_strategy_names():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        csv = root / "evaluation_data.csv"
        _evaluation_frame().to_csv(csv, index=False)

        alpha_csv = {}
        for alpha in (0.0, 0.01, 0.1, 0.2, 0.5, 1.0):
            path = root / f"alpha_{alpha}.csv"
            _evaluation_frame().to_csv(path, index=False)
            alpha_csv[alpha] = path
        alpha_outputs = plot_alpha_curves(alpha_csv, root / "alpha")
        assert [path.name for path in alpha_outputs] == [
            "alpha_curve_accuracy.png",
            "alpha_curve_nll.png",
        ]

        models = {r"$\alpha{=}0$": csv, "AT": csv}
        radius_output = plot_purification_radii(
            models,
            root / "purify_radius" / "purify_acc_vs_radius.png",
        )
        assert radius_output.name == "purify_acc_vs_radius.png"
        defense_outputs = plot_defense_comparison(models, root / "robustness")
        assert [path.name for path in defense_outputs] == [
            "defense_comparison_eps0.1.png",
            "defense_comparison_eps0.2.png",
            "defense_comparison_eps0.3.png",
        ]

        keyed_models = {
            "seed_sweep_a0": (r"$\alpha{=}0$", csv),
            "at_seed_sweep": ("AT", csv),
        }
        detection_outputs = plot_detection_thresholds(
            keyed_models,
            root / "detection",
        )
        assert [path.name for path in detection_outputs] == [
            "accept_acc_vs_threshold_seed_sweep_a0.png",
            "accept_acc_vs_threshold_at_seed_sweep.png",
        ]
        table_outputs = write_mnist_tables(keyed_models, root / "tables")
        assert {path.name for path in table_outputs} == {
            "detection.tex",
            "detection.txt",
            "joint_vs_normal_eps0.1.tex",
            "joint_vs_normal_eps0.2.tex",
            "joint_vs_normal_eps0.3.tex",
            "joint_vs_normal.txt",
        }
        assert "Purif. (lk.)" in (root / "tables/detection.txt").read_text()
        assert "Purif. (samp., k=5)" in (
            root / "tables/detection.txt"
        ).read_text()


def test_jem_sampling_uses_the_mps_two_by_five_mean_grid():
    samples = torch.linspace(-1, 1, 10 * 2 * 144).reshape(10, 2, 144)
    with TemporaryDirectory() as directory:
        with patch(
            "baselines.jem.generate.sample_all_classes",
            return_value=(samples, (-1.0, 1.0)),
        ):
            figure = sample_and_plot(
                "unused",
                n_per_class=2,
                steps=1,
                save_dir=directory,
                device="cpu",
            )
        assert len(figure.axes) == 10
        assert (Path(directory) / "mnist_samples.png").exists()
