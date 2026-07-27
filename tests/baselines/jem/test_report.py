from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from baselines.jem.report import write_summary


def test_compact_summary_contains_standard_and_joint_tables():
    rows = []
    for seed in range(2):
        row = {
            "run_name": str(seed),
            "alpha": 0.1,
            "acc": 0.95,
            "dis_loss": 0.1,
            "mixed_loss": 0.2,
            "px_cd_loss": 1.0,
            "joint_cd_loss": 1.1,
        }
        for epsilon in (0.1, 0.2, 0.3):
            row[f"rob/{epsilon}"] = 0.8
            row[f"uq_joint_adv_acc/{epsilon}"] = 0.75
            row[f"uq_detection/10pct/{epsilon}"] = 0.5
            row[f"uq_joint_detection/10pct/{epsilon}"] = 0.4
            row[f"uq_purify_acc/{epsilon}/0.2"] = 0.85
            row[f"sgld_purify_acc/{epsilon}/1"] = 0.84
            row[f"sgld_purify_acc/{epsilon}/3"] = 0.85
            row[f"sgld_purify_acc/{epsilon}/5"] = 0.86
            row[f"uq_joint_purify_acc/{epsilon}/0.2"] = 0.8
            row[f"sgld_joint_purify_acc/{epsilon}/1"] = 0.79
            row[f"sgld_joint_purify_acc/{epsilon}/3"] = 0.8
            row[f"sgld_joint_purify_acc/{epsilon}/5"] = 0.81
        row["uq_clean_purify_acc/0.2"] = 0.94
        row["sgld_clean_purify_acc/1"] = 0.93
        row["sgld_clean_purify_acc/3"] = 0.92
        row["sgld_clean_purify_acc/5"] = 0.91
        row["sgld_purify_radius"] = 0.2
        row["sgld_steps_per_sweep"] = 20
        row["sgld_purify_step_size"] = 0.01
        row["sgld_purify_noise_std"] = 0.005
        rows.append(row)

    with TemporaryDirectory() as directory:
        output = Path(directory)
        write_summary(
            pd.DataFrame(rows),
            output,
            sweep_name="test",
            device="cpu",
        )
        text = (output / "evaluation_summary.txt").read_text()
        assert "Standard PGD" in text
        assert "Likelihood-aware joint PGD" in text
        assert "Purif. (lk.) [r=0.2]" in text
        assert "Purif. (samp., k=1)" in text
        assert "Purif. (samp., k=3)" in text
        assert "Purif. (samp., k=5)" in text
        assert "delta=0.2" in text
        assert "step_size=0.01" in text
        assert (output / "evaluation_summary.csv").exists()
