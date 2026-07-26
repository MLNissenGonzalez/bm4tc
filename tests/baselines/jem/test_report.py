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
            row[f"sgld_purify_acc/{epsilon}/0.2"] = 0.84
            row[f"uq_joint_purify_acc/{epsilon}/0.2"] = 0.8
            row[f"sgld_joint_purify_acc/{epsilon}/0.2"] = 0.79
        row["uq_clean_purify_acc/0.2"] = 0.94
        row["sgld_clean_purify_acc/0.2"] = 0.93
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
        assert (output / "evaluation_summary.csv").exists()
