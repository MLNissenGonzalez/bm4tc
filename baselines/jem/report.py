"""Compact MPS-style reporting for JEM seed-sweep evaluation CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _numeric_suffix(columns, prefix: str) -> list[tuple[float, str]]:
    parsed = []
    for column in columns:
        if not column.startswith(prefix):
            continue
        try:
            parsed.append((float(column.split("/")[-1]), column.split("/")[-1]))
        except ValueError:
            continue
    return sorted(set(parsed))


def write_summary(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    sweep_name: str,
    device: str,
) -> None:
    """Write numeric CSV and a compact human-readable report."""
    numeric = df.select_dtypes(include="number")
    summary = numeric.agg(["mean", "std", "count"]).T
    summary.to_csv(output_dir / "evaluation_summary.csv")

    def mean(column: str) -> float:
        if column not in df:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.mean()) if len(values) else float("nan")

    def std(column: str) -> float:
        if column not in df:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.std()) if len(values) > 1 else float("nan")

    def fmt(value: float, width: int = 9) -> str:
        return ("—" if np.isnan(value) else f"{value:.4f}").rjust(width)

    eps = _numeric_suffix(df.columns, "rob/")
    radii = sorted(
        {
            column.split("/")[-1]
            for column in df.columns
            if column.startswith(("uq_purify_acc/", "uq_clean_purify_acc/"))
        },
        key=float,
    )
    sgld_sweeps = sorted(
        {
            int(column.split("/")[-1])
            for column in df.columns
            if column.startswith(("sgld_purify_acc/", "sgld_clean_purify_acc/"))
            and column.split("/")[-1].isdigit()
        }
    )
    percentiles = sorted(
        {
            int(column.split("/")[-2].replace("pct", ""))
            for column in df.columns
            if column.startswith("uq_detection/")
        }
    )
    headers = ["eps=0", *[value for _, value in eps]]

    def accuracy_rows(joint: bool = False):
        rows = []
        if joint:
            rows.append(
                (
                    "No defense",
                    [mean(f"uq_joint_adv_acc/{value}") for _, value in eps],
                )
            )
            if percentiles:
                q = percentiles[0]
                rows.append(
                    (
                        f"Detection (tau={q}%)",
                        [
                            mean(f"uq_joint_detection/{q}pct/{value}")
                            for _, value in eps
                        ],
                    )
                )
            for radius in radii:
                rows.append(
                    (
                        f"Purif. (lk.) [r={radius}]",
                        [
                            mean(f"uq_joint_purify_acc/{value}/{radius}")
                            for _, value in eps
                        ],
                    )
                )
            for sweep in sgld_sweeps:
                rows.append(
                    (
                        f"Purif. (samp., k={sweep})",
                        [
                            mean(f"sgld_joint_purify_acc/{value}/{sweep}")
                            for _, value in eps
                        ],
                    )
                )
        else:
            rows.append(
                ("No defense", [mean("acc")] + [mean(f"rob/{value}") for _, value in eps])
            )
            if percentiles:
                q = percentiles[0]
                rows.append(
                    (
                        f"Detection (tau={q}%)",
                        [float("nan")]
                        + [
                            mean(f"uq_detection/{q}pct/{value}")
                            for _, value in eps
                        ],
                    )
                )
            for radius in radii:
                rows.append(
                    (
                        f"Purif. (lk.) [r={radius}]",
                        [mean(f"uq_clean_purify_acc/{radius}")]
                        + [
                            mean(f"uq_purify_acc/{value}/{radius}")
                            for _, value in eps
                        ],
                    )
                )
            for sweep in sgld_sweeps:
                rows.append(
                    (
                        f"Purif. (samp., k={sweep})",
                        [mean(f"sgld_clean_purify_acc/{sweep}")]
                        + [
                            mean(f"sgld_purify_acc/{value}/{sweep}")
                            for _, value in eps
                        ],
                    )
                )
        return rows

    def write_table(handle, title: str, rows, table_headers=None) -> None:
        table_headers = headers if table_headers is None else table_headers
        handle.write("-" * 76 + "\n")
        handle.write(title + "\n")
        handle.write("-" * 76 + "\n\n")
        label_width = max([len(label) for label, _ in rows] + [12]) + 2
        handle.write(
            " " * label_width + "".join(h.rjust(9) for h in table_headers) + "\n"
        )
        for label, values in rows:
            handle.write(f"{label:<{label_width}}" + "".join(fmt(v) for v in values) + "\n")
        handle.write("\n")

    path = output_dir / "evaluation_summary.txt"
    with path.open("w") as handle:
        handle.write("=" * 76 + "\n")
        handle.write(f"JEM Seed Sweep: {sweep_name}\n")
        handle.write("=" * 76 + "\n\n")
        handle.write(f"Runs: {len(df)}  |  Device: {device}\n")
        # JEM has no embedding, so its budgets are absolute (pixel space) by design —
        # the MPS summaries state "relative". Neither convention is guessable from the
        # numbers alone, so each file says which one it is.
        handle.write("Budgets: ABSOLUTE (pixel space; JEM has no embedding rescaling)\n")
        alpha_mean = mean("alpha")
        if not np.isnan(alpha_mean):
            handle.write(f"Alpha: {alpha_mean:.4g}\n")
        handle.write("Attack: PGD Linf, 40 steps\n\n")
        sampling_radius = mean("sgld_purify_radius")
        steps_per_sweep = mean("sgld_steps_per_sweep")
        sampling_step_size = mean("sgld_purify_step_size")
        sampling_noise_std = mean("sgld_purify_noise_std")
        if not np.isnan(sampling_radius):
            handle.write(
                "Sampling purification: local projected SGLD"
                f"  |  delta={sampling_radius:.4g}"
                f"  |  steps/sweep={steps_per_sweep:.0f}"
            )
            if not np.isnan(sampling_step_size):
                handle.write(
                    f"  |  step_size={sampling_step_size:.4g}"
                    f"  |  noise_std={sampling_noise_std:.4g}"
                )
            handle.write("\n\n")

        handle.write("Core metrics (mean ± std)\n")
        handle.write("-" * 76 + "\n")
        for column, label in (
            ("acc", "Clean accuracy"),
            ("dis_loss", "Discriminative CE"),
            ("mixed_loss", "Post-hoc mixed CD diagnostic"),
            ("px_cd_loss", "Marginal CD surrogate"),
            ("joint_cd_loss", "Joint CD surrogate"),
        ):
            if column in df:
                handle.write(f"  {label:<38} {mean(column):.4f} ± {std(column):.4f}\n")
        handle.write("\n")

        write_table(handle, "Standard PGD — accuracy vs perturbation strength", accuracy_rows())
        if any(column.startswith("uq_joint_adv_acc/") for column in df.columns):
            write_table(
                handle,
                "Likelihood-aware joint PGD — accuracy vs perturbation strength",
                accuracy_rows(joint=True),
                headers[1:],
            )

        if percentiles and eps:
            target = eps[1] if len(eps) > 1 else eps[0]
            _, eps_name = target
            handle.write("-" * 76 + "\n")
            handle.write(f"Detection rates at eps={eps_name} (mean ± std)\n")
            handle.write("-" * 76 + "\n\n")
            for q in percentiles:
                standard = f"uq_detection/{q}pct/{eps_name}"
                joint = f"uq_joint_detection/{q}pct/{eps_name}"
                handle.write(
                    f"  tau={q:>2}%  standard {mean(standard):.4f} ± {std(standard):.4f}"
                )
                if joint in df:
                    handle.write(f"  |  joint {mean(joint):.4f} ± {std(joint):.4f}")
                handle.write("\n")
            handle.write("\n")

        handle.write("=" * 76 + "\n")
