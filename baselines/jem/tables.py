"""MNIST paper tables matching the MPS notebook schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _mean_std(
    frame: pd.DataFrame | None,
    column: str,
    *,
    flip: bool = False,
) -> tuple[float, float]:
    if frame is None or column not in frame:
        return float("nan"), float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if not len(values):
        return float("nan"), float("nan")
    if flip:
        values = 1.0 - values
    return float(values.mean()), float(values.std() if len(values) > 1 else 0.0)


def _delta(
    frame: pd.DataFrame | None,
    joint_column: str,
    normal_column: str,
) -> tuple[float, float]:
    if (
        frame is None
        or joint_column not in frame
        or normal_column not in frame
    ):
        return float("nan"), float("nan")
    values = (
        pd.to_numeric(frame[joint_column], errors="coerce").to_numpy()
        - pd.to_numeric(frame[normal_column], errors="coerce").to_numpy()
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std() if len(values) > 1 else 0.0)


def _tex(
    mean: float,
    std: float,
    *,
    bold: bool = False,
    signed: bool = False,
) -> str:
    if np.isnan(mean):
        return "---"
    number = f"{mean:+.3f}" if signed else f"{mean:.3f}"
    body = f"{number} \\pm {std:.3f}" if np.isfinite(std) and std else number
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def _plain(
    mean: float,
    std: float,
    *,
    best: bool = False,
    signed: bool = False,
) -> str:
    if np.isnan(mean):
        return "---"
    number = f"{mean:+.3f}" if signed else f"{mean:.3f}"
    body = f"{number}±{std:.3f}" if np.isfinite(std) and std else number
    return body + ("*" if best else "")


def _plain_label(label: str) -> str:
    return (
        label.replace(r"\ ", " ")
        .replace(r"\%", "%")
        .replace("$", "")
        .replace("{=}", "=")
        .replace(r"\alpha", "alpha")
        .replace(r"\varepsilon", "eps")
        .replace("{", "")
        .replace("}", "")
    )


def write_mnist_tables(
    models: dict[str, tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    epsilons: tuple[float, ...] = (0.1, 0.2, 0.3),
    radius: float = 0.2,
    sampling_sweep: int = 5,
    percentile: int = 10,
) -> list[Path]:
    """Write the same detection and joint-vs-normal tables as MPS."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [
        (
            label,
            pd.read_csv(path) if Path(path).exists() else None,
        )
        for label, path in models.values()
    ]
    for _, frame in loaded:
        if frame is None:
            continue
        missing = [
            f"sgld_purify_acc/{epsilon}/{sampling_sweep}"
            for epsilon in epsilons
            if f"sgld_purify_acc/{epsilon}/{sampling_sweep}" not in frame
        ]
        if missing:
            raise ValueError(
                f"Missing sampling-purification metrics {missing}. Rerun "
                "baselines.jem.sweep with the locally projected SGLD analysis."
            )
    percentile_key = f"{percentile}pct"
    detection_metrics = [
        ("Clean", "acc", False, False),
        ("Rob.", "rob/{eps}", True, False),
        (
            r"Purif.\ (lk.)",
            f"uq_purify_acc/{{eps}}/{radius}",
            True,
            False,
        ),
        (
            rf"Purif.\ (samp., $k{{=}}{sampling_sweep}$)",
            f"sgld_purify_acc/{{eps}}/{sampling_sweep}",
            True,
            False,
        ),
        (
            rf"Det.\ rate ($q{{=}}{percentile}\%$)",
            f"uq_detection/{percentile_key}/{{eps}}",
            True,
            False,
        ),
        (
            rf"Accept.\ ($q{{=}}{percentile}\%$)",
            f"uq_det_err_passed/{percentile_key}/{{eps}}",
            True,
            True,
        ),
    ]

    data = []
    for _, frame in loaded:
        model_data = []
        for _, template, epsilon_dependent, flip in detection_metrics:
            if epsilon_dependent:
                values = [
                    _mean_std(frame, template.format(eps=eps), flip=flip)
                    for eps in epsilons
                ]
            else:
                values = [_mean_std(frame, template, flip=flip)]
            model_data.append(values)
        data.append(model_data)

    best = []
    for metric_index in range(len(detection_metrics)):
        metric_best = []
        for epsilon_index in range(len(data[0][metric_index])):
            means = [
                data[model_index][metric_index][epsilon_index][0]
                for model_index in range(len(loaded))
            ]
            valid = [
                (value, index)
                for index, value in enumerate(means)
                if np.isfinite(value)
            ]
            metric_best.append(max(valid)[1] if valid else -1)
        best.append(metric_best)

    latex = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Model & Metric & "
        + " & ".join(rf"$\varepsilon={eps}$" for eps in epsilons)
        + r" \\",
        r"\midrule",
    ]
    for model_index, (label, _) in enumerate(loaded):
        for metric_index, (
            metric_label,
            _,
            epsilon_dependent,
            _,
        ) in enumerate(detection_metrics):
            model_cell = (
                f"\\multirow{{{len(detection_metrics)}}}{{*}}{{{label}}}"
                if metric_index == 0
                else ""
            )
            slots = data[model_index][metric_index]
            if epsilon_dependent:
                cells = [
                    _tex(
                        *slots[index],
                        bold=best[metric_index][index] == model_index,
                    )
                    for index in range(len(epsilons))
                ]
                latex.append(
                    f"{model_cell} & {metric_label} & " + " & ".join(cells) + r" \\"
                )
            else:
                value = _tex(
                    *slots[0],
                    bold=best[metric_index][0] == model_index,
                )
                latex.append(
                    f"{model_cell} & {metric_label} & "
                    f"\\multicolumn{{{len(epsilons)}}}{{c}}{{{value}}} \\\\"
                )
        if model_index < len(loaded) - 1:
            latex.append(r"\cmidrule(lr){1-5}")
    latex += [r"\bottomrule", r"\end{tabular}"]
    detection_tex = output_dir / "detection.tex"
    detection_tex.write_text("\n".join(latex) + "\n")

    cell_width = 18
    header = f"{'Model':<22}{'Metric':<24}" + "".join(
        f"eps={eps:<{cell_width - 4}}" for eps in epsilons
    )
    plain = [header, "-" * len(header)]
    for model_index, (label, _) in enumerate(loaded):
        for metric_index, (
            metric_label,
            _,
            epsilon_dependent,
            _,
        ) in enumerate(detection_metrics):
            model_cell = _plain_label(label) if metric_index == 0 else ""
            slots = data[model_index][metric_index]
            if epsilon_dependent:
                cells = [
                    _plain(
                        *slots[index],
                        best=best[metric_index][index] == model_index,
                    )
                    for index in range(len(epsilons))
                ]
                values = "".join(f"{cell:<{cell_width}}" for cell in cells)
            else:
                values = _plain(
                    *slots[0],
                    best=best[metric_index][0] == model_index,
                )
            plain.append(
                f"{model_cell:<22}{_plain_label(metric_label):<24}{values}"
            )
        plain.append("-" * len(header))
    detection_txt = output_dir / "detection.txt"
    detection_txt.write_text("\n".join(plain) + "\n")

    joint_metrics = [
        ("uq_joint_adv_acc/{eps}", "uq_adv_acc/{eps}", "Rob."),
        (
            f"uq_joint_purify_acc/{{eps}}/{radius}",
            f"uq_purify_acc/{{eps}}/{radius}",
            r"Purif.\ (lk.)",
        ),
        (
            f"sgld_joint_purify_acc/{{eps}}/{sampling_sweep}",
            f"sgld_purify_acc/{{eps}}/{sampling_sweep}",
            rf"Purif.\ (samp., $k{{=}}{sampling_sweep}$)",
        ),
        (
            f"uq_joint_detection/{percentile_key}/{{eps}}",
            f"uq_detection/{percentile_key}/{{eps}}",
            rf"Det.\ rate ($q{{=}}{percentile}\%$)",
        ),
        (
            f"uq_joint_det_err_passed/{percentile_key}/{{eps}}",
            f"uq_det_err_passed/{percentile_key}/{{eps}}",
            rf"Accept.\ err ($q{{=}}{percentile}\%$)",
        ),
    ]
    joint_outputs = []
    combined_plain = []
    alignment = "l" + "c" * len(loaded)
    column_headers = " & ".join(["Metric", *[label for label, _ in loaded]])
    for epsilon in epsilons:
        latex = [
            r"\begin{table}[htbp]",
            r"  \centering",
            f"  \\begin{{tabular}}{{{alignment}}}",
            r"    \toprule",
            f"    {column_headers} \\\\",
            r"    \midrule",
        ]
        plain = [
            f"=== eps={epsilon} ===",
            f"{'Metric':<26}"
            + "".join(f"{_plain_label(label):<18}" for label, _ in loaded),
        ]
        for joint_template, normal_template, metric_label in joint_metrics:
            cells = [
                _delta(
                    frame,
                    joint_template.format(eps=epsilon),
                    normal_template.format(eps=epsilon),
                )
                for _, frame in loaded
            ]
            latex.append(
                "    "
                + " & ".join(
                    [metric_label, *[_tex(*cell, signed=True) for cell in cells]]
                )
                + r" \\"
            )
            plain.append(
                f"{_plain_label(metric_label):<26}"
                + "".join(f"{_plain(*cell, signed=True):<18}" for cell in cells)
            )
        latex += [
            r"    \bottomrule",
            r"  \end{tabular}",
            (
                r"  \caption{Difference (likelihood-aware adaptive minus standard "
                rf"attack) at $\varepsilon={epsilon}$, $q={percentile}\%$, MNIST JEM. "
                r"Negative means the adaptive attack is stronger.}"
            ),
            rf"  \label{{tab:joint_vs_normal_jem_eps{epsilon}}}",
            r"\end{table}",
        ]
        path = output_dir / f"joint_vs_normal_eps{epsilon}.tex"
        path.write_text("\n".join(latex) + "\n")
        joint_outputs.append(path)
        combined_plain += [*plain, ""]
    joint_txt = output_dir / "joint_vs_normal.txt"
    joint_txt.write_text("\n".join(combined_plain) + "\n")
    return [detection_tex, detection_txt, *joint_outputs, joint_txt]
