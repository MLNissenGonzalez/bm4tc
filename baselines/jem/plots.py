"""MNIST figures shared by the JEM notebook and command-line workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ALPHA_TICKS = [0, 0.01, 0.1, 0.2, 0.5, 1.0]
LABEL_FS = 18
TICK_FS = 15
LEGEND_FS = 14


def load_alpha_sweeps(csv_by_alpha: dict[float, str | Path]) -> pd.DataFrame:
    frames = []
    for alpha, path in sorted(csv_by_alpha.items()):
        frame = pd.read_csv(path)
        frame["alpha"] = float(alpha)
        frames.append(frame)
    if not frames:
        raise ValueError("No alpha sweep CSVs were provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _aggregate(df: pd.DataFrame, group: str, column: str):
    if column not in df:
        return None
    stats = df.groupby(group)[column].agg(["mean", "std"]).sort_index()
    return (
        stats.index.to_numpy(),
        stats["mean"].to_numpy(),
        stats["std"].fillna(0).to_numpy(),
    )


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise ValueError(
            f"Missing required metric {column!r}. Rerun baselines.jem.sweep "
            "with the locally projected SGLD analysis."
        )


def _line(ax, values, label, color, linestyle="-"):
    if values is None:
        return
    x, mean, std = values
    ax.plot(
        x,
        mean,
        marker="o",
        markersize=4,
        linewidth=1.8,
        color=color,
        linestyle=linestyle,
        label=label,
    )
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def _alpha_axis(ax):
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlim(0, 1)
    ax.set_xticks(ALPHA_TICKS)
    ax.set_xticklabels(
        ["0", "0.01", "0.1", "0.2", "0.5", "1"],
        rotation=45,
        ha="right",
        fontsize=TICK_FS,
    )
    ax.grid(alpha=0.3)


def plot_alpha_curves(
    csv_by_alpha: dict[float, str | Path],
    output_dir: str | Path,
    *,
    epsilon: float = 0.3,
    radius: float = 0.2,
    sampling_sweep: int = 1,
) -> tuple[Path, Path]:
    """MPS-notebook-style accuracy and loss figures over alpha."""
    df = load_alpha_sweeps(csv_by_alpha)
    _require_column(df, f"sgld_purify_acc/{epsilon}/{sampling_sweep}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _line(ax, _aggregate(df, "alpha", "acc"), "Clean", "steelblue")
    _line(
        ax,
        _aggregate(df, "alpha", f"rob/{epsilon}"),
        rf"Rob. ($\varepsilon={epsilon}$)",
        "darkorange",
        "--",
    )
    _line(
        ax,
        _aggregate(df, "alpha", f"uq_purify_acc/{epsilon}/{radius}"),
        "Purif. (lk.)",
        "seagreen",
        "-.",
    )
    _line(
        ax,
        _aggregate(df, "alpha", f"sgld_purify_acc/{epsilon}/{sampling_sweep}"),
        rf"Purif. (samp., $k={sampling_sweep}$)",
        "mediumpurple",
        ":",
    )
    ax.set_xlabel(r"$\alpha$", fontsize=LABEL_FS)
    ax.set_ylabel("Accuracy", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylim(0, 1.05)
    _alpha_axis(ax)
    ax.legend(fontsize=LEGEND_FS)
    fig.tight_layout()
    accuracy_path = output_dir / "alpha_curve_accuracy.png"
    fig.savefig(accuracy_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, left = plt.subplots(figsize=(6, 4.5))
    right = left.twinx()
    _line(left, _aggregate(df, "alpha", "dis_loss"), "Dis. loss", "darkred")
    _line(
        right,
        _aggregate(df, "alpha", "gen_loss"),
        "Gen. loss (CD)",
        "steelblue",
    )
    left.set_xlabel(r"$\alpha$", fontsize=LABEL_FS)
    left.set_ylabel("Dis. loss", color="darkred", fontsize=LABEL_FS)
    right.set_ylabel("Gen. loss (CD)", color="steelblue", fontsize=LABEL_FS)
    left.tick_params(axis="y", labelcolor="darkred", labelsize=TICK_FS)
    right.tick_params(axis="y", labelcolor="steelblue", labelsize=TICK_FS)
    _alpha_axis(left)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], fontsize=LEGEND_FS)
    fig.tight_layout()
    loss_path = output_dir / "alpha_curve_nll.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return accuracy_path, loss_path


def _mean_std(df: pd.DataFrame, column: str, *, flip: bool = False):
    if column not in df:
        return float("nan"), 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if flip:
        values = 1.0 - values
    return (
        float(values.mean()) if len(values) else float("nan"),
        float(values.std()) if len(values) > 1 else 0.0,
    )


def plot_purification_radii(
    models: dict[str, str | Path],
    output: str | Path,
) -> Path:
    """Faceted radius comparison matching the MNIST MPS notebook."""
    frames = {label: pd.read_csv(path) for label, path in models.items()}
    prefix = "uq_purify_acc/"
    radii = sorted(
        {
            column.split("/")[-1]
            for frame in frames.values()
            for column in frame
            if column.startswith(prefix)
        },
        key=float,
    )
    radius_colors = ("seagreen", "darkviolet", "saddlebrown", "teal")
    fig, axes = plt.subplots(
        1, len(frames), figsize=(5.2 * len(frames), 4.5), sharey=True, squeeze=False
    )
    for ax, (label, frame) in zip(axes[0], frames.items()):
        eps = sorted(
            float(column.split("/")[-1])
            for column in frame
            if column.startswith("rob/")
        )
        ax.plot(
            eps,
            [_mean_std(frame, f"rob/{value}")[0] for value in eps],
            color="grey",
            linestyle="--",
            marker="x",
            markersize=5,
            linewidth=1.4,
            label="Rob. (no def.)",
        )
        for index, radius in enumerate(radii):
            means, stds = zip(
                *[
                    _mean_std(frame, f"{prefix}{value}/{radius}")
                    for value in eps
                ]
            )
            color = radius_colors[index % len(radius_colors)]
            ax.plot(
                eps,
                means,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4,
                label=rf"$r={radius}$",
            )
            ax.fill_between(
                eps,
                np.array(means) - stds,
                np.array(means) + stds,
                color=color,
                alpha=0.15,
            )
        ax.set_title(label, fontsize=LABEL_FS)
        ax.set_xlabel(r"attack $\varepsilon$", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=LEGEND_FS - 1)
    axes[0, 0].set_ylabel("Purify accuracy", fontsize=LABEL_FS)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_defense_comparison(
    models: dict[str, str | Path],
    output_dir: str | Path,
    *,
    radius: float = 0.2,
    sampling_sweep: int = 5,
    percentile: int = 10,
) -> list[Path]:
    """One grouped defense bar chart per attack epsilon."""
    frames = {label: pd.read_csv(path) for label, path in models.items()}
    for frame in frames.values():
        for epsilon in (
            float(column.split("/")[-1])
            for column in frame
            if column.startswith("rob/")
        ):
            _require_column(
                frame,
                f"sgld_purify_acc/{epsilon}/{sampling_sweep}",
            )
    first = next(iter(frames.values()))
    eps = sorted(
        float(column.split("/")[-1])
        for column in first
        if column.startswith("rob/")
    )
    defenses = [
        ("Clean", "#9E9E9E", "acc", False),
        ("Rob.", "#FF9800", "rob/{eps}", False),
        ("Purif. (lk.)", "#4CAF50", f"uq_purify_acc/{{eps}}/{radius}", False),
        (
            rf"Purif. (samp., $k={sampling_sweep}$)",
            "#2196F3",
            f"sgld_purify_acc/{{eps}}/{sampling_sweep}",
            False,
        ),
        (
            rf"Accept ($q={percentile}\%$)",
            "#9C27B0",
            f"uq_det_err_passed/{percentile}pct/{{eps}}",
            True,
        ),
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    x = np.arange(len(frames))
    width = 0.8 / len(defenses)
    for epsilon in eps:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for index, (label, color, template, flip) in enumerate(defenses):
            values = [
                _mean_std(frame, template.format(eps=epsilon), flip=flip)
                for frame in frames.values()
            ]
            means, stds = zip(*values)
            ax.bar(
                x + (index - (len(defenses) - 1) / 2) * width,
                means,
                width,
                yerr=stds,
                label=label,
                color=color,
                capsize=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(frames.keys(), fontsize=TICK_FS)
        ax.set_ylabel("Accuracy", fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=LEGEND_FS, loc="upper right")
        fig.tight_layout()
        path = output_dir / f"defense_comparison_eps{epsilon}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_detection_thresholds(
    models: dict[str, tuple[str, str | Path] | str | Path],
    output_dir: str | Path,
    *,
    percentiles: tuple[int, ...] = (1, 5, 10, 20),
) -> list[Path]:
    """Accepted accuracy and detection rate versus clean rejection threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    colors = ("steelblue", "darkorange", "seagreen", "firebrick")
    for model_key, value in models.items():
        if isinstance(value, tuple):
            model_label, csv_path = value
        else:
            model_label, csv_path = model_key, value
        frame = pd.read_csv(csv_path)
        eps = sorted(
            float(column.split("/")[-1])
            for column in frame
            if column.startswith("rob/")
        )
        fig, left = plt.subplots(figsize=(6, 4.5))
        right = left.twinx()
        handles = []
        for color, epsilon in zip(colors, eps):
            q_values = [0, *percentiles]
            acc_mean = [_mean_std(frame, f"uq_adv_acc/{epsilon}")[0]]
            acc_std = [_mean_std(frame, f"uq_adv_acc/{epsilon}")[1]]
            det_mean, det_std = [0.0], [0.0]
            for q in percentiles:
                mean, std = _mean_std(
                    frame, f"uq_det_err_passed/{q}pct/{epsilon}", flip=True
                )
                acc_mean.append(mean)
                acc_std.append(std)
                mean, std = _mean_std(frame, f"uq_detection/{q}pct/{epsilon}")
                det_mean.append(mean)
                det_std.append(std)
            line = left.plot(
                q_values,
                acc_mean,
                color=color,
                marker="o",
                label=rf"$\varepsilon={epsilon}$",
            )[0]
            left.fill_between(
                q_values,
                np.array(acc_mean) - acc_std,
                np.array(acc_mean) + acc_std,
                color=color,
                alpha=0.15,
            )
            right.plot(q_values, det_mean, color=color, linestyle="--", marker="s")
            right.fill_between(
                q_values,
                np.array(det_mean) - det_std,
                np.array(det_mean) + det_std,
                color=color,
                alpha=0.1,
            )
            handles.append(line)
        percent_symbol = r"\%" if plt.rcParams["text.usetex"] else "%"
        left.set_xlabel(
            rf"Rejection threshold $q$ (clean percentile, {percent_symbol})",
            fontsize=LABEL_FS,
        )
        left.set_ylabel("Accuracy on accepted", fontsize=LABEL_FS)
        right.set_ylabel("Detection rate", fontsize=LABEL_FS)
        left.tick_params(axis="both", labelsize=TICK_FS)
        right.tick_params(axis="y", labelsize=TICK_FS)
        left.set_ylim(0, 1.05)
        right.set_ylim(0, 1.05)
        left.grid(alpha=0.3)
        legend = left.legend(
            handles=handles,
            title=model_label,
            fontsize=LEGEND_FS,
            title_fontsize=LEGEND_FS,
            loc="upper left",
        )
        left.add_artist(legend)
        left.legend(
            handles=[
                Line2D([], [], color="gray", marker="o", label="Acc. accepted"),
                Line2D([], [], color="gray", linestyle="--", marker="s", label="Det. rate"),
            ],
            fontsize=LEGEND_FS - 1,
            loc="lower right",
        )
        fig.tight_layout()
        path = output_dir / f"accept_acc_vs_threshold_{model_key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs
