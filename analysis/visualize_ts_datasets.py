"""Standalone visualization of UCR time-series classification datasets.

Plots individual time-series traces per class (density through overlap) in a
2×2 grid: rows = datasets (ECG200, ItalyPowerDemand), cols = classes.

Usage:
    python -m analysis.visualize_ts_datasets
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UCR_TS_ROOT  = PROJECT_ROOT / ".datasets" / "ucr_ts"
OUTPUT_DIR   = PROJECT_ROOT / "analysis" / "outputs" / "datasets"

DATASETS = [
    {
        "canonical": "ECG200",
        "class_names": ["Normal", "Myocardial Infarction"],
        "row_label": "ECG200 (T=96)",
    },
    {
        "canonical": "ItalyPowerDemand",
        "class_names": ["Oct–Mar", "Apr–Sep"],
        "row_label": "ItalyPowerDemand (T=24)",
    },
]

COLORS = ["steelblue", "tomato"]


def load_ucr(folder: str):
    """Load train + test TSVs, remap labels to 0-indexed integers."""
    ts_dir = UCR_TS_ROOT / folder
    train_raw = np.loadtxt(ts_dir / f"{folder}_TRAIN.tsv")
    test_raw  = np.loadtxt(ts_dir / f"{folder}_TEST.tsv")
    raw = np.vstack([train_raw, test_raw])
    labels_raw = raw[:, 0]
    X = raw[:, 1:]
    unique = sorted(np.unique(labels_raw))
    label_map = {v: i for i, v in enumerate(unique)}
    t = np.array([label_map[v] for v in labels_raw], dtype=int)
    return X, t


def plot_density(ax, X_class, color, alpha):
    """Plot all traces for one class; overlap creates the density effect."""
    steps = np.arange(X_class.shape[1])
    for row in X_class:
        ax.plot(steps, row, color=color, alpha=alpha, linewidth=0.5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(DATASETS)
    n_cols = 2  # one column per class

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(10, 4 * n_rows),
        constrained_layout=True,
        sharey="row",   # same y-scale within each dataset row
    )

    for row_idx, ds in enumerate(DATASETS):
        X, t = load_ucr(ds["canonical"])

        for col_idx, class_name in enumerate(ds["class_names"]):
            ax = axes[row_idx, col_idx]
            mask = t == col_idx
            X_class = X[mask]
            n = mask.sum()

            # Scale alpha so dense classes don't wash out
            alpha = max(0.03, min(0.4, 20.0 / n))

            plot_density(ax, X_class, color=COLORS[col_idx], alpha=alpha)

            ax.set_title(f"{class_name}  (n={n})", fontsize=11)
            ax.set_xlabel("Time step")
            ax.grid(True, alpha=0.25)

            if col_idx == 0:
                ax.set_ylabel(ds["row_label"])

    out_path = OUTPUT_DIR / "ucr_ts_datasets.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
