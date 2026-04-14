#!/usr/bin/env python
"""
Visualise raw UCR time-series datasets present in .datasets/.

For each dataset produces a figure with one column per class showing
all samples (low alpha) overlaid with the class mean.

Usage
-----
    python analysis/plot_ts_datasets.py          # all available TS datasets
    python analysis/plot_ts_datasets.py ecg200   # single dataset by name
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / ".datasets"
OUT_DIR    = ROOT / "analysis" / "outputs" / "ts_datasets"

DPI     = 150
FIGSIZE_PER_CLASS = (4.5, 3.0)  # per-class column width, height

CLASS_COLOURS = ["steelblue", "darkorange", "seagreen", "firebrick",
                 "mediumpurple", "saddlebrown", "hotpink"]

TS_DATASETS = ["ecg200", "italypowerdemand"]

PRETTY_NAMES = {
    "ecg200":           "ECG200",
    "italypowerdemand": "ItalyPowerDemand",
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(name: str):
    path = DATA_DIR / name / f"{name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    ucr = int(d["ucr_train_size"]) if "ucr_train_size" in d else None
    return d["X"], d["y"], ucr


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_dataset(name: str):
    result = load(name)
    if result is None:
        print(f"  Skipping {name}: .npz not found.")
        return

    X, y, ucr_train_size = result
    classes   = sorted(np.unique(y))
    n_classes = len(classes)
    T         = X.shape[1]
    t_axis    = np.arange(T)

    fig, axes = plt.subplots(
        1, n_classes,
        figsize=(FIGSIZE_PER_CLASS[0] * n_classes, FIGSIZE_PER_CLASS[1]),
        squeeze=False,
    )

    for col, cls in enumerate(classes):
        colour  = CLASS_COLOURS[cls % len(CLASS_COLOURS)]
        mask    = y == cls
        X_cls   = X[mask]
        n_cls   = len(X_cls)
        mean_ts = X_cls.mean(axis=0)

        ax = axes[0, col]
        for i in range(n_cls):
            ax.plot(t_axis, X_cls[i], color=colour, alpha=0.06, linewidth=0.6)
        ax.plot(t_axis, mean_ts, color=colour, linewidth=2.0, label="mean")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise raw UCR time-series datasets."
    )
    parser.add_argument("datasets", nargs="*",
                        help="Dataset name(s) to plot (default: all present).")
    args = parser.parse_args()

    names = args.datasets if args.datasets else TS_DATASETS

    print(f"Plotting {len(names)} dataset(s):\n")
    for name in names:
        print(f"[{name}]")
        plot_dataset(name)

    print("\nDone.")


if __name__ == "__main__":
    main()
