#!/usr/bin/env python
"""
Visualise raw UCR time-series datasets present in .datasets/.

For each dataset produces a figure with:
  - One row per class
  - Left panel : all samples (low alpha) + class mean
  - Right panel: first 10 individual samples

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
FIGSIZE_PER_CLASS = (11, 2.2)   # per-class row height

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
        n_classes, 2,
        figsize=(FIGSIZE_PER_CLASS[0], FIGSIZE_PER_CLASS[1] * n_classes),
        squeeze=False,
    )
    fig.suptitle(
        f"{PRETTY_NAMES.get(name, name)}  "
        f"(N={len(X)}, T={T}, classes={n_classes}"
        + (f", UCR train={ucr_train_size}" if ucr_train_size else "")
        + ")",
        fontsize=12,
    )

    for row, cls in enumerate(classes):
        colour  = CLASS_COLOURS[cls % len(CLASS_COLOURS)]
        mask    = y == cls
        X_cls   = X[mask]
        n_cls   = len(X_cls)
        mean_ts = X_cls.mean(axis=0)

        # ---- Left: all samples + mean ---------------------------------
        ax_l = axes[row, 0]
        for i in range(n_cls):
            ax_l.plot(t_axis, X_cls[i], color=colour, alpha=0.06, linewidth=0.6)
        ax_l.plot(t_axis, mean_ts, color=colour, linewidth=2.0, label="mean")
        ax_l.set_title(f"Class {cls}  (n={n_cls}) — all samples + mean", fontsize=9)
        ax_l.set_xlabel("Time step")
        ax_l.set_ylabel("Value")
        ax_l.legend(fontsize=8)
        ax_l.grid(True, alpha=0.25)

        # ---- Right: first 10 samples ----------------------------------
        ax_r = axes[row, 1]
        n_show = min(10, n_cls)
        for i in range(n_show):
            ax_r.plot(t_axis, X_cls[i], color=colour,
                      alpha=0.7 + 0.03 * i, linewidth=1.0,
                      label=f"sample {i}" if n_show <= 5 else None)
        ax_r.set_title(f"Class {cls} — first {n_show} samples", fontsize=9)
        ax_r.set_xlabel("Time step")
        ax_r.set_ylabel("Value")
        ax_r.grid(True, alpha=0.25)

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
