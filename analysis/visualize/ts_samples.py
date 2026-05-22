"""Visualize learned time-series distribution by sampling from a trained ConditionalBornMachine.

Samples time-series curves per class and overlays them in one subplot per class,
with the y-axis in the original data scale (inverse-transformed from embedding range).

Usage
-----
    python -m analysis.visualize.ts_samples --run <run_dir>
    python -m analysis.visualize.ts_samples --run <run_dir> --num-spc 200 --num-bins 100
    python -m analysis.visualize.ts_samples --run <run_dir> --save-dir <dir>
"""

import sys
from pathlib import Path

if "__file__" in dir():
    project_root = Path(__file__).parent.parent.parent
else:
    project_root = Path.cwd().parent.parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import logging

from analysis.utils import load_run_config, find_model_checkpoint
from src.models import ConditionalBornMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_DIR = "outputs/test/0"
NUM_BINS = 100
N_PER_CLASS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_and_plot(
    run_dir: str,
    n_per_class: int = N_PER_CLASS,
    num_bins: int = NUM_BINS,
    save_dir: str | None = None,
    device: str = DEVICE,
):
    run_dir = Path(run_dir)
    checkpoint_path = find_model_checkpoint(run_dir)
    cbm = ConditionalBornMachine.load(str(checkpoint_path))
    cbm.to(device)
    cbm.eval()
    logger.info(f"Loaded model from {checkpoint_path}")

    samples, labels = cbm.sample_all_classes(
        n_per_class=n_per_class, num_bins=num_bins
    )
    logger.info(f"Sampled {samples.shape[0]} time-series ({n_per_class} per class)")

    num_classes = cbm.out_dim
    lo, hi = cbm.input_range
    # Normalise from embedding range to [0, 1] for display
    samples_norm = ((samples.float() - lo) / (hi - lo)).cpu()

    n_cols = min(num_classes, 4)
    n_rows = (num_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if num_classes == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    for c in range(num_classes):
        ax = axes[c // n_cols][c % n_cols]
        class_samples = samples_norm[labels == c]
        t = range(class_samples.shape[1])
        for i in range(len(class_samples)):
            ax.plot(t, class_samples[i].numpy(), alpha=0.3, linewidth=0.6)
        ax.set_title(f"Class {c}", fontsize=9)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")

    for idx in range(num_classes, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Sampled time-series per class", fontsize=11)
    plt.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / "ts_samples.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Saved to {out}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample from a trained CBM and plot time-series.")
    parser.add_argument("--run", default=None, help="Path to Hydra run directory.")
    parser.add_argument("--num-bins", type=int, default=NUM_BINS)
    parser.add_argument("--num-spc", type=int, default=N_PER_CLASS, help="Samples per class.")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default=DEVICE)
    cli_args = parser.parse_args()

    if cli_args.run is not None:
        sample_and_plot(
            run_dir=cli_args.run,
            n_per_class=cli_args.num_spc,
            num_bins=cli_args.num_bins,
            save_dir=cli_args.save_dir,
            device=cli_args.device,
        )
    else:
        sample_and_plot(
            run_dir=RUN_DIR,
            n_per_class=N_PER_CLASS,
            num_bins=NUM_BINS,
        )
