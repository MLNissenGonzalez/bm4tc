"""Visualize learned MNIST distribution by sampling from a trained ConditionalBornMachine.

Samples a few images per digit class and displays one randomly-chosen
sample per class in a 2×5 grid (digits 0–9).

Usage
-----
    python -m analysis.visualize.mnist_samples --run <run_dir>
    python -m analysis.visualize.mnist_samples --run <run_dir> --num-bins 100 --binarize
    python -m analysis.visualize.mnist_samples --run <run_dir> --save-dir <dir>
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

import math
import torch
import matplotlib.pyplot as plt
import logging

from analysis.utils import load_run_config, find_model_checkpoint
from src.model import ConditionalBornMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_DIR = "outputs/test/0"
NUM_BINS = 100
N_PER_CLASS = 8
BINARIZE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_and_plot(
    run_dir: str,
    n_per_class: int = N_PER_CLASS,
    num_bins: int = NUM_BINS,
    binarize: bool = BINARIZE,
    save_dir: str | None = None,
    device: str = DEVICE,
):
    run_dir = Path(run_dir)
    checkpoint_path = find_model_checkpoint(run_dir)
    cbm = ConditionalBornMachine.load(str(checkpoint_path))
    cbm.to(device)
    cbm.eval()
    cbm.renormalize_(log_target=0.0)
    logger.info(f"Loaded model from {checkpoint_path}")

    samples, labels = cbm.sample_all_classes(
        n_per_class=n_per_class, num_bins=num_bins
    )
    logger.info(f"Sampled {samples.shape[0]} images ({n_per_class} per class)")

    num_classes = cbm.out_dim
    img_dim = math.isqrt(samples.shape[1])

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for c in range(min(num_classes, 10)):
        ax = axes[c // 5, c % 5]
        class_samples = samples[labels == c]
        idx = torch.randint(len(class_samples), (1,)).item()
        img = class_samples[idx].float()
        lo, hi = cbm.input_range
        img = (img - lo) / (hi - lo)
        if binarize:
            img = (img > 0.5).float()
        ax.imshow(img.reshape(img_dim, img_dim), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {c}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Sampled MNIST digits (one per class)", fontsize=11)
    plt.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / "mnist_samples.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Saved to {out}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample from a trained CBM and plot MNIST digits.")
    parser.add_argument("--run", default=None, help="Path to Hydra run directory.")
    parser.add_argument("--num-bins", type=int, default=NUM_BINS)
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default=DEVICE)
    cli_args = parser.parse_args()

    if cli_args.run is not None:
        sample_and_plot(
            run_dir=cli_args.run,
            n_per_class=cli_args.n_per_class,
            num_bins=cli_args.num_bins,
            binarize=cli_args.binarize,
            save_dir=cli_args.save_dir,
            device=cli_args.device,
        )
    else:
        sample_and_plot(
            run_dir=RUN_DIR,
            n_per_class=N_PER_CLASS,
            num_bins=NUM_BINS,
            binarize=BINARIZE,
        )
