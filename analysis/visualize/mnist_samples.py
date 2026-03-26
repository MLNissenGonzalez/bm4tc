"""Visualize learned MNIST distribution by sampling from a trained BornMachine.

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

import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

from analysis.utils import load_run_config, find_model_checkpoint
from src.models import BornMachine
from src.data.handler import DataHandler
from src.utils.schemas import SamplingConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _inverse_transform(scaler, X_np: np.ndarray) -> np.ndarray:
    """Inverse-transform X from embedding range back to original data range.

    Handles both sklearn MinMaxScaler (has inverse_transform) and the
    project's LinearScaler (stores _scale / _offset).
    """
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(X_np)
    return (X_np - scaler._offset) / scaler._scale


def sample_mnist(
    run_dir: str | Path,
    num_bins: int = 10,
    num_spc: int = 3,
    binarize: bool = False,
    device: str = "cpu",
    save_path: Path | None = None,
    rng_seed: int = 0,
) -> plt.Figure:
    """Sample digit images from a trained MNIST BornMachine and plot them.

    Args:
        run_dir: Hydra run directory containing .hydra/config.yaml and models/.
        num_bins: Number of discretization bins per pixel during sampling.
            10 is usually sufficient for MNIST; 100 gives smoother images.
        num_spc: Number of samples to draw per class (3 is enough for display).
        binarize: If True, threshold sampled pixel values at 0.5.
        device: Torch device string.
        save_path: If given, save the figure to this path.
        rng_seed: Random seed for picking one sample per class.

    Returns:
        Matplotlib Figure with a 2×5 grid of sampled digits (one per class).
    """
    run_dir = Path(run_dir)
    dev = torch.device(device)

    logger.info(f"Loading config from {run_dir}")
    cfg = load_run_config(run_dir)

    checkpoint_path = find_model_checkpoint(run_dir)
    logger.info(f"Loading model from {checkpoint_path}")
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(dev)
    bm.eval()

    logger.info(f"Loading dataset: {cfg.dataset.name}")
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)
    scaler = datahandler.scaler

    num_cls = bm.out_dim
    logger.info(f"num_classes={num_cls}, input_range={bm.input_range}")

    # batch_spc=1: MNIST has 784 features; even one sample is memory-intensive
    sampling_cfg = SamplingConfig(
        method="secant",
        num_spc=num_spc,
        num_bins=num_bins,
        batch_spc=1,
    )

    logger.info(f"Sampling {num_spc} images per class with num_bins={num_bins}...")
    with torch.no_grad():
        samples = bm.generator.sample_all_classes(sampling_cfg)
    # samples: (num_spc, num_cls, 784) on CPU

    samples_np = samples.numpy()  # (num_spc, num_cls, 784)
    flat = samples_np.reshape(-1, 784)
    flat_orig = _inverse_transform(scaler, flat)
    flat_orig = np.clip(flat_orig, 0.0, 1.0)
    samples_orig = flat_orig.reshape(num_spc, num_cls, 784)

    rng = np.random.default_rng(rng_seed)

    fig, axes = plt.subplots(2, 5, figsize=(7, 3.5))
    for cls_idx, ax in enumerate(axes.flat):
        idx = int(rng.integers(0, num_spc))
        img = samples_orig[idx, cls_idx].reshape(28, 28)
        if binarize:
            img = (img > 0.5).astype(np.float32)
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(str(cls_idx), fontsize=9)
        ax.axis("off")

    fig.suptitle("Sampled MNIST digits", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample and visualize MNIST digits from a trained BornMachine."
    )
    parser.add_argument("--run", required=True, help="Path to Hydra run directory.")
    parser.add_argument("--num-bins", type=int, default=10,
                        help="Discretization bins per pixel (default: 10).")
    parser.add_argument("--num-spc", type=int, default=3,
                        help="Samples to draw per class (default: 3).")
    parser.add_argument("--binarize", action="store_true",
                        help="Threshold pixel values at 0.5 before display.")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save mnist_samples.png. Defaults to "
                             "analysis/outputs/visualize/<run_relative>/.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for picking which sample to display per class.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if args.save_dir is not None:
        save_path = Path(args.save_dir) / "mnist_samples.png"
    else:
        try:
            rel = run_dir.resolve().relative_to((project_root / "outputs").resolve())
            save_path = project_root / "analysis" / "outputs" / "visualize" / rel / "mnist_samples.png"
        except ValueError:
            save_path = project_root / "analysis" / "outputs" / "visualize" / "mnist_samples.png"

    fig = sample_mnist(
        run_dir=run_dir,
        num_bins=args.num_bins,
        num_spc=args.num_spc,
        binarize=args.binarize,
        device=args.device,
        save_path=save_path,
        rng_seed=args.seed,
    )
    plt.show()
