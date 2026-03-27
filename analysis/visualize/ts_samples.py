"""Visualize learned time-series distribution by sampling from a trained BornMachine.

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

# Colour cycle for classes (wraps for more than 5 classes)
_CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _inverse_transform(scaler, X_np: np.ndarray) -> np.ndarray:
    """Inverse-transform X from embedding range back to original data range.

    Handles both sklearn MinMaxScaler (has inverse_transform) and the
    project's LinearScaler (stores _scale / _offset).
    """
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(X_np)
    return (X_np - scaler._offset) / scaler._scale


def sample_ts(
    run_dir: str | Path,
    num_spc: int = 100,
    num_bins: int = 50,
    batch_spc: int = 10,
    device: str = "cpu",
    save_path: Path | None = None,
) -> plt.Figure:
    """Sample time-series curves from a trained BornMachine and plot them.

    Args:
        run_dir: Hydra run directory containing .hydra/config.yaml and models/.
        num_spc: Total curves to sample per class.
        num_bins: Discretization bins per feature (higher = smoother but slower).
        batch_spc: Samples per batch (keep small to avoid OOM; default 5).
        device: Torch device string.
        save_path: If given, save the figure to this path.

    Returns:
        Matplotlib Figure with one subplot per class, each showing overlaid curves.
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

    sampling_cfg = SamplingConfig(
        method="multinomial",
        num_spc=num_spc,
        num_bins=num_bins,
        batch_spc=batch_spc,
    )

    logger.info(
        f"Sampling {num_spc} curves per class "
        f"(num_bins={num_bins}, batch_spc={batch_spc})..."
    )
    with torch.no_grad():
        samples = bm.generator.sample_all_classes(sampling_cfg)
    # samples: (num_spc, num_cls, T) on CPU

    T = samples.shape[2]
    samples_np = samples.numpy()  # (num_spc, num_cls, T)

    # Inverse-transform: operate on (num_spc*num_cls, T) then reshape back
    flat = samples_np.reshape(-1, T)
    flat_orig = _inverse_transform(scaler, flat)
    samples_orig = flat_orig.reshape(num_spc, num_cls, T)

    # Shared y-limits across all subplots
    y_min = float(samples_orig.min())
    y_max = float(samples_orig.max())
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    ylim = (y_min - y_pad, y_max + y_pad)

    # Alpha scales down gracefully for many curves
    alpha = min(0.6, 15.0 / num_spc)

    t = np.arange(T)
    fig, axes = plt.subplots(
        1, num_cls,
        figsize=(4.5 * num_cls, 3.5),
        sharey=True,
    )
    if num_cls == 1:
        axes = [axes]

    for cls_idx, ax in enumerate(axes):
        color = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]
        for s in range(num_spc):
            ax.plot(t, samples_orig[s, cls_idx], color=color, alpha=alpha,
                    linewidth=0.7)
        ax.set_title(f"Class {cls_idx}", fontsize=10)
        ax.set_xlabel("time step", fontsize=8)
        ax.set_ylim(ylim)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("value", fontsize=8)
    dataset_name = getattr(cfg.dataset, "name", str(run_dir.name))
    fig.suptitle(f"Sampled time series — {dataset_name}", fontsize=10)
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
        description="Sample and visualize time-series curves from a trained BornMachine."
    )
    parser.add_argument("--run", required=True, help="Path to Hydra run directory.")
    parser.add_argument("--num-spc", type=int, default=10,
                        help="Curves to sample per class (default: 100).")
    parser.add_argument("--num-bins", type=int, default=10,
                        help="Discretization bins per feature (default: 100).")
    parser.add_argument("--batch-spc", type=int, default=10,
                        help="Samples per batch — lower reduces memory use (default: 5).")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save ts_samples.png. Defaults to "
                             "analysis/outputs/visualize/<run_relative>/.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if args.save_dir is not None:
        save_path = Path(args.save_dir) / "ts_samples.png"
    else:
        try:
            rel = run_dir.resolve().relative_to((project_root / "outputs").resolve())
            save_path = project_root / "analysis" / "outputs" / "visualize" / rel / "ts_samples.png"
        except ValueError:
            save_path = project_root / "analysis" / "outputs" / "visualize" / "ts_samples.png"

    fig = sample_ts(
        run_dir=run_dir,
        num_spc=args.num_spc,
        num_bins=args.num_bins,
        batch_spc=args.batch_spc,
        device=args.device,
        save_path=save_path,
    )
    plt.show()
