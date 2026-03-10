"""Visualize decision boundary and marginal p(x) evolution across cls_reg sweeps.

For each value of ``trainer.generative.max_epoch`` the best seed (by validation
accuracy) is selected and two panels are produced:

    Row 0 – Decision boundary: argmax_c p(c|x)   [tab10 colourmap]
    Row 1 – Marginal p(x) = sum_c p(x,c)          [viridis, shared colour scale]

Usage::

    python analysis/visualize_cls_reg_evolution.py <sweep_dir> [options]

"""

# %%
import sys
import argparse
from pathlib import Path

if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

from analysis.visualize_distributions import (
    make_grid,
    compute_conditional_probs,
    compute_joint_probs,
    _overlay_data,
)
from analysis.utils import (
    load_run_config,
    find_model_checkpoint,
    EvalConfig,
    evaluate_sweep,
    get_best_run,
)
from src.models import BornMachine
from src.data.handler import DataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# =============================================================================
# CONFIGURATION — edit for interactive / notebook use
# =============================================================================

SWEEP_DIR = "outputs/cls_reg/gen/fourier/d6D4/moons_4k_XXXX"
MAX_EPOCH_VALUES = [1, 5, 10, 50, 100]
RESOLUTION = 150
NORMALIZE_JOINT = True
SHOW_DATA = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = None  # e.g. "analysis/outputs/cls_reg_evo/moons_4k.png"

# --- CLI overrides (no-op when running interactively) ---
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("sweep_dir", nargs="?", default=None)
_cli.add_argument("--save", default=None)
_cli.add_argument("--resolution", type=int, default=None)
_cli.add_argument("--no-data", action="store_true")
_cli.add_argument("--no-normalize-joint", action="store_true")
_cli.add_argument("--device", default=None)
_cli_args, _ = _cli.parse_known_args()

if _cli_args.sweep_dir is not None:
    SWEEP_DIR = _cli_args.sweep_dir
if _cli_args.save is not None:
    SAVE_PATH = _cli_args.save
if _cli_args.resolution is not None:
    RESOLUTION = _cli_args.resolution
if _cli_args.no_data:
    SHOW_DATA = False
if _cli_args.no_normalize_joint:
    NORMALIZE_JOINT = False
if _cli_args.device is not None:
    DEVICE = _cli_args.device


# %%
# =============================================================================
# Main logic
# =============================================================================

def build_evolution_figure(
    sweep_dir: str,
    max_epoch_values=None,
    resolution: int = 150,
    normalize_joint: bool = True,
    show_data: bool = True,
    device: str = "cpu",
    save_path=None,
):
    """Build 2×N figure showing decision boundary + p(x) evolution.

    Args:
        sweep_dir: Path to the cls_reg sweep directory.
        max_epoch_values: Ordered list of max_epoch values to plot.
        resolution: Grid resolution per axis.
        normalize_joint: Normalise p(x,c) by partition function.
        show_data: Overlay training data scatter on each panel.
        device: Torch device string.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    if max_epoch_values is None:
        max_epoch_values = [1, 5, 10, 50, 100]

    sweep_path = Path(sweep_dir)
    if not sweep_path.is_absolute():
        sweep_path = project_root / sweep_path

    device_ = torch.device(device)

    # ------------------------------------------------------------------
    # Step 1: build metric DataFrame via evaluate_sweep
    # ------------------------------------------------------------------
    eval_cfg = EvalConfig(
        compute_acc=True,
        compute_rob=False,
        compute_mia=False,
        compute_cls_loss=False,
        compute_gen_loss=False,
        compute_fid=False,
        compute_uq=False,
        splits=["valid"],
        device=device,
    )

    logger.info(f"Evaluating sweep at {sweep_path} ...")
    df = evaluate_sweep(
        sweep_dir=str(sweep_path),
        eval_cfg=eval_cfg,
        config_keys=["trainer.generative.max_epoch", "tracking.seed"],
    )

    if df.empty:
        raise RuntimeError(f"No runs found in {sweep_path}")

    acc_col = "eval/valid/acc"
    epoch_col = "config/max_epoch"
    seed_col = "config/seed"

    # ------------------------------------------------------------------
    # Step 2: per max_epoch, select best seed
    # ------------------------------------------------------------------
    best_runs = {}  # max_epoch -> row Series
    for epoch in max_epoch_values:
        sub = df[df[epoch_col] == epoch]
        if sub.empty:
            logger.warning(f"No runs found for max_epoch={epoch}, skipping.")
            continue
        best = get_best_run(sub, acc_col, minimize=False)
        if best is None:
            logger.warning(f"Could not determine best run for max_epoch={epoch}, skipping.")
            continue
        best_runs[epoch] = best
        logger.info(
            f"max_epoch={epoch}: best seed={best.get(seed_col, '?')}, "
            f"val_acc={best.get(acc_col, float('nan')):.4f}, run={best['run_path']}"
        )

    epochs_present = [e for e in max_epoch_values if e in best_runs]
    n_cols = len(epochs_present)

    if n_cols == 0:
        raise RuntimeError("No valid runs found for any max_epoch value.")

    # ------------------------------------------------------------------
    # Step 3: load models and compute distributions
    # ------------------------------------------------------------------
    results = {}  # epoch -> dict with computed arrays

    for epoch in epochs_present:
        row = best_runs[epoch]
        run_path = Path(row["run_path"])

        logger.info(f"Loading model for max_epoch={epoch} from {run_path} ...")
        cfg = load_run_config(run_path)
        checkpoint = find_model_checkpoint(run_path)
        bm = BornMachine.load(str(checkpoint))
        bm.to(device_)

        dh = DataHandler(cfg.dataset)
        dh.load()
        dh.split_and_rescale(bm)
        dh.get_classification_loaders()

        grid_x1, grid_x2, grid_points = make_grid(bm.input_range, resolution)

        logger.info(f"  Computing p(c|x) ...")
        conditional = compute_conditional_probs(bm, grid_points, device_)

        logger.info(f"  Computing p(x,c) ...")
        joint = compute_joint_probs(bm, grid_points, device_, normalize=normalize_joint)

        results[epoch] = {
            "conditional": conditional,
            "joint": joint,
            "grid_x1": grid_x1,
            "grid_x2": grid_x2,
            "input_range": bm.input_range,
            "train_data": dh.data["train"] if show_data else None,
            "train_labels": dh.labels["train"] if show_data else None,
            "num_classes": bm.out_dim,
            "seed": row.get(seed_col, "?"),
            "acc": row.get(acc_col, float("nan")),
        }

        # Free GPU memory between runs
        del bm
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: build figure
    # ------------------------------------------------------------------
    # Shared colour scale for p(x) row: max over all epochs
    global_joint_max = max(
        r["joint"].sum(dim=1).max().item() for r in results.values()
    )

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), squeeze=False)

    for col_idx, epoch in enumerate(epochs_present):
        r = results[epoch]
        cond_np = r["conditional"].numpy()
        joint_np = r["joint"].numpy()
        resolution_actual = r["grid_x1"].shape[0]
        lo, hi = r["input_range"]
        num_classes = r["num_classes"]

        # ---------- Row 0: decision boundary ----------
        ax0 = axes[0, col_idx]
        decision = np.argmax(cond_np, axis=1).reshape(resolution_actual, resolution_actual)
        cmap_disc = plt.colormaps.get_cmap("tab10").resampled(num_classes)
        ax0.pcolormesh(
            r["grid_x1"], r["grid_x2"], decision,
            cmap=cmap_disc, shading="auto",
            vmin=-0.5, vmax=num_classes - 0.5,
        )
        ax0.set_title(f"max_epoch = {epoch}\nseed {r['seed']}, acc={r['acc']:.3f}")
        ax0.set_xlabel("$x_1$")
        ax0.set_ylabel("$x_2$")
        ax0.set_xlim(lo, hi)
        ax0.set_ylim(lo, hi)
        ax0.set_aspect("equal")

        if show_data and r["train_data"] is not None:
            _overlay_data(ax0, r["train_data"], r["train_labels"], num_classes)

        # ---------- Row 1: marginal p(x) ----------
        ax1 = axes[1, col_idx]
        marginal = joint_np.sum(axis=1).reshape(resolution_actual, resolution_actual)
        pcm = ax1.pcolormesh(
            r["grid_x1"], r["grid_x2"], marginal,
            cmap="viridis", shading="auto",
            vmin=0.0, vmax=global_joint_max,
        )
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_aspect("equal")

        if show_data and r["train_data"] is not None:
            _overlay_data(ax1, r["train_data"], r["train_labels"], num_classes)

        # Store last pcm for shared colourbar
        last_pcm = pcm

    # Row labels
    axes[0, 0].annotate(
        "Decision boundary\nargmax p(c|x)",
        xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - 10, 0),
        xycoords=axes[0, 0].yaxis.label, textcoords="offset points",
        ha="right", va="center", fontsize=9,
    )
    axes[1, 0].annotate(
        r"$p(x) = \sum_c p(x,c)$",
        xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - 10, 0),
        xycoords=axes[1, 0].yaxis.label, textcoords="offset points",
        ha="right", va="center", fontsize=9,
    )

    # Shared colourbar for p(x) row
    fig.colorbar(last_pcm, ax=axes[1, :], shrink=0.8, pad=0.02, label="p(x)")

    fig.suptitle("cls_reg: Decision Boundary and Marginal p(x) vs Generative Budget", fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    return fig


# %%
if __name__ == "__main__":
    fig = build_evolution_figure(
        sweep_dir=SWEEP_DIR,
        max_epoch_values=MAX_EPOCH_VALUES,
        resolution=RESOLUTION,
        normalize_joint=NORMALIZE_JOINT,
        show_data=SHOW_DATA,
        device=DEVICE,
        save_path=SAVE_PATH,
    )
    plt.show()
