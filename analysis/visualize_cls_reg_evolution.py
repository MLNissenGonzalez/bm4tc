"""Visualize p(c=1|x) and marginal p(x) evolution across cls_reg sweeps.

For each value of ``trainer.{generative,adversarial}.max_epoch`` the same seed
(smallest common seed across all epochs) is used, plus an optional pretrained
baseline column at epoch 0.

    Row 0 – p(c=1|x): continuous [0,1] probability heatmap (viridis)
    Row 1 – Marginal p(x) = sum_c p(x,c)  [viridis, shared colour scale]

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
)
from analysis.utils.resolve import resolve_regime_from_path
from src.models import BornMachine
from src.data.handler import DataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# =============================================================================
# CONFIGURATION — edit for interactive / notebook use
# =============================================================================

SWEEP_DIR = "outputs/cls_reg/gen/fourier/d6D4/moons_4k_XXXX"
MAX_EPOCH_VALUES = None   # None = auto-detect from sweep
FIXED_SEED = None         # None = auto-select smallest common seed
INCLUDE_PRETRAINED = True
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
_cli.add_argument("--seed", type=int, default=None)
_cli.add_argument("--epochs", type=int, nargs="+", default=None)
_cli.add_argument("--no-pretrained", action="store_true")
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
if _cli_args.seed is not None:
    FIXED_SEED = _cli_args.seed
if _cli_args.epochs is not None:
    MAX_EPOCH_VALUES = _cli_args.epochs
if _cli_args.no_pretrained:
    INCLUDE_PRETRAINED = False


# %%
# =============================================================================
# Main logic
# =============================================================================

def build_evolution_figure(
    sweep_dir: str,
    max_epoch_values=None,       # None = auto-detect from df
    resolution: int = 150,
    normalize_joint: bool = True,
    show_data: bool = True,
    device: str = "cpu",
    save_path=None,
    fixed_seed=None,             # None = auto-select smallest common seed
    include_pretrained: bool = True,  # prepend epoch-0 column from model_path
    cmap_cond: str = "viridis",  # colormap for p(c=1|x) row
):
    """Build 2×N figure showing p(c=1|x) + p(x) evolution.

    Args:
        sweep_dir: Path to the cls_reg sweep directory.
        max_epoch_values: Ordered list of max_epoch values to plot. None = auto.
        resolution: Grid resolution per axis.
        normalize_joint: Normalise p(x,c) by partition function.
        show_data: Overlay training data scatter on each panel.
        device: Torch device string.
        save_path: If given, save figure to this path.
        fixed_seed: Seed to use for all epoch columns. None = smallest common.
        include_pretrained: If True, prepend an epoch-0 column from cfg.model_path.
        cmap_cond: Colormap for the p(c=1|x) row.

    Returns:
        Matplotlib Figure.
    """
    sweep_path = Path(sweep_dir)
    if not sweep_path.is_absolute():
        sweep_path = project_root / sweep_path

    device_ = torch.device(device)

    # ------------------------------------------------------------------
    # Step 1: detect regime + config key
    # ------------------------------------------------------------------
    regime = resolve_regime_from_path(str(sweep_path))
    epoch_key = (
        "trainer.adversarial.max_epoch" if regime == "adv"
        else "trainer.generative.max_epoch"
    )

    # ------------------------------------------------------------------
    # Step 2: build metric DataFrame via evaluate_sweep
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
        config_keys=[epoch_key, "tracking.seed"],
    )

    if df.empty:
        raise RuntimeError(f"No runs found in {sweep_path}")

    acc_col = "eval/valid/acc"
    epoch_col = "config/max_epoch"
    seed_col = "config/seed"

    # ------------------------------------------------------------------
    # Step 3: auto-detect max_epoch_values
    # ------------------------------------------------------------------
    if max_epoch_values is None:
        max_epoch_values = sorted(int(v) for v in df[epoch_col].dropna().unique())
        logger.info(f"Auto-detected max_epoch_values: {max_epoch_values}")

    # ------------------------------------------------------------------
    # Step 4: fixed-seed selection
    # ------------------------------------------------------------------
    seeds_per_epoch = {
        e: set(df[df[epoch_col] == e][seed_col].dropna().astype(int))
        for e in max_epoch_values
        if not df[df[epoch_col] == e].empty
    }
    common_seeds = (
        set.intersection(*seeds_per_epoch.values()) if seeds_per_epoch else set()
    )

    if fixed_seed is None:
        if common_seeds:
            fixed_seed = min(common_seeds)
            logger.info(f"Auto-selected fixed_seed={fixed_seed} (smallest common seed)")
        else:
            logger.warning("No common seed found across all epochs; will use first available row per epoch.")

    # ------------------------------------------------------------------
    # Step 5: per max_epoch, select fixed-seed run
    # ------------------------------------------------------------------
    selected_runs = {}  # epoch -> row Series
    for epoch in max_epoch_values:
        sub = df[df[epoch_col] == epoch]
        if sub.empty:
            logger.warning(f"No runs found for max_epoch={epoch}, skipping.")
            continue
        if fixed_seed is not None:
            sub_seed = sub[sub[seed_col] == fixed_seed]
            row = sub_seed.iloc[0] if not sub_seed.empty else sub.iloc[0]
        else:
            row = sub.iloc[0]
        selected_runs[epoch] = row
        logger.info(
            f"max_epoch={epoch}: seed={row.get(seed_col, '?')}, "
            f"val_acc={row.get(acc_col, float('nan')):.4f}, run={row['run_path']}"
        )

    epochs_present = [e for e in max_epoch_values if e in selected_runs]

    if not epochs_present:
        raise RuntimeError("No valid runs found for any max_epoch value.")

    # ------------------------------------------------------------------
    # Step 6: load models and compute distributions
    # ------------------------------------------------------------------
    results = {}  # epoch -> dict with computed arrays

    # --- Pretrained baseline (epoch 0) ---
    if include_pretrained:
        first_run_path = Path(df.iloc[0]["run_path"])
        first_cfg = load_run_config(first_run_path)
        model_path_rel = getattr(first_cfg, "model_path", None)

        if (
            model_path_rel
            and "<" not in str(model_path_rel)
            and "FILL" not in str(model_path_rel)
        ):
            logger.info(f"Loading pretrained model from {model_path_rel} ...")
            bm = BornMachine.load(str(project_root / model_path_rel))
            bm.to(device_)
            dh = DataHandler(first_cfg.dataset)
            dh.load()
            dh.split_and_rescale(bm)
            dh.get_classification_loaders()
            grid_x1, grid_x2, grid_points = make_grid(bm.input_range, resolution)
            conditional = compute_conditional_probs(bm, grid_points, device_)
            joint = compute_joint_probs(bm, grid_points, device_, normalize=normalize_joint)
            results[0] = {
                "conditional": conditional,
                "joint": joint,
                "grid_x1": grid_x1,
                "grid_x2": grid_x2,
                "input_range": bm.input_range,
                "train_data": dh.data["train"] if show_data else None,
                "train_labels": dh.labels["train"] if show_data else None,
                "num_classes": bm.out_dim,
                "label": "Pretrained\n(epoch 0)",
            }
            del bm
            torch.cuda.empty_cache()
        else:
            logger.warning("model_path is a placeholder or missing; skipping pretrained column.")

    # --- Fine-tuned epochs ---
    for epoch in epochs_present:
        row = selected_runs[epoch]
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

        seed = row.get(seed_col, "?")
        acc = row.get(acc_col, float("nan"))
        results[epoch] = {
            "conditional": conditional,
            "joint": joint,
            "grid_x1": grid_x1,
            "grid_x2": grid_x2,
            "input_range": bm.input_range,
            "train_data": dh.data["train"] if show_data else None,
            "train_labels": dh.labels["train"] if show_data else None,
            "num_classes": bm.out_dim,
            "label": f"Epoch {epoch}\n(seed {seed}, acc={acc:.3f})",
        }

        del bm
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 7: build figure
    # ------------------------------------------------------------------
    epochs_to_plot = ([0] if 0 in results else []) + epochs_present
    n_cols = len(epochs_to_plot)

    # Shared colour scale for p(x) row
    global_joint_max = max(
        r["joint"].sum(dim=1).max().item() for r in results.values()
    )

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), squeeze=False)

    last_pcm = None
    for col_idx, epoch in enumerate(epochs_to_plot):
        r = results[epoch]
        lo, hi = r["input_range"]
        res = r["grid_x1"].shape[0]
        num_classes = r["num_classes"]

        # ---------- Row 0: p(c=1|x) continuous heatmap ----------
        ax0 = axes[0, col_idx]
        prob1 = r["conditional"].numpy()[:, 1].reshape(res, res)
        ax0.pcolormesh(
            r["grid_x1"], r["grid_x2"], prob1,
            cmap=cmap_cond, shading="auto", vmin=0.0, vmax=1.0,
        )
        ax0.set_title(r["label"])
        ax0.set_xlabel("$x_1$")
        ax0.set_ylabel("$x_2$")
        ax0.set_xlim(lo, hi)
        ax0.set_ylim(lo, hi)
        ax0.set_aspect("equal")

        if show_data and r["train_data"] is not None:
            _overlay_data(ax0, r["train_data"], r["train_labels"], num_classes)

        # ---------- Row 1: marginal p(x) ----------
        ax1 = axes[1, col_idx]
        marginal = r["joint"].numpy().sum(axis=1).reshape(res, res)
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

        last_pcm = pcm

    # Row labels
    axes[0, 0].annotate(
        "$p(c=1\\,|\\,x)$",
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

    # Colorbars: one per row
    sm = plt.cm.ScalarMappable(cmap=cmap_cond, norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=axes[0, :], shrink=0.8, pad=0.02, label="$p(c=1\\,|\\,x)$")
    fig.colorbar(last_pcm, ax=axes[1, :], shrink=0.8, pad=0.02, label="$p(x)$")

    seed_str = f"seed={fixed_seed}" if fixed_seed is not None else "mixed seeds"
    fig.suptitle(
        f"cls_reg: p(c=1|x) and p(x) Evolution ({seed_str})\n{Path(sweep_dir).name}",
        fontsize=13,
    )
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
        fixed_seed=FIXED_SEED,
        include_pretrained=INCLUDE_PRETRAINED,
    )
    plt.show()
