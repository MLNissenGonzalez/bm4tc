"""
cls_reg Analysis Tool — local model evaluation.

Loads saved model checkpoints from each post-trained run in a cls_reg sweep
directory, evaluates clean accuracy, adversarial robustness (with and without
purification), cls_loss, and gen_loss, then produces summary CSVs and plots
grouped by max_epoch. A pretrained baseline (max_epoch=0) is included so that
per-seed diff columns (posttraining − pretrained) can be computed.

Usage:
    python analysis/cls_reg_analysis.py <sweep_dir> [--device cuda] [--force]

Example:
    python analysis/cls_reg_analysis.py \\
        outputs/cls_reg/gen/legendre/d10D6/circles_4k_1303 \\
        --device cpu --force
"""

import sys
import argparse
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from analysis.utils.evaluate import EvalConfig, evaluate_sweep, evaluate_pretrained_model
from analysis.utils.mia_utils import load_run_config
from analysis.utils.resolve import (
    resolve_regime_from_path,
    resolve_embedding_from_path,
    embedding_range_size,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

STRENGTH_FRACTIONS = [0.1, 0.2, 0.5]   # relative to embedding input range
PURIF_RADIUS_FRAC  = 0.1               # purification radius fraction
EVAL_SPLIT         = "test"

# =============================================================================
# CLI
# =============================================================================

_cli = argparse.ArgumentParser(
    description="cls_reg analysis: local model evaluation across max_epoch sweep.",
)
_cli.add_argument("sweep_dir", help="Path to sweep output directory")
_cli.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
_cli.add_argument("--force", action="store_true", help="Re-evaluate even if CSV exists")
args = _cli.parse_args()

sweep_dir = Path(args.sweep_dir).resolve()
device    = args.device
force     = args.force

# =============================================================================
# PATH PARSING  →  regime + output directory
# =============================================================================

def _parse_sweep_dir(sweep_dir: Path):
    """Parse sweep dir path into (regime, embedding, arch, dataset, date).

    Expected structure:
        .../{experiment}/{regime}/{embedding}/{arch}/{dataset}_{date}

    e.g. outputs/cls_reg/gen/legendre/d10D6/circles_4k_1303
      -> regime='gen', embedding='legendre', arch='d10D6',
         dataset='circles_4k', date='1303'
    """
    parts = sweep_dir.parts
    last  = parts[-1]   # e.g. "circles_4k_1303"
    arch  = parts[-2]   # e.g. "d10D6"

    # Detect embedding from path
    embedding = resolve_embedding_from_path(str(sweep_dir))
    if embedding is None:
        embedding = parts[-3] if len(parts) >= 3 else "unknown"

    # Detect regime from path
    regime = resolve_regime_from_path(str(sweep_dir))
    if regime is None:
        regime = "gen"

    # Split last component into dataset + date (trailing 4-digit tokens)
    tokens     = last.split("_")
    date_parts = []
    for token in reversed(tokens):
        if re.match(r"^\d{4}$", token):
            date_parts.insert(0, token)
        else:
            break

    if not date_parts:
        raise ValueError(f"Could not extract date from sweep dir last component: '{last}'")

    dataset = "_".join(tokens[: len(tokens) - len(date_parts)])
    date    = "_".join(date_parts)

    return regime, embedding, arch, dataset, date


regime, embedding, arch, dataset, date = _parse_sweep_dir(sweep_dir)
print(f"Parsed sweep dir:")
print(f"  regime={regime}, embedding={embedding}, arch={arch}, dataset={dataset}, date={date}")

range_size = embedding_range_size(embedding)
print(f"  range_size={range_size}")

# Epoch config key depends on regime
epoch_key = (
    "trainer.adversarial.max_epoch" if regime == "adv"
    else "trainer.generative.max_epoch"
)
print(f"  epoch_key={epoch_key}")

output_dir = (
    project_root / "analysis" / "outputs"
    / "seed_sweep" / "cls_reg" / regime / embedding / arch / f"{dataset}_{date}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# =============================================================================
# EVAL CONFIG
# =============================================================================

def build_eval_cfg(range_size: float, device: str) -> EvalConfig:
    """Build a single EvalConfig used for model evaluation."""
    strengths = [f * range_size for f in STRENGTH_FRACTIONS]
    return EvalConfig(
        compute_acc=True,
        compute_rob=True,
        compute_mia=False,
        compute_cls_loss=True,
        compute_gen_loss=True,
        compute_fid=False,
        compute_uq=True,
        splits=[EVAL_SPLIT],
        evasion_override={
            "method": "PGD",
            "norm": 2,
            "num_steps": 20,
            "strengths": strengths,
        },
        uq_config={
            "radii": [PURIF_RADIUS_FRAC * range_size],
            "percentiles": [1, 5, 10, 20],
            "norm": 2,
            "num_steps": 20,
            "attack_method": "PGD",
            "attack_strengths": strengths,
        },
        device=device,
    )

# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

eval_csv = output_dir / "evaluation_data.csv"

if eval_csv.exists() and not force:
    print(f"\nevaluation_data.csv already exists (use --force to re-evaluate). Loading...")
    df = pd.read_csv(eval_csv)
    print(f"Loaded {len(df)} rows from {eval_csv}")
else:
    eval_cfg = build_eval_cfg(range_size, device)

    # 2a. Evaluate all post-trained runs
    df = evaluate_sweep(
        sweep_dir=str(sweep_dir),
        eval_cfg=eval_cfg,
        config_keys=[epoch_key, "tracking.seed"],
    )

    if df.empty:
        print(f"No valid runs found in {sweep_dir}. Exiting.")
        sys.exit(1)

    # Rename config columns to short names
    df.rename(
        columns={
            "config/max_epoch": "max_epoch",
            "config/seed": "seed",
        },
        inplace=True,
    )

    # 2b. Evaluate pretrained baseline (once, using run 0 as reference)
    run_dirs = sorted(
        [d for d in sweep_dir.iterdir()
         if d.is_dir() and (d / ".hydra" / "config.yaml").exists()],
        key=lambda d: d.name,
    )
    ref_run = run_dirs[0]
    cfg0    = load_run_config(ref_run)
    pretrained_path = OmegaConf.select(cfg0, "model_path")

    if pretrained_path is None:
        logger.warning("Could not find model_path in run config; skipping pretrained baseline.")
        pre_metrics = {}
    else:
        print(f"\nEvaluating pretrained baseline: {pretrained_path}")
        try:
            pre_metrics = evaluate_pretrained_model(pretrained_path, ref_run, eval_cfg)
        except Exception as e:
            logger.warning(f"Pretrained evaluation failed: {e}")
            pre_metrics = {}

    # 2c. Add one pretrained row per unique seed (all share the same model)
    if pre_metrics:
        pre_rows = []
        for seed in df["seed"].dropna().unique():
            row = {
                "max_epoch":  0,
                "seed":       int(seed),
                "run_path":   str(pretrained_path),
                "range_size": range_size,
            }
            row.update(pre_metrics)
            pre_rows.append(row)
        df = pd.concat([pd.DataFrame(pre_rows), df], ignore_index=True)

    # Ensure range_size column present
    if "range_size" not in df.columns:
        df["range_size"] = range_size

    df.to_csv(eval_csv, index=False)
    print(f"\nSaved evaluation_data.csv ({len(df)} rows)")

# =============================================================================
# DIFF COLUMNS  (posttraining − pretrained, per seed)
# =============================================================================

pre_df = df[df["max_epoch"] == 0].copy()
metric_cols_base = [c for c in df.columns if c.startswith("eval/")]

if not pre_df.empty and "seed" in pre_df.columns:
    pre_df_indexed = pre_df.set_index("seed")

    for idx, row in df[df["max_epoch"] != 0].iterrows():
        seed = row.get("seed")
        if seed is None or seed not in pre_df_indexed.index:
            continue
        pre_row = pre_df_indexed.loc[seed]
        # If multiple pretrained rows for same seed, take the first
        if isinstance(pre_row, pd.DataFrame):
            pre_row = pre_row.iloc[0]
        for col in metric_cols_base:
            if col not in df.columns:
                continue
            diff_col = f"diff/{col[len('eval/'):]}"
            try:
                df.at[idx, diff_col] = float(row[col]) - float(pre_row[col])
            except (TypeError, ValueError, KeyError):
                df.at[idx, diff_col] = np.nan

# =============================================================================
# SUMMARY CSV  (mean ± std per max_epoch)
# =============================================================================

metric_cols = [c for c in df.columns if c not in ("max_epoch", "seed", "run_path", "range_size", "run_name")]
max_epochs  = sorted(df["max_epoch"].dropna().unique())

agg_rows = []
for ep in max_epochs:
    group = df[df["max_epoch"] == ep]
    agg   = {"max_epoch": int(ep), "n_seeds": len(group)}
    for col in metric_cols:
        vals = pd.to_numeric(group[col], errors="coerce").dropna()
        agg[f"{col}/mean"] = vals.mean()  if len(vals) > 0 else np.nan
        agg[f"{col}/std"]  = vals.std()   if len(vals) > 1 else np.nan
    agg_rows.append(agg)

summary_df = pd.DataFrame(agg_rows)
summary_csv = output_dir / "summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary.csv ({len(summary_df)} rows, {len(max_epochs)} max_epoch values)")

# =============================================================================
# PLOTTING HELPERS
# =============================================================================

_x = np.array(max_epochs, dtype=float)


def _plot_line(ax, summary_df, col, label, color):
    """Plot mean ± std fill_between for a column from summary_df."""
    mean_col = f"{col}/mean"
    std_col  = f"{col}/std"
    if mean_col not in summary_df.columns:
        return False
    means = summary_df[mean_col].values
    stds  = summary_df[std_col].fillna(0).values
    ax.plot(_x, means, "o-", color=color, linewidth=2, markersize=5, label=label)
    ax.fill_between(_x, means - stds, means + stds, alpha=0.2, color=color)
    return True


def _find_col(df_or_summary, pattern):
    """Return first column matching a regex pattern, or None."""
    for col in df_or_summary.columns:
        if re.search(pattern, col):
            return col
    return None


def _strip_mean(col: str) -> str:
    """Remove trailing '/mean' from column name."""
    return col[: -len("/mean")] if col.endswith("/mean") else col


def _plot_diff_line(ax, summary_df, diff_col, label, color):
    """Like _plot_line but adds a y=0 reference line."""
    ok = _plot_line(ax, summary_df, diff_col, label, color)
    if ok:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    return ok


def _plot_seed_lines(ax, df, col, color):
    """Overlay thin per-seed lines for individual trajectory visibility."""
    if col not in df.columns:
        return
    for seed, grp in df[df["max_epoch"] > 0].groupby("seed"):
        grp = grp.sort_values("max_epoch")
        x = grp["max_epoch"].values
        y = pd.to_numeric(grp[col], errors="coerce").values
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.3)


# =============================================================================
# FIGURE: acc.png
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

acc_col = f"eval/{EVAL_SPLIT}/acc"
_plot_seed_lines(ax, df, acc_col, "darkorange")
_plot_line(ax, summary_df, acc_col, "clean acc", "darkorange")

ax.set_xlabel("max_epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "acc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved acc.png")

# =============================================================================
# FIGURE: rob.png  — rob w/o purification and w/ purification
# at 0.1 × range_size
# =============================================================================

eps_plot    = STRENGTH_FRACTIONS[0] * range_size   # 0.1 × range_size
radius_plot = PURIF_RADIUS_FRAC * range_size        # 0.1 × range_size

# rob w/o purification
rob_col_candidates = [
    f"eval/uq_adv_acc/{eps_plot}",
    f"eval/{EVAL_SPLIT}/rob/{eps_plot}",
]
rob_col = next((c for c in rob_col_candidates if f"{c}/mean" in summary_df.columns), None)

# rob w/ purification
purif_col_candidates = [
    f"eval/uq_purify_acc/{eps_plot}/{radius_plot}",
]
if not any(f"{c}/mean" in summary_df.columns for c in purif_col_candidates):
    found = _find_col(
        summary_df,
        rf"^eval/uq_purify_acc/{re.escape(str(eps_plot))}/[^/]+/mean$",
    )
    if found:
        purif_col_candidates.insert(0, _strip_mean(found))
purif_col = next(
    (c for c in purif_col_candidates if f"{c}/mean" in summary_df.columns), None
)

fig, ax = plt.subplots(figsize=(8, 5))
plotted = False
if rob_col:
    _plot_seed_lines(ax, df, rob_col, "steelblue")
    _plot_line(ax, summary_df, rob_col, f"rob w/o purif (eps={eps_plot:.3g})", "steelblue")
    plotted = True
if purif_col:
    _plot_seed_lines(ax, df, purif_col, "seagreen")
    _plot_line(ax, summary_df, purif_col, f"rob w/ purif (eps={eps_plot:.3g}, r={radius_plot:.3g})", "seagreen")
    plotted = True
if not plotted:
    ax.text(0.5, 0.5, "No rob columns found", transform=ax.transAxes, ha="center")

ax.set_xlabel("max_epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "rob.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved rob.png")

# =============================================================================
# FIGURE: purif.png  — purification accuracy and recovery rate
# =============================================================================

purif_acc_col = purif_col
purif_rec_pattern = rf"^eval/uq_purify_recovery/{re.escape(str(eps_plot))}"
purif_rec_found = _find_col(summary_df, purif_rec_pattern + r".*/mean$")
purif_rec_col = _strip_mean(purif_rec_found) if purif_rec_found else None

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
if purif_acc_col:
    _plot_seed_lines(ax, df, purif_acc_col, "seagreen")
    _plot_line(ax, summary_df, purif_acc_col, f"purif acc (eps={eps_plot:.3g})", "seagreen")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Accuracy after purification")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
if purif_rec_col:
    _plot_seed_lines(ax, df, purif_rec_col, "seagreen")
    _plot_line(ax, summary_df, purif_rec_col, f"recovery rate (eps={eps_plot:.3g})", "seagreen")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Recovery rate")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "purif.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved purif.png")

# =============================================================================
# FIGURE: loss.png  — cls_loss and gen_loss
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

clsloss_col = f"eval/{EVAL_SPLIT}/clsloss"
_plot_seed_lines(axes[0], df, clsloss_col, "darkorange")
_plot_line(axes[0], summary_df, clsloss_col, "cls loss", "darkorange")
axes[0].set_xlabel("max_epoch")
axes[0].set_ylabel("cls_loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

genloss_col = f"eval/{EVAL_SPLIT}/genloss"
_plot_seed_lines(axes[1], df, genloss_col, "seagreen")
_plot_line(axes[1], summary_df, genloss_col, "gen loss", "seagreen")
axes[1].set_xlabel("max_epoch")
axes[1].set_ylabel("gen_loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved loss.png")

# =============================================================================
# FIGURE: diff.png  — posttraining − pretrained diffs
# =============================================================================

# Summary for diff columns (max_epoch > 0 only, since diff is undefined at 0)
diff_metric_cols = [c for c in df.columns if c.startswith("diff/")]
diff_max_epochs  = sorted(df[df["max_epoch"] > 0]["max_epoch"].dropna().unique())
_x_diff = np.array(diff_max_epochs, dtype=float)

diff_agg_rows = []
for ep in diff_max_epochs:
    group = df[df["max_epoch"] == ep]
    agg   = {"max_epoch": int(ep)}
    for col in diff_metric_cols:
        vals = pd.to_numeric(group[col], errors="coerce").dropna()
        agg[f"{col}/mean"] = vals.mean()  if len(vals) > 0 else np.nan
        agg[f"{col}/std"]  = vals.std()   if len(vals) > 1 else np.nan
    diff_agg_rows.append(agg)

diff_summary_df = pd.DataFrame(diff_agg_rows) if diff_agg_rows else pd.DataFrame()


def _plot_diff_summary_line(ax, col, label, color):
    """Plot diff mean ± std using diff_summary_df and _x_diff."""
    if diff_summary_df.empty:
        return False
    mean_col = f"{col}/mean"
    std_col  = f"{col}/std"
    if mean_col not in diff_summary_df.columns:
        return False
    means = diff_summary_df[mean_col].values
    stds  = diff_summary_df[std_col].fillna(0).values
    ax.plot(_x_diff, means, "o-", color=color, linewidth=2, markersize=5, label=label)
    ax.fill_between(_x_diff, means - stds, means + stds, alpha=0.2, color=color)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    return True


def _plot_diff_seed_lines(ax, col, color):
    """Overlay thin per-seed diff lines."""
    if col not in df.columns:
        return
    for seed, grp in df[df["max_epoch"] > 0].groupby("seed"):
        grp = grp.sort_values("max_epoch")
        x = grp["max_epoch"].values
        y = pd.to_numeric(grp[col], errors="coerce").values
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.3)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# [0,0] Δ acc
ax = axes[0, 0]
_plot_diff_seed_lines(ax, "diff/test/acc", "darkorange")
_plot_diff_summary_line(ax, "diff/test/acc", "Δ acc", "darkorange")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Δ (posttraining − pretrained)")
ax.set_title("Δ clean accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# [0,1] Δ cls_loss
ax = axes[0, 1]
_plot_diff_seed_lines(ax, "diff/test/clsloss", "steelblue")
_plot_diff_summary_line(ax, "diff/test/clsloss", "Δ cls_loss", "steelblue")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Δ (posttraining − pretrained)")
ax.set_title("Δ cls_loss")
ax.legend()
ax.grid(True, alpha=0.3)

# [0,2] Δ gen_loss
ax = axes[0, 2]
_plot_diff_seed_lines(ax, "diff/test/genloss", "seagreen")
_plot_diff_summary_line(ax, "diff/test/genloss", "Δ gen_loss", "seagreen")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Δ (posttraining − pretrained)")
ax.set_title("Δ gen_loss")
ax.legend()
ax.grid(True, alpha=0.3)

# [1,0] Δ rob at all strengths
ax = axes[1, 0]
strengths_plot = [f * range_size for f in STRENGTH_FRACTIONS]
_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(strengths_plot)))
any_rob_diff = False
for eps_val, col_color in zip(strengths_plot, _colors):
    diff_rob_col = f"diff/{EVAL_SPLIT}/rob/{eps_val}"
    _plot_diff_seed_lines(ax, diff_rob_col, col_color)
    ok = _plot_diff_summary_line(ax, diff_rob_col, f"eps={eps_val:.3g}", col_color)
    any_rob_diff = any_rob_diff or ok
if not any_rob_diff:
    ax.text(0.5, 0.5, "No rob diff columns found", transform=ax.transAxes, ha="center")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Δ (posttraining − pretrained)")
ax.set_title("Δ rob (all strengths)")
ax.legend()
ax.grid(True, alpha=0.3)

# [1,1] Δ rob w/ purif at 0.1×range_size
ax = axes[1, 1]
eps_p  = STRENGTH_FRACTIONS[0] * range_size
rad_p  = PURIF_RADIUS_FRAC * range_size
diff_purif_col = f"diff/uq_purify_acc/{eps_p}/{rad_p}"
if not diff_summary_df.empty and f"{diff_purif_col}/mean" not in diff_summary_df.columns:
    found_purif = _find_col(
        diff_summary_df,
        rf"^diff/uq_purify_acc/{re.escape(str(eps_p))}/[^/]+/mean$",
    )
    if found_purif:
        diff_purif_col = _strip_mean(found_purif)
_plot_diff_seed_lines(ax, diff_purif_col, "seagreen")
_plot_diff_summary_line(ax, diff_purif_col, f"Δ purif acc (eps={eps_p:.3g}, r={rad_p:.3g})", "seagreen")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Δ (posttraining − pretrained)")
ax.set_title("Δ rob w/ purification")
ax.legend()
ax.grid(True, alpha=0.3)

# [1,2] Δ clean acc (UQ)
ax = axes[1, 2]
_plot_diff_seed_lines(ax, "diff/uq_clean_accuracy", "darkorange")
_plot_diff_summary_line(ax, "diff/uq_clean_accuracy", "Δ uq clean acc", "darkorange")
ax.set_xlabel("max_epoch")
ax.set_ylabel("Δ (posttraining − pretrained)")
ax.set_title("Δ clean accuracy (UQ)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "diff.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved diff.png")

# =============================================================================
# EVOLUTION FIGURE (cls_reg_evolution.py)
# =============================================================================

try:
    from analysis.visualize.cls_reg_evolution import build_evolution_figure
    fig = build_evolution_figure(
        sweep_dir=str(sweep_dir),
        device=device,
        save_path=str(output_dir / "evolution.png"),
    )
    plt.close(fig)
    print("Saved evolution.png")
except Exception as e:
    logger.warning(f"Evolution figure failed: {e}")

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("cls_reg Analysis Complete!")
print("=" * 60)
print(f"  sweep_dir: {sweep_dir}")
print(f"  regime={regime}, embedding={embedding}, arch={arch}, dataset={dataset}, date={date}")
print(f"  {len(df)} rows, {len(max_epochs)} max_epoch values")
print(f"\nOutputs saved to: {output_dir}")
for fname in ["evaluation_data.csv", "summary.csv", "acc.png", "rob.png", "purif.png", "loss.png", "diff.png", "evolution.png"]:
    p = output_dir / fname
    status = "OK" if p.exists() else "MISSING"
    print(f"  [{status}] {fname}")
print("=" * 60)
