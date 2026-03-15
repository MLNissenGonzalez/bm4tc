"""
dev_comb Analysis Tool — local model evaluation.

Loads saved model checkpoints (models/cls, models/gen) from each run in a
dev_comb sweep directory, evaluates clean accuracy, adversarial robustness
(with and without purification), cls_loss, and gen_loss, then produces
summary CSVs and plots grouped by cls_epoch.

Usage:
    python analysis/dev_comb_analysis.py <sweep_dir> [--device cuda] [--force]

Example:
    python analysis/dev_comb_analysis.py \\
        outputs/dev_comb/clsgen/legendre/d10D6/circles_4k_1303 \\
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

from analysis.utils.evaluate import EvalConfig, evaluate_model_at_path
from analysis.utils.mia_utils import load_run_config
from analysis.utils.resolve import resolve_embedding_from_path, embedding_range_size

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
    description="dev_comb analysis: local model evaluation across cls_epoch sweep.",
)
_cli.add_argument("sweep_dir", help="Path to sweep output directory")
_cli.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
_cli.add_argument("--force", action="store_true", help="Re-evaluate even if CSV exists")
args = _cli.parse_args()

sweep_dir = Path(args.sweep_dir).resolve()
device    = args.device
force     = args.force

# =============================================================================
# PATH PARSING  →  output directory
# =============================================================================

def _parse_sweep_dir(sweep_dir: Path):
    """Parse sweep dir path into (embedding, arch, dataset, date).

    Expected structure:
        .../{experiment}/{regime}/{embedding}/{arch}/{dataset}_{date}

    e.g. outputs/dev_comb/clsgen/legendre/d10D6/circles_4k_1303
      -> embedding='legendre', arch='d10D6', dataset='circles_4k', date='1303'
    """
    parts = sweep_dir.parts
    last  = parts[-1]   # e.g. "circles_4k_1303"
    arch  = parts[-2]   # e.g. "d10D6"

    # Detect embedding from path
    embedding = resolve_embedding_from_path(str(sweep_dir))
    if embedding is None:
        # Fallback: second-to-last directory before arch
        embedding = parts[-3] if len(parts) >= 3 else "unknown"

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

    return embedding, arch, dataset, date


embedding, arch, dataset, date = _parse_sweep_dir(sweep_dir)
print(f"Parsed sweep dir:")
print(f"  embedding={embedding}, arch={arch}, dataset={dataset}, date={date}")

range_size = embedding_range_size(embedding)
print(f"  range_size={range_size}")

output_dir = (
    project_root / "analysis" / "outputs"
    / "seed_sweep" / "comb" / embedding / arch / f"{dataset}_{date}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# =============================================================================
# DISCOVER RUNS
# =============================================================================

def discover_runs(sweep_dir: Path) -> List[Path]:
    """Return sorted list of run dirs that have both models/cls and models/gen."""
    runs = []
    for d in sorted(sweep_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
        if not d.is_dir():
            continue
        if not re.match(r"^\d+$", d.name):
            continue
        if (d / "models" / "cls").exists() and (d / "models" / "gen").exists():
            runs.append(d)
    return runs

# =============================================================================
# EVAL CONFIG
# =============================================================================

def build_eval_cfg(range_size: float, device: str) -> EvalConfig:
    """Build a single EvalConfig used for both cls and gen model evaluation."""
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
    run_dirs = discover_runs(sweep_dir)
    if not run_dirs:
        print(f"No valid runs found in {sweep_dir}. Exiting.")
        sys.exit(1)
    print(f"\nFound {len(run_dirs)} runs with both models/cls and models/gen")

    eval_cfg = build_eval_cfg(range_size, device)
    rows: List[Dict[str, Any]] = []

    for i, run_dir in enumerate(run_dirs):
        print(f"\n[{i + 1}/{len(run_dirs)}] {run_dir.name}")

        try:
            cfg       = load_run_config(run_dir)
            cls_epoch = int(cfg.trainer.classification.max_epoch)
            seed      = int(cfg.tracking.seed)
        except Exception as e:
            logger.warning(f"  Skipping {run_dir.name}: could not load config: {e}")
            continue

        print(f"  cls_epoch={cls_epoch}, seed={seed}")

        row: Dict[str, Any] = {
            "cls_epoch": cls_epoch,
            "seed":      seed,
            "run_path":  str(run_dir),
            "range_size": range_size,
        }

        # Evaluate cls model (sync_gen=True: sync tensors before gen_loss)
        print("  Evaluating models/cls ...")
        try:
            cls_metrics = evaluate_model_at_path(
                run_dir / "models" / "cls",
                run_dir,
                eval_cfg,
                sync_gen=True,
            )
            row.update({f"cls/{k}": v for k, v in cls_metrics.items()})
        except Exception as e:
            logger.warning(f"  models/cls failed: {e}")

        # Evaluate gen model (sync_gen=False: gen tensors already trained)
        print("  Evaluating models/gen ...")
        try:
            gen_metrics = evaluate_model_at_path(
                run_dir / "models" / "gen",
                run_dir,
                eval_cfg,
                sync_gen=False,
            )
            row.update({f"gen/{k}": v for k, v in gen_metrics.items()})
        except Exception as e:
            logger.warning(f"  models/gen failed: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(eval_csv, index=False)
    print(f"\nSaved evaluation_data.csv ({len(df)} rows)")

# After df is built/loaded, add/refresh diff columns (gen - cls)
cls_metric_keys = {c[len("cls/"):] for c in df.columns if c.startswith("cls/")}
gen_metric_keys  = {c[len("gen/"):] for c in df.columns if c.startswith("gen/")}
for k in cls_metric_keys & gen_metric_keys:
    df[f"diff/{k}"] = (
        pd.to_numeric(df[f"gen/{k}"], errors="coerce")
        - pd.to_numeric(df[f"cls/{k}"], errors="coerce")
    )

# =============================================================================
# SUMMARY CSV  (mean ± std per cls_epoch)
# =============================================================================

metric_cols = [c for c in df.columns if c not in ("cls_epoch", "seed", "run_path", "range_size")]
cls_epochs  = sorted(df["cls_epoch"].dropna().unique())

agg_rows = []
for ep in cls_epochs:
    group = df[df["cls_epoch"] == ep]
    agg   = {"cls_epoch": int(ep), "n_seeds": len(group)}
    for col in metric_cols:
        vals = pd.to_numeric(group[col], errors="coerce").dropna()
        agg[f"{col}/mean"] = vals.mean()  if len(vals) > 0 else np.nan
        agg[f"{col}/std"]  = vals.std()   if len(vals) > 1 else np.nan
    agg_rows.append(agg)

summary_df = pd.DataFrame(agg_rows)
summary_csv = output_dir / "summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary.csv ({len(summary_df)} rows, {len(cls_epochs)} cls_epoch values)")

# =============================================================================
# PLOTTING HELPERS
# =============================================================================

_x = np.array(cls_epochs, dtype=float)


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


# =============================================================================
# FIGURE: acc.png
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

acc_gen = f"gen/eval/{EVAL_SPLIT}/acc"
_plot_line(ax, summary_df, acc_gen, "clean acc", "darkorange")

ax.set_xlabel("cls_epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "acc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved acc.png")

# =============================================================================
# FIGURE: rob.png  — rob w/o purification and w/ purification (gen model)
# at 0.1 × range_size
# =============================================================================

eps_plot    = STRENGTH_FRACTIONS[0] * range_size   # 0.1 × range_size
radius_plot = PURIF_RADIUS_FRAC * range_size        # 0.1 × range_size

# rob w/o purification: uq_adv_acc at eps_plot (from UQ) or fallback test/rob
rob_col_candidates = [
    f"gen/eval/uq_adv_acc/{eps_plot}",
    f"gen/eval/{EVAL_SPLIT}/rob/{eps_plot}",
]
rob_col = next((c for c in rob_col_candidates if f"{c}/mean" in summary_df.columns), None)

# rob w/ purification: uq_purify_acc at (eps_plot, radius_plot)
purif_col_candidates = [
    f"gen/eval/uq_purify_acc/{eps_plot}/{radius_plot}",
]
# Also try scanning columns for a matching purify_acc pattern
if not any(f"{c}/mean" in summary_df.columns for c in purif_col_candidates):
    pattern = rf"gen/eval/uq_purify_acc/[^/]+/[^/]+"
    found = _find_col(
        summary_df,
        rf"^gen/eval/uq_purify_acc/{re.escape(str(eps_plot))}/.*\/mean$",
    )
    if found:
        purif_col_candidates.insert(0, _strip_mean(found))
purif_col = next(
    (c for c in purif_col_candidates if f"{c}/mean" in summary_df.columns), None
)

fig, ax = plt.subplots(figsize=(8, 5))
plotted = False
if rob_col:
    _plot_line(ax, summary_df, rob_col, f"rob w/o purif (eps={eps_plot:.3g})", "steelblue")
    plotted = True
if purif_col:
    _plot_line(ax, summary_df, purif_col, f"rob w/ purif (eps={eps_plot:.3g}, r={radius_plot:.3g})", "seagreen")
    plotted = True
if not plotted:
    ax.text(0.5, 0.5, "No rob columns found", transform=ax.transAxes, ha="center")

ax.set_xlabel("cls_epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "rob.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved rob.png")

# =============================================================================
# FIGURE: purif.png  — purification accuracy and recovery rate (gen model)
# =============================================================================

purif_acc_col  = purif_col  # same as used in rob.png
purif_rec_pattern = rf"^gen/eval/uq_purify_recovery/{re.escape(str(eps_plot))}"
purif_rec_found = _find_col(summary_df, purif_rec_pattern + r".*/mean$")
purif_rec_col = _strip_mean(purif_rec_found) if purif_rec_found else None

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
if purif_acc_col:
    _plot_line(ax, summary_df, purif_acc_col, f"purif acc (eps={eps_plot:.3g})", "seagreen")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Accuracy after purification")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
if purif_rec_col:
    _plot_line(ax, summary_df, purif_rec_col, f"recovery rate (eps={eps_plot:.3g})", "seagreen")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Recovery rate")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "purif.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved purif.png")

# =============================================================================
# FIGURE: loss.png  — cls_loss (gen), gen_loss (gen)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

_plot_line(axes[0], summary_df,
           f"gen/eval/{EVAL_SPLIT}/clsloss", "cls loss", "darkorange")
axes[0].set_xlabel("cls_epoch")
axes[0].set_ylabel("cls_loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

_plot_line(axes[1], summary_df,
           f"gen/eval/{EVAL_SPLIT}/genloss", "gen loss", "seagreen")
axes[1].set_xlabel("cls_epoch")
axes[1].set_ylabel("gen_loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved loss.png")

# =============================================================================
# FIGURE: diff.png  — gen − cls for all shared metrics
# =============================================================================

def _plot_diff_line(ax, summary_df, diff_col, label, color):
    """Like _plot_line but adds a y=0 reference line."""
    ok = _plot_line(ax, summary_df, diff_col, label, color)
    if ok:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    return ok


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# [0,0] Δ acc
ax = axes[0, 0]
_plot_diff_line(ax, summary_df, f"diff/eval/{EVAL_SPLIT}/acc", "Δ acc", "darkorange")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Δ (gen − cls)")
ax.set_title("Δ clean accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# [0,1] Δ cls_loss
ax = axes[0, 1]
_plot_diff_line(ax, summary_df, f"diff/eval/{EVAL_SPLIT}/clsloss", "Δ cls_loss", "steelblue")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Δ (gen − cls)")
ax.set_title("Δ cls_loss")
ax.legend()
ax.grid(True, alpha=0.3)

# [0,2] Δ gen_loss
ax = axes[0, 2]
_plot_diff_line(ax, summary_df, f"diff/eval/{EVAL_SPLIT}/genloss", "Δ gen_loss", "seagreen")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Δ (gen − cls)")
ax.set_title("Δ gen_loss")
ax.legend()
ax.grid(True, alpha=0.3)

# [1,0] Δ rob at all strengths
ax = axes[1, 0]
strengths_plot = [f * range_size for f in STRENGTH_FRACTIONS]
_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(strengths_plot)))
any_rob_diff = False
for eps_val, col_color in zip(strengths_plot, _colors):
    diff_rob_col = f"diff/eval/{EVAL_SPLIT}/rob/{eps_val}"
    ok = _plot_diff_line(ax, summary_df, diff_rob_col, f"eps={eps_val:.3g}", col_color)
    any_rob_diff = any_rob_diff or ok
if not any_rob_diff:
    ax.text(0.5, 0.5, "No rob diff columns found", transform=ax.transAxes, ha="center")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Δ (gen − cls)")
ax.set_title("Δ rob (all strengths)")
ax.legend()
ax.grid(True, alpha=0.3)

# [1,1] Δ rob w/ purif at 0.1×range_size
ax = axes[1, 1]
eps_p   = STRENGTH_FRACTIONS[0] * range_size
rad_p   = PURIF_RADIUS_FRAC * range_size
diff_purif_col = f"diff/eval/uq_purify_acc/{eps_p}/{rad_p}"
# Scan for the column if the exact key misses
if f"{diff_purif_col}/mean" not in summary_df.columns:
    found_purif = _find_col(
        summary_df,
        rf"^diff/eval/uq_purify_acc/{re.escape(str(eps_p))}/[^/]+/mean$",
    )
    if found_purif:
        diff_purif_col = _strip_mean(found_purif)
_plot_diff_line(ax, summary_df, diff_purif_col, f"Δ purif acc (eps={eps_p:.3g}, r={rad_p:.3g})", "seagreen")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Δ (gen − cls)")
ax.set_title("Δ rob w/ purification")
ax.legend()
ax.grid(True, alpha=0.3)

# [1,2] Δ clean acc (UQ)
ax = axes[1, 2]
_plot_diff_line(ax, summary_df, "diff/eval/uq_clean_accuracy", "Δ uq clean acc", "darkorange")
ax.set_xlabel("cls_epoch")
ax.set_ylabel("Δ (gen − cls)")
ax.set_title("Δ clean accuracy (UQ)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "diff.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved diff.png")

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("dev_comb Analysis Complete!")
print("=" * 60)
print(f"  sweep_dir: {sweep_dir}")
print(f"  embedding={embedding}, arch={arch}, dataset={dataset}, date={date}")
print(f"  {len(df)} runs, {len(cls_epochs)} cls_epoch values")
print(f"\nOutputs saved to: {output_dir}")
for fname in ["evaluation_data.csv", "summary.csv", "acc.png", "rob.png", "purif.png", "loss.png", "diff.png"]:
    p = output_dir / fname
    status = "OK" if p.exists() else "MISSING"
    print(f"  [{status}] {fname}")
print("=" * 60)
