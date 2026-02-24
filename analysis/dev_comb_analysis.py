"""
dev_comb Analysis Tool — W&B-only approach.

Fetches run summaries + training histories from W&B for a dev_comb seed_sweep,
groups by `cls_epochs` (trainer.classification.max_epoch), and produces:

  dev_comb_data.csv      — per-run summary metrics
  dev_comb_summary.csv   — aggregated mean ± std per cls_epochs
  dev_comb_tradeoff.png  — final FID / acc / genloss vs cls_epochs
  dev_comb_dynamics.png  — per-epoch FID evolution (pre + gen phases stitched)

Usage:
    python analysis/dev_comb_analysis.py <wandb_group_name>

Example:
    python analysis/dev_comb_analysis.py dev_comb_clsgen_d6D4fourier_circles_4k_2302
"""

import sys
import argparse
import re
from pathlib import Path

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
import matplotlib.cm as cm

# =============================================================================
# CONSTANTS
# =============================================================================

WANDB_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
WANDB_PROJECT = "gan_train"

# W&B summary keys extracted per run
SUMMARY_KEYS = [
    "gen/valid/fid",
    "gen/valid/acc",
    "gen/valid/genloss",
    "pre/valid/fid",
    "pre/valid/acc",
    "pre/valid/clsloss",
]

# =============================================================================
# CLI
# =============================================================================

_cli = argparse.ArgumentParser(
    description="dev_comb analysis: tradeoff curve + FID dynamics from W&B.",
    add_help=False,
)
_cli.add_argument("group_name", help="W&B group name, e.g. dev_comb_clsgen_d6D4fourier_circles_4k_2302")
_cli_args, _ = _cli.parse_known_args()
GROUP_NAME = _cli_args.group_name

# =============================================================================
# GROUP NAME PARSING  →  output directory
# =============================================================================

def parse_group_name(group_name: str):
    """Parse dev_comb group name into (arch, embedding, dataset, date).

    Expected format:
        dev_comb_{regime}_{arch}{embedding}_{dataset}_{date}
    where date is DDMM (4 digits) or DDMM_HHMM (8 digits with underscore).

    Examples:
        dev_comb_clsgen_d6D4fourier_circles_4k_2302
          → arch='d6D4', embedding='fourier', dataset='circles_4k', date='2302'
        dev_comb_clsgen_d10D6legendre_moons_4k_2302_1430
          → arch='d10D6', embedding='legendre', dataset='moons_4k', date='2302_1430'
    """
    prefix = "dev_comb_"
    if not group_name.startswith(prefix):
        raise ValueError(
            f"Group name must start with '{prefix}', got: {group_name}\n"
            "Expected format: dev_comb_{{regime}}_{{arch}}{{embedding}}_{{dataset}}_{{date}}"
        )
    rest = group_name[len(prefix):]

    # Strip the regime token (e.g. "clsgen_") that precedes the arch.
    # Regime names are purely alphabetical; arch always starts with d<digit>.
    rest = re.sub(r'^[a-zA-Z]+_', '', rest)

    # Match arch (d<n>D<m>) + embedding at the start
    m = re.match(
        r'^(d\d+D\d+)(fourier|legendre|hermite|chebychev1|chebychev2)(_.+)$',
        rest, re.IGNORECASE,
    )
    if not m:
        raise ValueError(
            f"Could not parse arch+embedding from: '{rest}'\n"
            "Expected something like 'd6D4fourier_circles_4k_2302'."
        )

    arch      = m.group(1)
    embedding = m.group(2).lower()
    tail      = m.group(3)[1:]   # strip leading '_', e.g. "circles_4k_2302"

    # Tail = {dataset}_{date[_time]}
    # Date is always pure digits; time suffix is also pure digits.
    # Strategy: strip trailing digit-only tokens (up to 2) to get the date.
    parts = tail.split("_")
    date_parts = []
    for token in reversed(parts):
        if re.match(r'^\d{4}$', token):
            date_parts.insert(0, token)
        else:
            break

    if not date_parts:
        raise ValueError(f"Could not extract date from tail: '{tail}'")

    n_date = len(date_parts)
    dataset_parts = parts[:len(parts) - n_date]
    dataset = "_".join(dataset_parts)
    date    = "_".join(date_parts)

    return arch, embedding, dataset, date


arch, embedding, dataset, date = parse_group_name(GROUP_NAME)
print(f"Parsed group name:")
print(f"  arch={arch}, embedding={embedding}, dataset={dataset}, date={date}")

output_dir = (
    project_root / "analysis" / "outputs"
    / "seed_sweep" / "comb" / embedding / arch / f"{dataset}_{date}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# =============================================================================
# FETCH RUNS FROM W&B
# =============================================================================

from analysis.utils.wandb_fetcher import WandbFetcher, WANDB_AVAILABLE

if not WANDB_AVAILABLE:
    raise ImportError("wandb is required. Install with: pip install wandb")

fetcher = WandbFetcher(entity=WANDB_ENTITY, project=WANDB_PROJECT)

print(f"\nFetching runs from group: {GROUP_NAME}")
runs = fetcher.fetch_runs(
    filters={"group": GROUP_NAME, "state": "finished"},
    order="created_at",
)
print(f"Found {len(runs)} finished runs")

if not runs:
    print("No finished runs found. Exiting.")
    sys.exit(0)

# =============================================================================
# BUILD SUMMARY DATAFRAME
# =============================================================================

records = []
for run in runs:
    config  = run.config
    summary = run.summary._json_dict

    cls_epochs = fetcher._get_nested(config, "trainer.classification.max_epoch")
    seed       = fetcher._get_nested(config, "tracking.seed")

    record = {
        "run_id":     run.id,
        "run_name":   run.name,
        "cls_epochs": cls_epochs,
        "seed":       seed,
    }
    for key in SUMMARY_KEYS:
        val = summary.get(key)
        try:
            val = float(val)
            if not np.isfinite(val):
                val = float("nan")
        except (TypeError, ValueError):
            val = float("nan")
        record[key] = val

    records.append(record)

df = pd.DataFrame(records)
print(f"\nRaw data: {len(df)} runs")

cls_epochs_values = sorted(df["cls_epochs"].dropna().unique())
seeds             = sorted(df["seed"].dropna().unique())
print(f"cls_epochs values: {cls_epochs_values}")
print(f"seeds:             {seeds}")

df.to_csv(output_dir / "dev_comb_data.csv", index=False)
print(f"Saved dev_comb_data.csv")

# =============================================================================
# AGGREGATE BY CLS_EPOCHS  →  summary CSV
# =============================================================================

agg_records = []
for cls_ep in cls_epochs_values:
    group = df[df["cls_epochs"] == cls_ep]
    agg = {"cls_epochs": int(cls_ep), "n_runs": len(group)}
    for key in SUMMARY_KEYS:
        vals = pd.to_numeric(group[key], errors="coerce")
        vals = vals[np.isfinite(vals)]
        agg[f"{key}/mean"] = vals.mean() if len(vals) > 0 else float("nan")
        agg[f"{key}/std"]  = vals.std()  if len(vals) > 1 else float("nan")
    agg_records.append(agg)

summary_df = pd.DataFrame(agg_records)
summary_df.to_csv(output_dir / "dev_comb_summary.csv", index=False)
print(f"Saved dev_comb_summary.csv")
print(f"\nAggregated summary:\n{summary_df.to_string(index=False)}")

# =============================================================================
# TRADEOFF PLOT
# =============================================================================

_x = np.array(cls_epochs_values, dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"dev_comb Tradeoff — {GROUP_NAME}", fontsize=11)

_tradeoff_metrics = [
    ("gen/valid/fid",     "Final FID (gen phase)",       "steelblue"),
    ("gen/valid/acc",     "Final Accuracy (gen phase)",   "darkorange"),
    ("gen/valid/genloss", "Final Gen Loss (gen phase)",   "seagreen"),
]

for ax, (key, label, color) in zip(axes, _tradeoff_metrics):
    mean_col = f"{key}/mean"
    std_col  = f"{key}/std"

    if mean_col not in summary_df.columns:
        ax.set_visible(False)
        continue

    means = summary_df[mean_col].values
    stds  = summary_df[std_col].fillna(0).values

    ax.plot(_x, means, "o-", color=color, linewidth=2, markersize=6)
    ax.fill_between(_x, means - stds, means + stds, alpha=0.25, color=color)
    ax.set_xlabel("cls_epochs")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(_x)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "dev_comb_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved dev_comb_tradeoff.png")

# =============================================================================
# FETCH RUN HISTORIES FOR DYNAMICS
# =============================================================================

print("\nFetching run histories for dynamics plot...")

# Map cls_epochs → list of (epochs_arr, fids_arr) — one entry per seed
dynamics_data = {cls_ep: [] for cls_ep in cls_epochs_values}

for i, run in enumerate(runs):
    config     = run.config
    cls_epochs = fetcher._get_nested(config, "trainer.classification.max_epoch")

    if cls_epochs not in dynamics_data:
        continue

    run_id_str = f"[{i + 1}/{len(runs)}] {run.name} (cls_epochs={int(cls_epochs)})"
    print(f"  Fetching history: {run_id_str} ...")

    # Fetch pre and gen FID histories separately.
    # Requesting both keys in one call can yield empty results from the W&B
    # sampledHistory endpoint when the two keys never appear in the same step
    # (pre/valid/fid only during pre-phase, gen/valid/fid only during gen-phase).
    try:
        pre_hist = fetcher.fetch_run_history(run, keys=["pre/valid/fid"])
        gen_hist = fetcher.fetch_run_history(run, keys=["gen/valid/fid"])
    except Exception as e:
        print(f"    Warning: could not fetch history for {run.name}: {e}")
        continue

    def _fid_rows(hist, key):
        if hist is None or hist.empty or key not in hist.columns:
            return pd.DataFrame()
        rows = hist[hist[key].notna()].copy()
        return rows

    # Separate pre and gen phase rows
    pre_rows = _fid_rows(pre_hist, "pre/valid/fid")
    gen_rows = _fid_rows(gen_hist, "gen/valid/fid")

    n_pre = len(pre_rows)
    n_gen = len(gen_rows)

    # Assign global epochs by evenly spacing within each phase's range:
    #   pre-phase: epochs (0 .. cls_epochs]
    #   gen-phase: epochs (cls_epochs .. 100]
    # This is robust regardless of how W&B stores the internal step counter.
    pre_epochs = np.linspace(0, cls_epochs, n_pre + 1)[1:] if n_pre > 0 else np.array([])
    gen_epochs = np.linspace(cls_epochs, 100, n_gen + 1)[1:] if n_gen > 0 else np.array([])

    pre_fids = pre_rows["pre/valid/fid"].values if n_pre > 0 else np.array([])
    gen_fids = gen_rows["gen/valid/fid"].values if n_gen > 0 else np.array([])

    all_epochs = np.concatenate([pre_epochs, gen_epochs])
    all_fids   = np.concatenate([pre_fids,   gen_fids])

    if len(all_epochs) == 0:
        print(f"    Warning: no FID values found for {run.name}")
        continue

    # Sort by epoch
    sort_idx   = np.argsort(all_epochs)
    all_epochs = all_epochs[sort_idx]
    all_fids   = all_fids[sort_idx]

    dynamics_data[cls_epochs].append((all_epochs, all_fids))
    print(f"    → {n_pre} pre + {n_gen} gen FID points")

# =============================================================================
# DYNAMICS PLOT
# =============================================================================

n_cls_vals = len(cls_epochs_values)
colors     = cm.viridis(np.linspace(0.05, 0.95, n_cls_vals))

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle(f"dev_comb FID Dynamics — {GROUP_NAME}", fontsize=11)

for color, cls_ep in zip(colors, cls_epochs_values):
    seed_runs = dynamics_data[cls_ep]
    if not seed_runs:
        continue

    # Build a common epoch grid from the union of all observed epochs
    all_ep_union = sorted({ep for epochs_arr, _ in seed_runs for ep in epochs_arr.tolist()})
    if not all_ep_union:
        continue
    epoch_grid = np.array(all_ep_union)

    # Interpolate each seed run to the common grid
    fid_matrix = []
    for epochs_arr, fids_arr in seed_runs:
        if len(epochs_arr) < 2:
            continue
        fid_interp = np.interp(epoch_grid, epochs_arr, fids_arr, left=np.nan, right=np.nan)
        fid_matrix.append(fid_interp)

    if not fid_matrix:
        continue

    fid_mat = np.array(fid_matrix)
    means   = np.nanmean(fid_mat, axis=0)
    stds    = np.nanstd(fid_mat,  axis=0)

    label = f"cls={int(cls_ep)}"
    ax.plot(epoch_grid, means, color=color, linewidth=1.5, label=label)
    ax.fill_between(epoch_grid, means - stds, means + stds, alpha=0.12, color=color)

    # Vertical dashed line at phase transition
    if 0 < cls_ep < 100:
        ax.axvline(x=cls_ep, color=color, linestyle="--", linewidth=0.7, alpha=0.5)

ax.set_xlabel("Global Epoch")
ax.set_ylabel("FID")
ax.set_title("FID Dynamics (pre-phase + gen-phase stitched, mean ± std across seeds)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / "dev_comb_dynamics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved dev_comb_dynamics.png")

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("dev_comb Analysis Complete!")
print("=" * 60)
print(f"\nGroup:    {GROUP_NAME}")
print(f"Runs:     {len(df)}")
print(f"arch:     {arch}  |  embedding: {embedding}")
print(f"dataset:  {dataset}  |  date: {date}")
print(f"\nOutputs saved to: {output_dir}")
for fname in ["dev_comb_data.csv", "dev_comb_summary.csv",
              "dev_comb_tradeoff.png", "dev_comb_dynamics.png"]:
    p = output_dir / fname
    status = "OK" if p.exists() else "MISSING"
    print(f"  [{status}] {fname}")
print("=" * 60)
