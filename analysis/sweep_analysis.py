# %% [markdown]
# # Post-Hoc Sweep Analysis
#
# This notebook **loads each saved model** from a sweep directory and
# **recomputes metrics post-hoc**, enabling:
# - Computing metrics that weren't computed during training
# - Ensuring data splits match each run's config (correct seeds)
# - Consistent evaluation settings across all runs (e.g., same attack strengths)
#
# **Split policy:** Metrics are computed on both validation and test.
# Model selection (best run, Pareto frontiers) uses **validation only**
# to prevent label leakage. The summary reports both splits.
#
# **Sections:**
# 1. Configuration
# 2. Per-model evaluation
# 3. Statistics & visualization (histogram, bar, correlation, scatter, Pareto, summary)
# 4. Sanity check against W&B summary metrics
# 5. Learned distribution visualization for best model
# 6. Summary export

# %% [markdown]
# ## 1. Configuration

# %%
import sys
from pathlib import Path

# Handle both script and interactive execution
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
import torch

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# Path to sweep directory (contains numbered sub-dirs with .hydra/config.yaml)
SWEEP_DIR = "outputs/adv_seed_sweep_moons_4k_12Feb26"

# Training regime: "pre", "gen", "adv", "gan"
REGIME = "adv"

# --- METRIC TOGGLES ---
COMPUTE_ACC = True
COMPUTE_ROB = True
COMPUTE_MIA = True
COMPUTE_CLS_LOSS = True
COMPUTE_GEN_LOSS = False
COMPUTE_FID = False

# --- EVASION OVERRIDE ---
# Set to a dict to override evasion config across ALL runs for consistency.
# Set to None to use each run's own evasion config.
EVASION_OVERRIDE = None
# Example:
# EVASION_OVERRIDE = {
#     "method": "PGD",
#     "strengths": [0.05, 0.1, 0.2],
#     "num_steps": 20,
# }

# --- SAMPLING OVERRIDE ---
# Set to a dict to override sampling config, or None to use per-run config.
SAMPLING_OVERRIDE = None

# --- MIA SETTINGS ---
MIA_FEATURES = {
    "max_prob": True,
    "entropy": True,
    "correct_prob": True,
    "loss": True,
    "margin": True,
    "modified_entropy": True,
}

# --- EVALUATION SETTINGS ---
# Always evaluate on both validation and test.
# Validation is used for model selection; test is used for reporting.
EVAL_SPLITS = ["valid", "test"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- STATISTICS SETTINGS ---
EFFECTIVE_N = None  # Override for stderr calc (None = use actual N)

# --- PLOT SETTINGS ---
FIGSIZE = (10, 6)
DPI = 100

# --- SANITY CHECK ---
# Map eval column -> W&B summary column for comparison.
# Set to None or {} to skip sanity check.
SANITY_CHECK_METRICS = {
    "eval/test/acc": "summary/adv/test/acc",
    "eval/valid/clsloss": "summary/adv/valid/clsloss",
}
SANITY_CHECK_TOL = 1e-4

# --- CONFIG KEYS TO EXTRACT ---
# Hydra config keys to include in the DataFrame alongside eval metrics.
CONFIG_KEYS = [
    "dataset.name",
    "tracking.seed",
    "dataset.gen_dow_kwargs.seed",
]

# %% [markdown]
# ## 2. Per-Model Evaluation

# %%
from analysis.utils import EvalConfig, evaluate_sweep

eval_cfg = EvalConfig(
    compute_acc=COMPUTE_ACC,
    compute_rob=COMPUTE_ROB,
    compute_mia=COMPUTE_MIA,
    compute_cls_loss=COMPUTE_CLS_LOSS,
    compute_gen_loss=COMPUTE_GEN_LOSS,
    compute_fid=COMPUTE_FID,
    splits=EVAL_SPLITS,
    evasion_override=EVASION_OVERRIDE,
    sampling_override=SAMPLING_OVERRIDE,
    mia_features=MIA_FEATURES,
    device=DEVICE,
)

sweep_path = project_root / SWEEP_DIR
print("=" * 60)
print(f"Evaluating sweep: {sweep_path}")
print(f"Device: {DEVICE}")
print("=" * 60)

df = evaluate_sweep(str(sweep_path), eval_cfg, config_keys=CONFIG_KEYS)

# %%
# Show DataFrame summary
if not df.empty:
    eval_cols = [c for c in df.columns if c.startswith("eval/")]
    print(f"\nEval columns: {eval_cols}")
    print(f"\n{df[['run_name'] + eval_cols].to_string()}")

# %%
# Determine sweep name for output paths
sweep_name = Path(SWEEP_DIR).name.replace("/", "_")
output_dir = project_root / "analysis" / "outputs" / sweep_name
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## 3. Statistics & Visualization

# %%
# Resolve metric columns per split.
# Validation columns → model selection (best run, Pareto frontiers).
# Test columns → reporting.
VAL_ACC = "eval/valid/acc" if COMPUTE_ACC else None
VAL_ROB = [c for c in df.columns if c.startswith("eval/valid/rob/")] if COMPUTE_ROB and not df.empty else []
VAL_CLS_LOSS = "eval/valid/clsloss" if COMPUTE_CLS_LOSS else None

TEST_ACC = "eval/test/acc" if COMPUTE_ACC else None
TEST_ROB = [c for c in df.columns if c.startswith("eval/test/rob/")] if COMPUTE_ROB and not df.empty else []
TEST_CLS_LOSS = "eval/test/clsloss" if COMPUTE_CLS_LOSS else None

# MIA is split-agnostic (always uses train vs test internally)
MIA_COL = "eval/mia_accuracy" if COMPUTE_MIA else None

print(f"VAL_ACC:  {VAL_ACC}")
print(f"VAL_ROB:  {VAL_ROB}")
print(f"TEST_ACC: {TEST_ACC}")
print(f"TEST_ROB: {TEST_ROB}")
print(f"MIA_COL:  {MIA_COL}")

# %% [markdown]
# ### 3a. Best Model (selected on validation)

# %%
from analysis.utils import get_best_run, resolve_stop_criterion, load_run_config
from analysis.utils import REGIME_METRIC_PREFIX

STOP_CRIT_COL = None
STOP_CRIT_LABEL = None
STOP_CRIT_MINIMIZE = True
best_run = None

if not df.empty:
    # Resolve stopping criterion from first run's config (on validation split)
    try:
        first_run = Path(df.iloc[0]["run_path"])
        first_cfg = load_run_config(first_run)
        trainer_prefix_map = {
            "pre": "classification", "gen": "generative",
            "adv": "adversarial", "gan": "ganstyle",
        }
        trainer_key = trainer_prefix_map.get(REGIME)
        if trainer_key and hasattr(first_cfg.trainer, trainer_key):
            trainer_cfg = getattr(first_cfg.trainer, trainer_key)
            stop_name = getattr(trainer_cfg, "stop_crit", None)
            if stop_name:
                STOP_CRIT_COL, STOP_CRIT_MINIMIZE, STOP_CRIT_LABEL = resolve_stop_criterion(
                    stop_name, df, "valid",
                )
                print(f"Stop criterion: {stop_name} -> column {STOP_CRIT_COL}")
    except Exception as e:
        print(f"Could not resolve stop criterion: {e}")

    # Best run by stopping criterion (validation) or by validation accuracy
    best_metric = STOP_CRIT_COL if STOP_CRIT_COL else VAL_ACC
    best_minimize = STOP_CRIT_MINIMIZE if STOP_CRIT_COL else False

    if best_metric:
        best_run = get_best_run(df, best_metric, minimize=best_minimize)
        if best_run is not None:
            label = STOP_CRIT_LABEL if STOP_CRIT_COL else "Valid Accuracy"
            print(f"\n=== Best Run (by {label}, selected on valid) ===")
            print(f"Run: {best_run.get('run_name', 'unknown')}")
            for split_label, acc_col, rob_cols in [("Valid", VAL_ACC, VAL_ROB), ("Test", TEST_ACC, TEST_ROB)]:
                if acc_col and acc_col in best_run.index:
                    print(f"  {split_label} Clean Accuracy: {best_run[acc_col]:.4f}")
                for rob_col in rob_cols:
                    if rob_col in best_run.index:
                        strength = rob_col.split("/")[-1]
                        print(f"  {split_label} Robust Accuracy (eps={strength}): {best_run[rob_col]:.4f}")
            if MIA_COL and MIA_COL in best_run.index:
                print(f"  MIA Accuracy: {best_run[MIA_COL]:.4f}")

# %%
# Accuracy vs perturbation strength for best run (test-set performance)
from analysis.utils import plot_accuracy_vs_strength

if not df.empty and TEST_ACC and TEST_ROB and best_run is not None:
    best_run_df = df[df["run_name"] == best_run["run_name"]]
    fig = plot_accuracy_vs_strength(
        best_run_df, acc_col=TEST_ACC, rob_cols=TEST_ROB,
        title=f"Best Run ({best_run['run_name']}): Accuracy vs Perturbation Strength (test)",
        dpi=DPI,
    )
    if fig:
        plt.savefig(output_dir / "best_run_acc_vs_strength.png", bbox_inches="tight")
        print(f"Saved best_run_acc_vs_strength.png")
        plt.show()

# %% [markdown]
# ### 3b. Accuracy Histogram (test)

# %%
from analysis.utils import plot_accuracy_histogram

if not df.empty and TEST_ACC:
    fig = plot_accuracy_histogram(
        df, acc_col=TEST_ACC, rob_cols=TEST_ROB, mia_col=MIA_COL,
        title=f"Accuracy Distribution — test ({sweep_name})", dpi=DPI,
    )
    plt.savefig(output_dir / "accuracy_histogram.png", bbox_inches="tight")
    print(f"Saved accuracy_histogram.png")
    plt.show()

# %% [markdown]
# ### 3c. Mean + Std Bar Plot (test)

# %%
from analysis.utils import plot_mean_with_std

if not df.empty and TEST_ACC:
    fig = plot_mean_with_std(
        df, acc_col=TEST_ACC, rob_cols=TEST_ROB, mia_col=MIA_COL,
        title=f"Mean Accuracies \u00b1 Std Dev — test ({sweep_name})", dpi=DPI,
    )
    if fig:
        plt.savefig(output_dir / "mean_accuracies_errorbars.png", bbox_inches="tight")
        print(f"Saved mean_accuracies_errorbars.png")
        plt.show()

# %% [markdown]
# ### 3d. Metric-Metric Correlation Heatmap (all splits)

# %%
from analysis.utils import compute_metric_correlations, plot_correlation_heatmap

if not df.empty:
    all_eval_metrics = [c for c in df.columns if c.startswith("eval/")]
    if len(all_eval_metrics) >= 2:
        corr_df = compute_metric_correlations(df, all_eval_metrics)
        if not corr_df.empty:
            print("\nMetric-Metric Correlations:")
            print(corr_df.round(3).to_string())
            fig = plot_correlation_heatmap(corr_df, title="Metric-Metric Correlations", dpi=DPI)
            if fig:
                plt.savefig(output_dir / "metric_correlations.png", bbox_inches="tight")
                print(f"Saved metric_correlations.png")
                plt.show()

# %% [markdown]
# ### 3e. Accuracy vs Stopping Criterion Scatter (valid)

# %%
from analysis.utils import plot_scatter_vs_metric

if not df.empty and STOP_CRIT_COL and VAL_ACC:
    fig = plot_scatter_vs_metric(
        df, x_col=STOP_CRIT_COL, acc_col=VAL_ACC, rob_cols=VAL_ROB, mia_col=MIA_COL,
        x_label=STOP_CRIT_LABEL or "Stop Criterion",
        title=f"Accuracy vs Stopping Criterion — valid ({sweep_name})", dpi=DPI,
    )
    if fig:
        plt.savefig(output_dir / "accuracy_vs_stop_crit.png", bbox_inches="tight")
        print(f"Saved accuracy_vs_stop_crit.png")
        plt.show()

# %% [markdown]
# ### 3f. Pareto Frontiers (selected on valid)

# %%
from analysis.utils import plot_pareto_frontier, get_pareto_runs

if not df.empty:
    # Clean accuracy vs robust accuracy (both maximized, on validation)
    if VAL_ACC and VAL_ROB:
        for rob_col in VAL_ROB:
            fig = plot_pareto_frontier(
                df, VAL_ACC, rob_col, maximize1=True, maximize2=True, dpi=DPI,
            )
            if fig:
                strength = rob_col.split("/")[-1]
                plt.savefig(output_dir / f"pareto_acc_vs_rob_{strength}.png", bbox_inches="tight")
                print(f"Saved pareto_acc_vs_rob_{strength}.png")
                plt.show()

                pareto_df = get_pareto_runs(df, VAL_ACC, rob_col, True, True)
                if not pareto_df.empty:
                    print(f"\nPareto-optimal runs (valid acc vs rob/{strength}):")
                    display_cols = ["run_name", VAL_ACC, rob_col]
                    display_cols = [c for c in display_cols if c in pareto_df.columns]
                    print(pareto_df[display_cols].to_string(index=False))

    # MIA accuracy vs robust accuracy (MIA minimized, rob maximized, on validation)
    if MIA_COL and VAL_ROB:
        for rob_col in VAL_ROB:
            fig = plot_pareto_frontier(
                df, MIA_COL, rob_col, maximize1=False, maximize2=True, dpi=DPI,
            )
            if fig:
                strength = rob_col.split("/")[-1]
                plt.savefig(output_dir / f"pareto_mia_vs_rob_{strength}.png", bbox_inches="tight")
                print(f"Saved pareto_mia_vs_rob_{strength}.png")
                plt.show()

    # Accuracy vs strength curves for Pareto-optimal runs (test-set performance)
    if VAL_ACC and VAL_ROB and TEST_ACC and TEST_ROB:
        # Acc vs rob Pareto runs
        for rob_col in VAL_ROB:
            pareto_df = get_pareto_runs(df, VAL_ACC, rob_col, True, True)
            if not pareto_df.empty:
                strength = rob_col.split("/")[-1]
                fig = plot_accuracy_vs_strength(
                    pareto_df, acc_col=TEST_ACC, rob_cols=TEST_ROB,
                    title=f"Pareto Runs (Acc vs Rob/{strength}): Accuracy vs Strength (test)",
                    dpi=DPI,
                )
                if fig:
                    plt.savefig(output_dir / f"pareto_acc_rob_{strength}_acc_vs_strength.png", bbox_inches="tight")
                    print(f"Saved pareto_acc_rob_{strength}_acc_vs_strength.png")
                    plt.show()

        # MIA vs rob Pareto runs
        if MIA_COL:
            for rob_col in VAL_ROB:
                pareto_df = get_pareto_runs(df, MIA_COL, rob_col, False, True)
                if not pareto_df.empty:
                    strength = rob_col.split("/")[-1]
                    fig = plot_accuracy_vs_strength(
                        pareto_df, acc_col=TEST_ACC, rob_cols=TEST_ROB,
                        title=f"Pareto Runs (MIA vs Rob/{strength}): Accuracy vs Strength (test)",
                        dpi=DPI,
                    )
                    if fig:
                        plt.savefig(output_dir / f"pareto_mia_rob_{strength}_acc_vs_strength.png", bbox_inches="tight")
                        print(f"Saved pareto_mia_rob_{strength}_acc_vs_strength.png")
                        plt.show()

# %% [markdown]
# ### 3g. Summary Statistics Table
#
# Both validation and test metrics are reported.  The "Best" column shows
# metrics from the single run selected on validation (stopping criterion).

# %%
from analysis.utils import create_summary_table

if not df.empty and VAL_ACC:
    # Validation summary (with Best from valid-selected run)
    valid_summary = create_summary_table(
        df, acc_col=VAL_ACC, rob_cols=VAL_ROB, mia_col=MIA_COL,
        effective_n=EFFECTIVE_N,
        stop_crit_col=STOP_CRIT_COL,
        stop_crit_minimize=STOP_CRIT_MINIMIZE if STOP_CRIT_COL else True,
    )
    if not valid_summary.empty:
        valid_summary.insert(0, "Split", "valid")

    # Test summary (Best = test values of the same valid-selected run)
    test_summary = create_summary_table(
        df, acc_col=TEST_ACC, rob_cols=TEST_ROB, mia_col=MIA_COL,
        effective_n=EFFECTIVE_N,
        stop_crit_col=STOP_CRIT_COL,
        stop_crit_minimize=STOP_CRIT_MINIMIZE if STOP_CRIT_COL else True,
    )
    if not test_summary.empty:
        test_summary.insert(0, "Split", "test")

    summary_df = pd.concat([valid_summary, test_summary], ignore_index=True)

    if not summary_df.empty:
        print(f"\nEffective N: {EFFECTIVE_N if EFFECTIVE_N else len(df)} (actual runs: {len(df)})")
        print()
        print(summary_df.to_string(index=False))

        csv_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSaved summary to: {csv_path}")

# %% [markdown]
# ## 4. Sanity Check vs W&B Summary Metrics

# %%
if not df.empty and SANITY_CHECK_METRICS:
    from analysis.utils import load_local_hpo_runs

    print("\n" + "=" * 60)
    print("Sanity Check: Post-hoc vs W&B Summary Metrics")
    print("=" * 60)

    # Load W&B summary data
    wb_df = load_local_hpo_runs(sweep_path)

    if not wb_df.empty:
        # Merge on run_name
        merged = df.merge(wb_df, on="run_name", suffixes=("_eval", "_wb"))

        for eval_col, wb_col in SANITY_CHECK_METRICS.items():
            if eval_col not in merged.columns:
                print(f"\n  {eval_col}: not in eval results")
                continue
            # Handle suffix from merge
            wb_actual = wb_col if wb_col in merged.columns else wb_col + "_wb"
            if wb_actual not in merged.columns:
                print(f"\n  {wb_col}: not in W&B summary data")
                continue

            eval_vals = merged[eval_col].astype(float)
            wb_vals = merged[wb_actual].astype(float)
            diff = (eval_vals - wb_vals).abs()

            print(f"\n  {eval_col} vs {wb_col}:")
            print(f"    Max absolute diff: {diff.max():.6f}")
            print(f"    Mean absolute diff: {diff.mean():.6f}")

            mismatches = diff > SANITY_CHECK_TOL
            if mismatches.any():
                print(f"    WARNING: {mismatches.sum()} runs differ by > {SANITY_CHECK_TOL}")
                mismatch_df = merged.loc[mismatches, ["run_name", eval_col, wb_actual]]
                print(mismatch_df.to_string(index=False))
            else:
                print(f"    All runs match within tolerance {SANITY_CHECK_TOL}")
    else:
        print("Could not load W&B summary data for comparison.")

# %% [markdown]
# ## 5. Learned Distribution Visualization

# %%
if not df.empty and best_run is not None:
    best_run_path = best_run.get("run_path")
    if best_run_path:
        print(f"\n--- Visualizing distributions for best run ({best_run['run_name']}) ---")

        # Generated samples
        try:
            from src.tracking.visualisation import visualise_samples
            from analysis.utils import find_model_checkpoint
            from src.models import BornMachine

            bm = BornMachine.load(str(find_model_checkpoint(best_run_path)))
            bm.to(torch.device(DEVICE))
            bm.sync_tensors(after="classification")

            cfg = load_run_config(best_run_path)
            with torch.no_grad():
                synths = bm.sample(cfg=cfg.tracking.sampling)
            ax = visualise_samples(synths)
            if ax is not None:
                fig = ax.get_figure()
                fig.savefig(output_dir / "best_run_samples.png", bbox_inches="tight", dpi=DPI)
                print(f"Saved best_run_samples.png")
                plt.show()

            del bm
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not generate samples: {e}")

        # Distribution heatmaps
        try:
            from analysis.visualize_distributions import visualize_from_run_dir

            fig = visualize_from_run_dir(
                run_dir=best_run_path,
                resolution=150,
                normalize_joint=True,
                show_data=True,
                device=DEVICE,
                save_dir=str(output_dir),
            )
            default_path = output_dir / "distributions.png"
            final_path = output_dir / "best_run_distributions.png"
            if default_path.exists():
                default_path.rename(final_path)
                print(f"Saved best_run_distributions.png")
            plt.show()
        except Exception as e:
            print(f"Warning: Could not generate distribution visualization: {e}")

# %% [markdown]
# ## 6. Summary Export

# %%
if not df.empty:
    summary_path = output_dir / "sweep_analysis_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Post-Hoc Sweep Analysis Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Sweep: {sweep_name}\n")
        f.write(f"Regime: {REGIME}\n")
        f.write(f"Total runs: {len(df)}\n")
        f.write(f"Eval splits: {EVAL_SPLITS}\n")
        f.write(f"Selection split: valid\n")
        f.write(f"Device: {DEVICE}\n")
        if EFFECTIVE_N:
            f.write(f"Effective N: {EFFECTIVE_N}\n")
        if EVASION_OVERRIDE:
            f.write(f"Evasion override: {EVASION_OVERRIDE}\n")
        f.write("\n")

        # Metrics used
        f.write("Metrics computed:\n")
        if VAL_ACC:
            f.write(f"  Accuracy: {VAL_ACC}, {TEST_ACC}\n")
        if VAL_ROB:
            f.write(f"  Robustness (valid): {VAL_ROB}\n")
            f.write(f"  Robustness (test):  {TEST_ROB}\n")
        if MIA_COL:
            f.write(f"  MIA: {MIA_COL}\n")
        if VAL_CLS_LOSS:
            f.write(f"  Cls Loss: {VAL_CLS_LOSS}, {TEST_CLS_LOSS}\n")
        f.write("\n")

        # Summary statistics
        if "summary_df" in dir() and not summary_df.empty:
            f.write("-" * 60 + "\n")
            f.write("Summary Statistics\n")
            f.write("-" * 60 + "\n\n")
            f.write(summary_df.to_string(index=False) + "\n\n")

        # Best run (selected on validation)
        best_metric = STOP_CRIT_COL if STOP_CRIT_COL else VAL_ACC
        best_minimize = STOP_CRIT_MINIMIZE if STOP_CRIT_COL else False
        if best_metric:
            best_run_export = get_best_run(df, best_metric, minimize=best_minimize)
            label = STOP_CRIT_LABEL if STOP_CRIT_COL else "Valid Accuracy"
            f.write("-" * 60 + "\n")
            f.write(f"Best Run (by {label}, selected on valid)\n")
            f.write("-" * 60 + "\n\n")

            if best_run_export is not None:
                f.write(f"Run: {best_run_export.get('run_name', 'unknown')}\n")
                if "run_path" in best_run_export.index:
                    f.write(f"Path: {best_run_export['run_path']}\n")
                for split_label, acc_col, rob_cols in [("Valid", VAL_ACC, VAL_ROB), ("Test", TEST_ACC, TEST_ROB)]:
                    if acc_col and acc_col in best_run_export.index:
                        f.write(f"{split_label} Clean Accuracy: {best_run_export[acc_col]:.4f}\n")
                    for rob_col in rob_cols:
                        if rob_col in best_run_export.index:
                            strength = rob_col.split("/")[-1]
                            f.write(f"{split_label} Robust Accuracy (eps={strength}): {best_run_export[rob_col]:.4f}\n")
                if MIA_COL and MIA_COL in best_run_export.index:
                    f.write(f"MIA Accuracy: {best_run_export[MIA_COL]:.4f}\n")

        # Pareto-optimal runs (selected on validation)
        if VAL_ACC and VAL_ROB:
            f.write("\n" + "-" * 60 + "\n")
            f.write("Pareto-Optimal Runs (selected on valid)\n")
            f.write("-" * 60 + "\n\n")
            for rob_col in VAL_ROB:
                pareto_df = get_pareto_runs(df, VAL_ACC, rob_col, True, True)
                if not pareto_df.empty:
                    strength = rob_col.split("/")[-1]
                    f.write(f"Valid Acc vs Rob/{strength}:\n")
                    display_cols = ["run_name", VAL_ACC, rob_col]
                    display_cols = [c for c in display_cols if c in pareto_df.columns]
                    f.write(pareto_df[display_cols].to_string(index=False) + "\n\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\nExported summary to: {summary_path}")

# %% [markdown]
# ## Completion

# %%
print("\n" + "=" * 60)
print("Post-Hoc Sweep Analysis Complete!")
print("=" * 60)
print(f"\nSweep: {sweep_name}")
print(f"Regime: {REGIME}")
print(f"Total runs: {len(df) if not df.empty else 0}")
print(f"\nOutputs saved to: {output_dir}")
print("=" * 60)

# %%
