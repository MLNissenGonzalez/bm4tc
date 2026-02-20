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
import argparse
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
# Can be overridden from the command line:
#   python analysis/sweep_analysis.py outputs/seed_sweep/cls/fourier/d4D3/circles_4k_1802
SWEEP_DIR = "outputs/seed_sweep_uq_gen_d30D18fourier_moons_4k_1702"
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("sweep_dir", nargs="?", default=None)
_cli_args, _ = _cli.parse_known_args()
if _cli_args.sweep_dir is not None:
    SWEEP_DIR = _cli_args.sweep_dir

# Training regime: "pre", "gen", "adv", "gan".
# Auto-detected from SWEEP_DIR (which encodes the regime via the ${training_regime:} resolver).
# Override manually only if auto-detection gives the wrong result.
from analysis.utils.resolve import resolve_regime_from_path as _resolve_regime_from_path
REGIME = _resolve_regime_from_path(SWEEP_DIR)
if REGIME is None:
    print(
        f"WARNING: Could not auto-detect training regime from '{SWEEP_DIR}'.\n"
        "  Set REGIME manually to one of: 'pre', 'gen', 'adv', 'gan'."
    )
    REGIME = "pre"  # fallback — change if incorrect
else:
    print(f"Auto-detected training regime: '{REGIME}' (from sweep_dir)")

from analysis.utils.resolve import resolve_embedding_from_path as _resolve_embedding
from analysis.utils.resolve import embedding_range_size as _embedding_range_size
_EMBEDDING = _resolve_embedding(SWEEP_DIR)
if _EMBEDDING is None:
    print(f"WARNING: Could not detect embedding from '{SWEEP_DIR}'. Assuming Fourier (range size 1.0).")
_RANGE_SIZE = _embedding_range_size(_EMBEDDING)
print(f"Embedding: '{_EMBEDDING or 'unknown'}'  →  input range size: {_RANGE_SIZE}")

# --- METRIC TOGGLES ---
COMPUTE_ACC = True
COMPUTE_ROB = True
COMPUTE_MIA = True
COMPUTE_CLS_LOSS = False
COMPUTE_GEN_LOSS = False
COMPUTE_FID = False
COMPUTE_UQ = True  # Uncertainty quantification (detection + purification)

# --- EVASION CONFIG (single source of truth for all adversarial attacks) ---
# Applies to: robustness eval, UQ adversarial examples, adversarial MIA.
# Set to None to use each run's own evasion config.
# Strengths are expressed as FRACTIONS of the input range:
#   Fourier (range 1.0): multiply by 1.0 → same value
#   Legendre (range 2.0): multiply by 2.0
_STRENGTH_FRACTIONS = [0.05, 0.10, 0.2, 0.5, 0.8]
EVASION_CONFIG = {
    "method": "PGD",
    "norm": 2,                                            # L2 (was "inf")
    "num_steps": 20,
    "strengths": [s * _RANGE_SIZE for s in _STRENGTH_FRACTIONS],
}

# --- SAMPLING OVERRIDE ---
# Set to a dict to override sampling config, or None to use per-run config.
SAMPLING_OVERRIDE = None

# --- MIA SETTINGS ---
# Feature toggles: which confidence features to extract from p(c|x).
# Label-free features (max_prob, entropy, margin, modified_entropy) are always
# derived from the probability vector alone.  correct_prob and loss require a
# reference label — controlled by use_true_labels below.
MIA_FEATURES = {
    "max_prob": True,
    "entropy": True,
    "correct_prob": True,
    "loss": False,
    "margin": False,
    "modified_entropy": False,
    # True  = use ground-truth labels for correct_prob/loss (worst-case risk).
    # False = use predicted labels (argmax of probs) to avoid label leakage.
    "use_true_labels": True,
}

# --- MIA ADVERSARIAL SETTINGS ---
# Set MIA_ADV_STRENGTH to None to skip adversarial MIA entirely.
# Attack settings (method, norm, num_steps) are derived from EVASION_CONFIG.
MIA_ADV_STRENGTH = 0.15 * _RANGE_SIZE  # 15% of input range; None = disabled.
                                       # Added to EVASION_CONFIG["strengths"] automatically.

# --- UQ SETTINGS (UQ-specific params only; attack settings from EVASION_CONFIG) ---
UQ_CONFIG = {
    "radii": [0.10 * _RANGE_SIZE],   # single radius: 10% of input range (was [0.15, 0.3])
    "percentiles": [1, 5, 10, 20],
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

# --- PARETO SETTINGS ---
# Robustness strength for Pareto frontier selection.
# Set to None to auto-select the weakest non-zero strength.
PARETO_ROB_STRENGTH = 0.10 * _RANGE_SIZE   # 10% of input range (was 0.15 absolute)

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
# Add PARETO and MIA strengths to EVASION_CONFIG; sort. Build full UQ config.
if EVASION_CONFIG:
    _strengths = [float(s) for s in EVASION_CONFIG.get("strengths", [])]
    if COMPUTE_ROB and PARETO_ROB_STRENGTH is not None:
        if float(PARETO_ROB_STRENGTH) not in _strengths:
            _strengths.append(float(PARETO_ROB_STRENGTH))
    if COMPUTE_MIA and MIA_ADV_STRENGTH is not None:
        if float(MIA_ADV_STRENGTH) not in _strengths:
            _strengths.append(float(MIA_ADV_STRENGTH))
    EVASION_CONFIG["strengths"] = sorted(set(_strengths))
    print(f"Final attack strengths: {EVASION_CONFIG['strengths']}")
elif PARETO_ROB_STRENGTH is not None and COMPUTE_ROB:
    print(f"Note: EVASION_CONFIG is None; using per-run evasion configs. "
          f"Ensure each run includes eps={PARETO_ROB_STRENGTH}.")

_full_uq_config = None
if COMPUTE_UQ and UQ_CONFIG is not None and EVASION_CONFIG:
    _full_uq_config = {
        "norm":             EVASION_CONFIG.get("norm", "inf"),
        "num_steps":        EVASION_CONFIG.get("num_steps", 20),  # purification steps
        "attack_method":    EVASION_CONFIG.get("method", "PGD"),
        "attack_strengths": EVASION_CONFIG["strengths"],
        "attack_num_steps": EVASION_CONFIG.get("num_steps", 20),
        **UQ_CONFIG,  # radii, percentiles (may override num_steps if user adds it)
    }

# %%
from analysis.utils import EvalConfig, evaluate_sweep

eval_cfg = EvalConfig(
    compute_acc=COMPUTE_ACC,
    compute_rob=COMPUTE_ROB,
    compute_mia=COMPUTE_MIA,
    compute_cls_loss=COMPUTE_CLS_LOSS,
    compute_gen_loss=COMPUTE_GEN_LOSS,
    compute_fid=COMPUTE_FID,
    compute_uq=COMPUTE_UQ,
    splits=EVAL_SPLITS,
    evasion_override=EVASION_CONFIG,
    sampling_override=SAMPLING_OVERRIDE,
    mia_features=MIA_FEATURES,
    mia_adversarial_strength=MIA_ADV_STRENGTH,
    mia_adversarial_num_steps=EVASION_CONFIG.get("num_steps", 20) if EVASION_CONFIG else 20,
    mia_adversarial_step_size=None,
    mia_adversarial_norm=EVASION_CONFIG.get("norm", "inf") if EVASION_CONFIG else "inf",
    uq_config=_full_uq_config if COMPUTE_UQ else None,
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
# Mirror sweep path under analysis/outputs/:
#   outputs/seed_sweep/X/Y/Z  →  analysis/outputs/seed_sweep/X/Y/Z
_sp = Path(SWEEP_DIR)
if _sp.is_absolute():
    try:
        _sp = _sp.relative_to(project_root)
    except ValueError:
        pass
try:
    _rel = _sp.relative_to("outputs")
except ValueError:
    _rel = _sp
output_dir = project_root / "analysis" / "outputs" / _rel
output_dir.mkdir(parents=True, exist_ok=True)
sweep_name = str(_rel)  # human-readable label used in plot titles and summary
mia_output_dir = output_dir / "mia"
mia_output_dir.mkdir(parents=True, exist_ok=True)
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

# Adversarial MIA: best worst-case threshold accuracy across all features
ADV_MIA_COL = None
ADV_MIA_FEATURE_COLS = []
if COMPUTE_MIA and MIA_ADV_STRENGTH is not None and not df.empty:
    if "eval/adv_mia_wc_best" in df.columns:
        ADV_MIA_COL = "eval/adv_mia_wc_best"
    ADV_MIA_FEATURE_COLS = [c for c in df.columns if c.startswith("eval/adv_mia_wc/")]

# Clean worst-case threshold (for apples-to-apples comparison with adversarial MIA)
MIA_WC_COL = None
MIA_WC_FEATURE_COLS = []
if COMPUTE_MIA and MIA_ADV_STRENGTH is not None and not df.empty:
    if "eval/mia_wc_best" in df.columns:
        MIA_WC_COL = "eval/mia_wc_best"
    MIA_WC_FEATURE_COLS = [c for c in df.columns if c.startswith("eval/mia_wc/")]

UQ_ADV_ACC_COLS = []
UQ_PURIFY_ACC_COLS = []
UQ_PURIFY_RECOVERY_COLS = []
UQ_DETECTION_COLS = []
if COMPUTE_UQ and not df.empty:
    UQ_ADV_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_adv_acc/"))
    UQ_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_purify_acc/"))
    UQ_PURIFY_RECOVERY_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_purify_recovery/"))
    UQ_DETECTION_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_detection/"))

print(f"VAL_ACC:      {VAL_ACC}")
print(f"VAL_ROB:      {VAL_ROB}")
print(f"TEST_ACC:     {TEST_ACC}")
print(f"TEST_ROB:     {TEST_ROB}")
print(f"MIA_COL:      {MIA_COL}")
print(f"ADV_MIA_COL:  {ADV_MIA_COL}")
print(f"MIA_WC_COL:   {MIA_WC_COL}")
if ADV_MIA_FEATURE_COLS:
    print(f"ADV_MIA per-feature: {ADV_MIA_FEATURE_COLS}")

# Resolve single robustness strength for Pareto frontier selection
PARETO_VAL_ROB_COL = None
PARETO_TEST_ROB_COL = None

if VAL_ROB:
    # Parse available non-zero strengths from VAL_ROB column names
    _strength_map = {}
    for col in VAL_ROB:
        try:
            s = float(col.split("/")[-1])
            if s > 0:
                _strength_map[s] = col
        except ValueError:
            continue

    if _strength_map:
        if PARETO_ROB_STRENGTH is not None:
            if PARETO_ROB_STRENGTH in _strength_map:
                _chosen = PARETO_ROB_STRENGTH
            else:
                print(f"WARNING: PARETO_ROB_STRENGTH={PARETO_ROB_STRENGTH} not in evaluated "
                      f"strengths {sorted(_strength_map.keys())}. Falling back to weakest.")
                _chosen = min(_strength_map.keys())
        else:
            _chosen = min(_strength_map.keys())

        PARETO_VAL_ROB_COL = _strength_map[_chosen]
        # Derive matching test column
        _test_candidate = PARETO_VAL_ROB_COL.replace("eval/valid/", "eval/test/")
        if _test_candidate in (TEST_ROB or []):
            PARETO_TEST_ROB_COL = _test_candidate

        print(f"\nPareto robustness strength: eps={_chosen}")
        print(f"  PARETO_VAL_ROB_COL: {PARETO_VAL_ROB_COL}")
        print(f"  PARETO_TEST_ROB_COL: {PARETO_TEST_ROB_COL}")

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
                # If stop crit is "rob", prefer the Pareto strength over averaging
                if stop_name == "rob" and PARETO_VAL_ROB_COL and PARETO_VAL_ROB_COL in df.columns:
                    STOP_CRIT_COL = PARETO_VAL_ROB_COL
                    _s = PARETO_VAL_ROB_COL.split("/")[-1]
                    STOP_CRIT_LABEL = f"Robust Accuracy eps={_s} (stop crit)"
                    STOP_CRIT_MINIMIZE = False
                    print(f"  -> Overridden to Pareto strength: {STOP_CRIT_COL}")
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
            if MIA_WC_COL and MIA_WC_COL in best_run.index:
                print(f"  MIA WC Threshold (clean):              {best_run[MIA_WC_COL]:.4f}")
            if ADV_MIA_COL and ADV_MIA_COL in best_run.index:
                print(f"  MIA WC Threshold (adv, eps={MIA_ADV_STRENGTH}): {best_run[ADV_MIA_COL]:.4f}")
                for col in ADV_MIA_FEATURE_COLS:
                    if col in best_run.index:
                        feat = col.split("/")[-1]
                        print(f"    {feat}: {best_run[col]:.4f}")
            if COMPUTE_UQ and UQ_PURIFY_ACC_COLS:
                print(f"  --- UQ (Detection + Purification on test) ---")
                uq_clean = best_run.get("eval/uq_clean_accuracy")
                if uq_clean is not None and not np.isnan(uq_clean):
                    print(f"  UQ Clean Acc: {uq_clean:.4f}")
                for adv_col in UQ_ADV_ACC_COLS:
                    eps = adv_col.split("/")[-1]
                    adv_val = best_run.get(adv_col, float("nan"))
                    print(f"  Adv Acc (eps={eps}, no defense): {adv_val:.4f}")
                    for col in [c for c in UQ_PURIFY_ACC_COLS if f"/{eps}/" in c]:
                        radius = col.split("/")[-1]
                        purify_val = best_run.get(col, float("nan"))
                        delta = purify_val - adv_val if not (np.isnan(purify_val) or np.isnan(adv_val)) else float("nan")
                        delta_str = f"  (Δ={delta:+.4f})" if not np.isnan(delta) else ""
                        print(f"  Purify Acc (eps={eps}, r={radius}): {purify_val:.4f}{delta_str}")
                if UQ_DETECTION_COLS:
                    print(f"  Detection rates (fraction of adv examples detected as OOD):")
                    for col in UQ_DETECTION_COLS:
                        parts = col.split("/")
                        pct_str, eps = parts[-2], parts[-1]
                        rate = best_run.get(col, float("nan"))
                        print(f"    tau={pct_str}, eps={eps}: {rate:.4f}")

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
    # Use only the Pareto robustness strength (the band plot shows all strengths)
    _hist_rob = [PARETO_TEST_ROB_COL] if PARETO_TEST_ROB_COL else []
    fig = plot_accuracy_histogram(
        df, acc_col=TEST_ACC, rob_cols=_hist_rob, mia_col=MIA_COL,
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
    # Use only the Pareto robustness strength (the band plot shows all strengths)
    _bar_rob = [PARETO_TEST_ROB_COL] if PARETO_TEST_ROB_COL else []
    fig = plot_mean_with_std(
        df, acc_col=TEST_ACC, rob_cols=_bar_rob, mia_col=MIA_COL,
        title=f"Mean Accuracies \u00b1 Std Dev — test ({sweep_name})", dpi=DPI,
    )
    if fig:
        plt.savefig(output_dir / "mean_accuracies_errorbars.png", bbox_inches="tight")
        print(f"Saved mean_accuracies_errorbars.png")
        plt.show()

# %% [markdown]
# ### 3d. Metric-Metric Correlation Heatmaps (per split) & Cross-Split Consistency

# %%
from analysis.utils import compute_metric_correlations, plot_correlation_heatmap
from scipy.stats import pearsonr

if not df.empty:
    # --- Valid-only heatmap ---
    valid_metrics = [c for c in [VAL_ACC, VAL_CLS_LOSS, MIA_COL, MIA_WC_COL, ADV_MIA_COL] if c] + list(VAL_ROB)
    # Keep only columns present in df with at least 2 distinct values (drops constants → avoids NaN rows)
    valid_metrics = [c for c in valid_metrics if c in df.columns and df[c].nunique() > 1]

    if len(valid_metrics) >= 2:
        corr_valid = compute_metric_correlations(df, valid_metrics)
        if not corr_valid.empty:
            print("\nMetric-Metric Correlations (valid):")
            print(corr_valid.round(3).to_string())
            fig = plot_correlation_heatmap(corr_valid, title="Metric Correlations — valid", dpi=DPI)
            if fig:
                plt.savefig(output_dir / "metric_correlations_valid.png", bbox_inches="tight")
                print("Saved metric_correlations_valid.png")
                plt.show()

    # --- Test-only heatmap ---
    test_metrics = [c for c in [TEST_ACC, TEST_CLS_LOSS, MIA_COL, MIA_WC_COL, ADV_MIA_COL] if c] + list(TEST_ROB)
    # Keep only columns present in df with at least 2 distinct values (drops constants → avoids NaN rows)
    test_metrics = [c for c in test_metrics if c in df.columns and df[c].nunique() > 1]

    if len(test_metrics) >= 2:
        corr_test = compute_metric_correlations(df, test_metrics)
        if not corr_test.empty:
            print("\nMetric-Metric Correlations (test):")
            print(corr_test.round(3).to_string())
            fig = plot_correlation_heatmap(corr_test, title="Metric Correlations — test", dpi=DPI)
            if fig:
                plt.savefig(output_dir / "metric_correlations_test.png", bbox_inches="tight")
                print("Saved metric_correlations_test.png")
                plt.show()

    # --- Cross-split consistency ---
    valid_cols = [c for c in df.columns if c.startswith("eval/valid/")]
    cross_rows = []
    for vc in valid_cols:
        tc = vc.replace("eval/valid/", "eval/test/")
        if tc in df.columns:
            valid_vals = df[vc].dropna()
            test_vals = df[tc].dropna()
            common = valid_vals.index.intersection(test_vals.index)
            if len(common) >= 3:
                r, _ = pearsonr(df.loc[common, vc], df.loc[common, tc])
                cross_rows.append({"valid_col": vc, "test_col": tc, "pearson_r": r})

    if cross_rows:
        cross_df = pd.DataFrame(cross_rows)
        mean_r = cross_df["pearson_r"].mean()
        print("\nCross-Split Consistency (valid vs test, Pearson r across runs):")
        print(cross_df.to_string(index=False))
        print(f"Mean cross-split r: {mean_r:.4f}")

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
from analysis.utils import plot_pareto_frontier, get_pareto_runs, plot_accuracy_vs_strength_band

if not df.empty and PARETO_VAL_ROB_COL:
    _pareto_strength = PARETO_VAL_ROB_COL.split("/")[-1]

    # Clean accuracy vs robust accuracy (both maximized, on validation)
    if VAL_ACC:
        fig = plot_pareto_frontier(
            df, VAL_ACC, PARETO_VAL_ROB_COL, maximize1=True, maximize2=True, dpi=DPI,
        )
        if fig:
            plt.savefig(output_dir / f"pareto_acc_vs_rob_{_pareto_strength}.png", bbox_inches="tight")
            print(f"Saved pareto_acc_vs_rob_{_pareto_strength}.png")
            plt.show()

            pareto_df = get_pareto_runs(df, VAL_ACC, PARETO_VAL_ROB_COL, True, True)
            if not pareto_df.empty:
                print(f"\nPareto-optimal runs (valid acc vs rob/{_pareto_strength}):")
                display_cols = ["run_name", VAL_ACC, PARETO_VAL_ROB_COL]
                display_cols = [c for c in display_cols if c in pareto_df.columns]
                print(pareto_df[display_cols].to_string(index=False))

    # MIA accuracy vs robust accuracy (MIA minimized, rob maximized, on validation)
    if MIA_COL:
        fig = plot_pareto_frontier(
            df, MIA_COL, PARETO_VAL_ROB_COL, maximize1=False, maximize2=True, dpi=DPI,
        )
        if fig:
            plt.savefig(output_dir / f"pareto_mia_vs_rob_{_pareto_strength}.png", bbox_inches="tight")
            print(f"Saved pareto_mia_vs_rob_{_pareto_strength}.png")
            plt.show()

    # Mean +/- std band plot across all runs (test-set performance)
    if TEST_ACC and TEST_ROB:
        fig = plot_accuracy_vs_strength_band(
            df, acc_col=TEST_ACC, rob_cols=TEST_ROB,
            title=f"Mean Accuracy vs Perturbation Strength — test ({sweep_name})",
            dpi=DPI,
        )
        if fig:
            plt.savefig(output_dir / "mean_acc_vs_strength_band.png", bbox_inches="tight")
            print(f"Saved mean_acc_vs_strength_band.png")
            plt.show()

# %% [markdown]
# ### 3g. Adversarial MIA Results

# %%
# Save per-run MIA summary text (includes both benign and adversarial sections)
if not df.empty and COMPUTE_MIA and "_mia_summary" in df.columns:
    for _, row in df.iterrows():
        summary = row.get("_mia_summary")
        if summary and isinstance(summary, str):
            mia_path = mia_output_dir / f"mia_summary_{row['run_name']}.txt"
            with open(mia_path, "w") as f:
                f.write(summary)
    print(f"Saved per-run MIA summaries to {mia_output_dir}/")

if not df.empty and ADV_MIA_COL and ADV_MIA_FEATURE_COLS:
    print(f"\nAdversarial MIA Worst-Case Threshold (eps={MIA_ADV_STRENGTH}):")
    print("  Per-feature accuracy (oracle threshold, mean +/- std across runs):\n")
    if MIA_WC_COL and MIA_WC_COL in df.columns:
        vals_wc = df[MIA_WC_COL].dropna()
        print(f"    {'WC Threshold (clean)':25s}  {vals_wc.mean():.4f} +/- {vals_wc.std():.4f}")
    for col in sorted(ADV_MIA_FEATURE_COLS):
        feat = col.split("/")[-1]
        vals = df[col].dropna()
        print(f"    {feat:25s}  {vals.mean():.4f} +/- {vals.std():.4f}")

    if ADV_MIA_COL in df.columns:
        vals = df[ADV_MIA_COL].dropna()
        print(f"\n    {'BEST (across features)':25s}  {vals.mean():.4f} +/- {vals.std():.4f}")

# %% [markdown]
# ### 3h. UQ Results (detection + purification across all runs)

# %%
if not df.empty and COMPUTE_UQ and UQ_PURIFY_ACC_COLS:
    print(f"\nUQ Purification Results (mean +/- std across runs, test split):\n")
    for adv_col in UQ_ADV_ACC_COLS:
        eps = adv_col.split("/")[-1]
        adv_vals = df[adv_col].dropna()
        print(f"  Adv Acc (eps={eps}, no defense): {adv_vals.mean():.4f} +/- {adv_vals.std():.4f}")
        for col in [c for c in UQ_PURIFY_ACC_COLS if f"/{eps}/" in c]:
            radius = col.split("/")[-1]
            purify_vals = df[col].dropna()
            adv_mean = adv_vals.mean()
            purify_mean, purify_std = purify_vals.mean(), purify_vals.std()
            delta = purify_mean - adv_mean
            print(f"  Purify Acc (eps={eps}, r={radius}): "
                  f"{purify_mean:.4f} +/- {purify_std:.4f}  (Δ={delta:+.4f})")
    if UQ_DETECTION_COLS:
        print(f"\nUQ Detection Rates (mean +/- std across runs):\n")
        for col in UQ_DETECTION_COLS:
            parts = col.split("/")
            pct_str, eps = parts[-2], parts[-1]
            vals = df[col].dropna()
            print(f"  tau={pct_str}, eps={eps}: {vals.mean():.4f} +/- {vals.std():.4f}")

# %% [markdown]
# ### 3i. Summary Statistics Table
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
            ax = visualise_samples(synths, input_range=bm.input_range)
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
        if EVASION_CONFIG:
            f.write(f"Evasion config: {EVASION_CONFIG}\n")
        if MIA_ADV_STRENGTH is not None:
            f.write(f"Adversarial MIA: eps={MIA_ADV_STRENGTH}, "
                    f"steps={EVASION_CONFIG.get('num_steps', 20) if EVASION_CONFIG else 20}, "
                    f"norm={EVASION_CONFIG.get('norm', 'inf') if EVASION_CONFIG else 'inf'}\n")
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
        if ADV_MIA_COL:
            f.write(f"  Adversarial MIA: {ADV_MIA_COL}\n")
        if VAL_CLS_LOSS:
            f.write(f"  Cls Loss: {VAL_CLS_LOSS}, {TEST_CLS_LOSS}\n")
        if COMPUTE_UQ and UQ_PURIFY_ACC_COLS:
            f.write(f"  UQ: {len(UQ_PURIFY_ACC_COLS)} purification (eps, radius) pairs, "
                    f"{len(UQ_DETECTION_COLS)} detection (pct, eps) pairs\n")
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
                if MIA_WC_COL and MIA_WC_COL in best_run_export.index:
                    f.write(f"MIA WC Threshold (clean): {best_run_export[MIA_WC_COL]:.4f}\n")
                if ADV_MIA_COL and ADV_MIA_COL in best_run_export.index:
                    f.write(f"MIA WC Threshold (adv, eps={MIA_ADV_STRENGTH}): "
                            f"{best_run_export[ADV_MIA_COL]:.4f}\n")
                    for col in ADV_MIA_FEATURE_COLS:
                        if col in best_run_export.index:
                            feat = col.split("/")[-1]
                            f.write(f"  Adv MIA {feat}: {best_run_export[col]:.4f}\n")
                if COMPUTE_UQ and best_run_export is not None and UQ_PURIFY_ACC_COLS:
                    uq_clean = best_run_export.get("eval/uq_clean_accuracy")
                    if uq_clean is not None and not np.isnan(float(uq_clean)):
                        f.write(f"UQ Clean Acc: {float(uq_clean):.4f}\n")
                    for adv_col in UQ_ADV_ACC_COLS:
                        eps = adv_col.split("/")[-1]
                        adv_val = best_run_export.get(adv_col, float("nan"))
                        f.write(f"UQ Adv Acc (eps={eps}, no defense): {float(adv_val):.4f}\n")
                        for col in [c for c in UQ_PURIFY_ACC_COLS if f"/{eps}/" in c]:
                            radius = col.split("/")[-1]
                            purify_val = best_run_export.get(col, float("nan"))
                            delta = float(purify_val) - float(adv_val)
                            f.write(f"UQ Purify Acc (eps={eps}, r={radius}): "
                                    f"{float(purify_val):.4f}  (Δ={delta:+.4f})\n")
                    if UQ_DETECTION_COLS:
                        for col in UQ_DETECTION_COLS:
                            parts = col.split("/")
                            pct_str, eps = parts[-2], parts[-1]
                            rate = best_run_export.get(col, float("nan"))
                            f.write(f"UQ Detect Rate (tau={pct_str}, eps={eps}): {float(rate):.4f}\n")

        # Adversarial MIA summary across all runs
        if ADV_MIA_COL and ADV_MIA_COL in df.columns:
            f.write("\n" + "-" * 60 + "\n")
            f.write(f"Adversarial MIA Worst-Case Threshold (eps={MIA_ADV_STRENGTH})\n")
            f.write("-" * 60 + "\n\n")
            if MIA_WC_COL and MIA_WC_COL in df.columns:
                vals_wc = df[MIA_WC_COL].dropna()
                f.write(f"  {'WC Threshold (clean)':25s}  {vals_wc.mean():.4f} +/- {vals_wc.std():.4f}\n")
            for col in sorted(ADV_MIA_FEATURE_COLS):
                feat = col.split("/")[-1]
                vals = df[col].dropna()
                f.write(f"  {feat:25s}  {vals.mean():.4f} +/- {vals.std():.4f}\n")
            vals = df[ADV_MIA_COL].dropna()
            f.write(f"\n  {'BEST (across features)':25s}  {vals.mean():.4f} +/- {vals.std():.4f}\n")

        if COMPUTE_UQ and UQ_PURIFY_ACC_COLS:
            f.write("\n" + "-" * 60 + "\n")
            f.write("UQ Results — Purification (mean +/- std across runs)\n")
            f.write("-" * 60 + "\n\n")
            for adv_col in UQ_ADV_ACC_COLS:
                eps = adv_col.split("/")[-1]
                adv_vals = df[adv_col].dropna()
                f.write(f"  Adv Acc (eps={eps}, no defense)  "
                        f"{adv_vals.mean():.4f} +/- {adv_vals.std():.4f}\n")
                for col in [c for c in UQ_PURIFY_ACC_COLS if f"/{eps}/" in c]:
                    radius = col.split("/")[-1]
                    purify_vals = df[col].dropna()
                    delta = purify_vals.mean() - adv_vals.mean()
                    f.write(f"  Purify Acc (eps={eps}, r={radius})  "
                            f"{purify_vals.mean():.4f} +/- {purify_vals.std():.4f}  "
                            f"(Δ={delta:+.4f})\n")
            if UQ_DETECTION_COLS:
                f.write("\n")
                for col in UQ_DETECTION_COLS:
                    parts = col.split("/")
                    pct_str, eps = parts[-2], parts[-1]
                    vals = df[col].dropna()
                    f.write(f"  Detect Rate (tau={pct_str}, eps={eps})  "
                            f"{vals.mean():.4f} +/- {vals.std():.4f}\n")

        # Pareto-optimal runs (selected on validation)
        if VAL_ACC and PARETO_VAL_ROB_COL:
            f.write("\n" + "-" * 60 + "\n")
            f.write("Pareto-Optimal Runs (selected on valid)\n")
            f.write("-" * 60 + "\n\n")
            pareto_df = get_pareto_runs(df, VAL_ACC, PARETO_VAL_ROB_COL, True, True)
            if not pareto_df.empty:
                strength = PARETO_VAL_ROB_COL.split("/")[-1]
                f.write(f"Valid Acc vs Rob/{strength}:\n")
                display_cols = ["run_name", VAL_ACC, PARETO_VAL_ROB_COL]
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
