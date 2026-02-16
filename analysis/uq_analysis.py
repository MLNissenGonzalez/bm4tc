# %% [markdown]
# # Uncertainty Quantification (UQ) Analysis
#
# This notebook evaluates two defenses against adversarial examples using
# the Born Machine's marginal likelihood p(x):
#
# 1. **Detection**: Reject inputs whose log p(x) falls below a threshold tau
#    (calibrated from clean data percentiles)
# 2. **Purification**: For adversarial inputs, gradient-descend on NLL within
#    a perturbation ball to find a nearby high-likelihood point, then classify
#
# **Data Sources:**
# - **local**: Loads model and config from Hydra output folder
# - **wandb**: Fetches config from W&B, requires checkpoint download

# %% [markdown]
# ## Setup and Configuration

# %%
import sys
from pathlib import Path

# Handle both script and interactive execution
if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    # Interactive/notebook mode - assume we're in analysis/
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION FOR YOUR EXPERIMENT
# =============================================================================

# Data source: "local" or "wandb"
DATA_SOURCE = "local"

# --- LOCAL SETTINGS (used if DATA_SOURCE == "local") ---
RUN_DIR = "outputs/gen_seed_sweep_circles_4k_02Feb26/1"  # Change to your run directory

# --- WANDB SETTINGS (used if DATA_SOURCE == "wandb") ---
WANDB_ENTITY = "your-entity"
WANDB_PROJECT = "gan_train"
WANDB_RUN_ID = "abc123"
WANDB_CHECKPOINT_PATH = None

# --- UQ SETTINGS ---
# Purification parameters
PURIFICATION_NORM = "inf"
PURIFICATION_NUM_STEPS = 20
PURIFICATION_STEP_SIZE = None  # None = auto (2.5 * radius / num_steps)
PURIFICATION_RADII = [0.05, 0.1, 0.15, 0.2, 0.3]
PURIFICATION_RANDOM_START = False
PURIFICATION_EPS = 1e-12

# Detection thresholds (percentiles of clean log p(x))
THRESHOLD_PERCENTILES = [1, 5, 10, 20]

# Attack parameters for generating adversarial test inputs
ATTACK_METHOD = "PGD"
ATTACK_STRENGTHS = [0.1, 0.2, 0.3]
ATTACK_NUM_STEPS = 20

# --- OUTPUT SETTINGS ---
OUTPUT_DIR = "analysis/outputs/uq"

# --- DEVICE SETTINGS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Load Model and Data

# %%
def load_model_and_data(source: str = DATA_SOURCE):
    """
    Load trained BornMachine and reconstruct DataHandler from config.

    Args:
        source: "local" or "wandb"

    Returns:
        Tuple of (bornmachine, datahandler, device, cfg)
    """
    from analysis.utils import (
        load_run_config,
        load_run_config_from_wandb,
        find_model_checkpoint,
    )

    from src.models import BornMachine
    from src.data import DataHandler

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    if source == "local":
        run_dir = Path(RUN_DIR)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir

        logger.info(f"Loading config from: {run_dir}")
        cfg = load_run_config(run_dir)

        checkpoint_path = find_model_checkpoint(run_dir)
        logger.info(f"Loading model from: {checkpoint_path}")
        bornmachine = BornMachine.load(str(checkpoint_path))

    elif source == "wandb":
        logger.info(f"Loading config from wandb: {WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")
        cfg = load_run_config_from_wandb(WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_ID)

        if WANDB_CHECKPOINT_PATH is None:
            raise ValueError(
                "For wandb source, you must provide WANDB_CHECKPOINT_PATH "
                "(download checkpoint manually or use download_wandb_checkpoint)"
            )
        logger.info(f"Loading model from: {WANDB_CHECKPOINT_PATH}")
        bornmachine = BornMachine.load(WANDB_CHECKPOINT_PATH)

    else:
        raise ValueError(f"Unknown data source: {source}")

    # Reconstruct DataHandler
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bornmachine)

    bornmachine.to(device)

    return bornmachine, datahandler, device, cfg


# %%
print("=" * 60)
print("Loading model and data...")
print("=" * 60)

bornmachine, datahandler, device, cfg = load_model_and_data()

print(f"\nDataset: {cfg.dataset.name}")
print(f"Train samples: {len(datahandler.data['train'])}")
print(f"Test samples: {len(datahandler.data['test'])}")
print(f"Data dimensions: {datahandler.data_dim}")
print(f"Number of classes: {datahandler.num_cls}")

# %% [markdown]
# ## Run UQ Evaluation

# %%
from analysis.utils import UQEvaluation, UQConfig

uq_config = UQConfig(
    norm=PURIFICATION_NORM,
    num_steps=PURIFICATION_NUM_STEPS,
    step_size=PURIFICATION_STEP_SIZE,
    radii=PURIFICATION_RADII,
    eps=PURIFICATION_EPS,
    random_start=PURIFICATION_RANDOM_START,
    percentiles=THRESHOLD_PERCENTILES,
    attack_method=ATTACK_METHOD,
    attack_strengths=ATTACK_STRENGTHS,
    attack_num_steps=ATTACK_NUM_STEPS,
)

uq_eval = UQEvaluation(config=uq_config)

# Get data loaders
datahandler.get_classification_loaders()
test_loader = datahandler.classification["test"]

# %%
print("\n" + "=" * 60)
print("Running UQ Evaluation...")
print("=" * 60)

results = uq_eval.evaluate(
    born=bornmachine,
    clean_loader=test_loader,
    device=device,
)

# %%
print("\n" + results.summary())

# %% [markdown]
# ## Visualize Results

# %%
# Setup output directory
output_dir = Path(OUTPUT_DIR)
if not output_dir.is_absolute():
    output_dir = project_root / output_dir
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# %%
# --- 1. Log-Likelihood Histogram: Clean vs Adversarial ---
print("\n--- Plotting Log-Likelihood Histogram ---")

fig, ax = plt.subplots(figsize=(10, 6))

# Clean distribution
ax.hist(results.clean_log_px, bins=50, alpha=0.6, density=True,
        label=f'Clean (n={len(results.clean_log_px)})', color='#2ecc71')

# Adversarial distributions
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(ATTACK_STRENGTHS)))
for i, eps in enumerate(sorted(results.adv_log_px.keys())):
    ax.hist(results.adv_log_px[eps], bins=50, alpha=0.5, density=True,
            label=f'Adv eps={eps} (n={len(results.adv_log_px[eps])})',
            color=colors[i])

# Threshold lines
for pct, tau in sorted(results.thresholds.items()):
    ax.axvline(tau, linestyle='--', alpha=0.7,
               label=f'tau={pct}th pct ({tau:.2f})')

ax.set_xlabel("log p(x)")
ax.set_ylabel("Density")
ax.set_title("Marginal Log-Likelihood Distribution: Clean vs Adversarial")
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
save_path = output_dir / "log_likelihood_histogram.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %%
# --- 2. Detection Rate vs Threshold ---
print("\n--- Plotting Detection Rate vs Threshold ---")

fig, ax = plt.subplots(figsize=(10, 6))

for eps in sorted(set(k[1] for k in results.detection_rates.keys())):
    pcts = sorted(set(k[0] for k in results.detection_rates.keys()))
    rates = [results.detection_rates[(p, eps)] for p in pcts]
    taus = [results.thresholds[p] for p in pcts]
    ax.plot(taus, rates, 'o-', label=f'eps={eps}', linewidth=2, markersize=8)

ax.set_xlabel("Threshold tau (log p(x))")
ax.set_ylabel("Detection Rate")
ax.set_title("Adversarial Detection Rate vs Threshold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
save_path = output_dir / "detection_rate_vs_threshold.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %%
# --- 3. Purification Accuracy Heatmap ---
print("\n--- Plotting Purification Accuracy Heatmap ---")

epsilons = sorted(set(k[0] for k in results.purification_results.keys()))
radii = sorted(set(k[1] for k in results.purification_results.keys()))

acc_matrix = np.zeros((len(epsilons), len(radii)))
for i, eps in enumerate(epsilons):
    for j, r in enumerate(radii):
        key = (eps, r)
        if key in results.purification_results:
            acc_matrix[i, j] = results.purification_results[key].accuracy_after_purify

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(radii)))
ax.set_xticklabels([f'{r}' for r in radii])
ax.set_yticks(range(len(epsilons)))
ax.set_yticklabels([f'{e}' for e in epsilons])
ax.set_xlabel("Purification Radius")
ax.set_ylabel("Attack Epsilon")
ax.set_title("Classification Accuracy After Purification")

# Add text annotations
for i in range(len(epsilons)):
    for j in range(len(radii)):
        text = f'{acc_matrix[i, j]:.2f}'
        color = 'white' if acc_matrix[i, j] < 0.5 else 'black'
        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

plt.colorbar(im, ax=ax, label='Accuracy')
plt.tight_layout()
save_path = output_dir / "purification_accuracy_heatmap.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %%
# --- 4. Before/After Purification Log p(x) Comparison ---
print("\n--- Plotting Before/After Purification ---")

# Use the middle radius for this plot
mid_radius = radii[len(radii) // 2]

fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 5),
                          squeeze=False)
axes = axes.flatten()

for i, eps in enumerate(epsilons):
    ax = axes[i]
    key = (eps, mid_radius)
    if key in results.purification_results:
        metrics = results.purification_results[key]

        # Bar comparison
        categories = ['Before', 'After']
        values = [metrics.mean_log_px_before, metrics.mean_log_px_after]
        colors = ['#e74c3c', '#2ecc71']
        bars = ax.bar(categories, values, color=colors, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        # Threshold line
        median_pct = THRESHOLD_PERCENTILES[len(THRESHOLD_PERCENTILES) // 2]
        tau = results.thresholds[median_pct]
        ax.axhline(tau, color='orange', linestyle='--', linewidth=1.5,
                   label=f'tau ({median_pct}th pct)')

        ax.set_title(f'eps={eps}, radius={mid_radius}\n'
                     f'acc: {metrics.accuracy_after_purify:.2f}, '
                     f'recovery: {metrics.recovery_rate:.0%}')
        ax.set_ylabel("Mean log p(x)")
        ax.legend(fontsize=8)

plt.suptitle(f"Purification Effect on Mean Log p(x) (radius={mid_radius})",
             fontsize=14, y=1.02)
plt.tight_layout()
save_path = output_dir / "purification_before_after.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %% [markdown]
# ## Export Summary

# %%
summary_path = output_dir / "uq_summary.txt"

summary_text = [
    "=" * 60,
    "Uncertainty Quantification (UQ) Evaluation Summary",
    "=" * 60,
    "",
    f"Run Directory: {RUN_DIR}",
    f"Dataset: {cfg.dataset.name}",
    f"Train Samples: {len(datahandler.data['train'])}",
    f"Test Samples: {len(datahandler.data['test'])}",
    "",
    f"Purification: norm={PURIFICATION_NORM}, steps={PURIFICATION_NUM_STEPS}",
    f"Attack: method={ATTACK_METHOD}, steps={ATTACK_NUM_STEPS}",
    "",
    results.summary(),
    "",
    f"Output files saved to: {output_dir}",
]

with open(summary_path, "w") as f:
    f.write("\n".join(summary_text))

print(f"\nSaved summary to: {summary_path}")

# %%
# Final summary
print("\n" + "=" * 60)
print("UQ Evaluation Complete")
print("=" * 60)
print(f"\nClean Accuracy: {results.clean_accuracy:.4f}")
print(f"Clean log p(x) mean: {results.clean_log_px.mean():.2f}")
for eps in sorted(results.adv_accuracies.keys()):
    print(f"Adv acc (eps={eps}): {results.adv_accuracies[eps]:.4f}")
print(f"\nOutput files:")
print(f"  - {output_dir / 'log_likelihood_histogram.png'}")
print(f"  - {output_dir / 'detection_rate_vs_threshold.png'}")
print(f"  - {output_dir / 'purification_accuracy_heatmap.png'}")
print(f"  - {output_dir / 'purification_before_after.png'}")
print(f"  - {output_dir / 'uq_summary.txt'}")
