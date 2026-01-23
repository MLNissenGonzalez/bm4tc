# %% [markdown]
# # Membership Inference Attack (MIA) Analysis
#
# This notebook evaluates privacy leakage via membership inference attacks.
#
# **What it does:**
# - Loads a trained BornMachine model and its original dataset
# - Extracts features from model outputs (class probabilities) for train and test sets
# - Trains an attack classifier to distinguish training samples from test samples
# - Reports attack success metrics (accuracy, AUC-ROC) as privacy indicators
#
# **Privacy Interpretation:**
# - AUC-ROC near 0.5 = good privacy (attacker can't distinguish train from test)
# - AUC-ROC >> 0.5 = privacy leakage (model memorizes training data)
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
# Path to run directory (contains .hydra/config.yaml and models/)
RUN_DIR = "outputs/classification_example"  # Change to your run directory

# --- WANDB SETTINGS (used if DATA_SOURCE == "wandb") ---
WANDB_ENTITY = "your-entity"
WANDB_PROJECT = "gan_train"
WANDB_RUN_ID = "abc123"  # Specific run ID
# For wandb, you need to download the checkpoint separately or provide a local path
WANDB_CHECKPOINT_PATH = None  # Set to local path if checkpoint downloaded

# --- MIA SETTINGS ---
# Features to use for the attack (all enabled by default)
MIA_FEATURES = {
    "max_prob": True,       # Maximum class probability
    "entropy": True,        # Prediction entropy
    "correct_prob": True,   # Probability of correct class
    "loss": True,           # Cross-entropy loss
    "margin": True,         # Difference between top two probabilities
    "modified_entropy": True,  # Normalized confidence
}

# Attack classifier
ATTACK_MODEL = "logistic"  # Only "logistic" supported currently
TEST_SPLIT = 0.3  # Fraction of attack data for evaluation
RANDOM_STATE = 42

# --- OUTPUT SETTINGS ---
OUTPUT_DIR = "analysis/outputs/mia"

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
        Tuple of (bornmachine, datahandler, device)
    """
    from analysis.utils import (
        load_run_config,
        load_run_config_from_wandb,
        find_model_checkpoint,
    )
    from src.data import DataHandler
    from src.models import BornMachine

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    if source == "local":
        run_dir = Path(RUN_DIR)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir

        logger.info(f"Loading config from: {run_dir}")
        cfg = load_run_config(run_dir)

        # Find and load checkpoint
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

    # Move model to device
    bornmachine.to(device)

    return bornmachine, datahandler, device, cfg


# %%
# Load everything
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
# ## Run MIA Evaluation

# %%
from analysis.utils import MIAEvaluation, MIAFeatureConfig

# Create feature config from settings
feature_config = MIAFeatureConfig(**MIA_FEATURES)
print(f"\nEnabled MIA features: {feature_config.enabled_features()}")

# Create evaluation object
mia_eval = MIAEvaluation(
    feature_config=feature_config,
    attack_model=ATTACK_MODEL,
    test_split=TEST_SPLIT,
    random_state=RANDOM_STATE,
)

# Get data loaders
datahandler.get_classification_loaders()
train_loader = datahandler.classification["train"]
test_loader = datahandler.classification["test"]

# %%
print("\n" + "=" * 60)
print("Running MIA Evaluation...")
print("=" * 60)

results = mia_eval.evaluate(
    model=bornmachine,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
)

# %%
# Print summary
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
# --- Feature Importance Bar Chart ---
print("\n--- Plotting Feature Importance ---")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by absolute importance
sorted_features = sorted(
    results.feature_importance.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)
names = [name for name, _ in sorted_features]
values = [val for _, val in sorted_features]
colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

bars = ax.barh(names, values, color=colors)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel("Logistic Regression Coefficient")
ax.set_ylabel("Feature")
ax.set_title("MIA Attack Feature Importance\n(Positive = higher value indicates training sample)")
ax.invert_yaxis()

# Add value labels
for bar, val in zip(bars, values):
    x_pos = val + 0.01 * max(abs(v) for v in values) * (1 if val >= 0 else -1)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=9)

plt.tight_layout()
save_path = output_dir / "feature_importance.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %%
# --- Per-Feature Threshold Attacks (AUC-ROC) ---
print("\n--- Plotting Threshold Attacks ---")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by AUC-ROC
sorted_thresh = sorted(
    results.threshold_metrics.items(),
    key=lambda x: x[1]["auc_roc"],
    reverse=True
)
names = [name for name, _ in sorted_thresh]
aucs = [metrics["auc_roc"] for _, metrics in sorted_thresh]

# Color by significance
colors = []
for auc in aucs:
    if auc >= 0.70:
        colors.append('#e74c3c')  # Red - significant leakage
    elif auc >= 0.60:
        colors.append('#f39c12')  # Orange - moderate leakage
    elif auc >= 0.55:
        colors.append('#f1c40f')  # Yellow - small leakage
    else:
        colors.append('#2ecc71')  # Green - good privacy

bars = ax.barh(names, aucs, color=colors)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Random (AUC=0.5)')
ax.axvline(x=0.55, color='#f1c40f', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=0.60, color='#f39c12', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=0.70, color='#e74c3c', linestyle=':', linewidth=1, alpha=0.7)
ax.set_xlabel("AUC-ROC")
ax.set_ylabel("Feature")
ax.set_title("Per-Feature Threshold Attack Performance")
ax.set_xlim(0.4, 1.0)
ax.invert_yaxis()

# Add value labels
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2, f'{auc:.3f}',
            va='center', ha='left', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='< 0.55: Excellent'),
    Patch(facecolor='#f1c40f', label='0.55-0.60: Good'),
    Patch(facecolor='#f39c12', label='0.60-0.70: Moderate'),
    Patch(facecolor='#e74c3c', label='>= 0.70: Significant'),
]
ax.legend(handles=legend_elements, loc='lower right', title='Privacy Level')

plt.tight_layout()
save_path = output_dir / "threshold_attacks.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %%
# --- Feature Distributions (Train vs Test) ---
print("\n--- Plotting Feature Distributions ---")

feature_names = results.feature_names
n_features = len(feature_names)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, name in enumerate(feature_names):
    ax = axes[idx]
    train_vals = results.train_features[:, idx]
    test_vals = results.test_features[:, idx]

    # Plot histograms
    bins = 50
    ax.hist(train_vals, bins=bins, alpha=0.6, density=True,
            label=f'Train (n={len(train_vals)})', color='#3498db')
    ax.hist(test_vals, bins=bins, alpha=0.6, density=True,
            label=f'Test (n={len(test_vals)})', color='#e74c3c')

    # Add statistics
    train_mean = train_vals.mean()
    test_mean = test_vals.mean()
    ax.axvline(train_mean, color='#3498db', linestyle='--', linewidth=1.5)
    ax.axvline(test_mean, color='#e74c3c', linestyle='--', linewidth=1.5)

    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.set_title(f"{name}\n(Train mean: {train_mean:.3f}, Test mean: {test_mean:.3f})")
    ax.legend(fontsize=8)

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Feature Distributions: Train vs Test Samples", fontsize=14, y=1.02)
plt.tight_layout()
save_path = output_dir / "feature_distributions.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

# %% [markdown]
# ## Export Summary

# %%
# Save text summary
summary_path = output_dir / "mia_summary.txt"

summary_text = [
    "=" * 60,
    "Membership Inference Attack (MIA) Evaluation Summary",
    "=" * 60,
    "",
    f"Run Directory: {RUN_DIR}",
    f"Dataset: {cfg.dataset.name}",
    f"Train Samples: {len(datahandler.data['train'])}",
    f"Test Samples: {len(datahandler.data['test'])}",
    "",
    results.summary(),
    "",
    "Privacy Interpretation:",
    "  - AUC-ROC < 0.55: Excellent privacy preservation",
    "  - AUC-ROC 0.55-0.60: Good privacy",
    "  - AUC-ROC 0.60-0.70: Moderate leakage",
    "  - AUC-ROC >= 0.70: Significant leakage",
    "",
    f"Output files saved to: {output_dir}",
]

with open(summary_path, "w") as f:
    f.write("\n".join(summary_text))

print(f"\nSaved summary to: {summary_path}")

# %%
# Final summary
print("\n" + "=" * 60)
print("MIA Evaluation Complete")
print("=" * 60)
print(f"\nPrivacy Assessment: {results.privacy_assessment()}")
print(f"Attack AUC-ROC: {results.auc_roc:.4f}")
print(f"Attack Accuracy: {results.attack_accuracy:.4f}")
print(f"\nOutput files:")
print(f"  - {output_dir / 'feature_importance.png'}")
print(f"  - {output_dir / 'threshold_attacks.png'}")
print(f"  - {output_dir / 'feature_distributions.png'}")
print(f"  - {output_dir / 'mia_summary.txt'}")
