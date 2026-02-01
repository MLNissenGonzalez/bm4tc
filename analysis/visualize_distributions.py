# %% [markdown]
# # Visualize Learned Probability Distributions
#
# This notebook visualizes the learned p(c|x) and p(x,c) distributions
# of a trained BornMachine over the 2D input space.
#
# **What it shows:**
# - Row 1: p(c|x) conditional class probabilities per class + decision boundary
# - Row 2: p(x,c) joint probabilities per class + marginal p(x)
# - Optional training data overlay for verification
#
# **Usage:**
# - Set `RUN_DIR` to your Hydra output directory
# - Run cells interactively (VS Code) or as a script

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
from analysis.utils import load_run_config, find_model_checkpoint
from src.models import BornMachine  # must import before src.data to avoid circular import
from src.data.handler import DataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION FOR YOUR EXPERIMENT
# =============================================================================

# Path to run directory (contains .hydra/config.yaml and models/)
RUN_DIR = "outputs/lrwdbs_hpo_circles_4k_18Jan26/41"  # Change to your run directory

# Grid resolution for heatmaps (resolution x resolution points)
RESOLUTION = 150

# Whether to normalize p(x,c) by the partition function
NORMALIZE_JOINT = True

# Whether to overlay training data points on plots
SHOW_DATA = True

# Device for computation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory for saved figures
SAVE_DIR = "analysis/outputs/distributions/"

# %% [markdown]
# ## Utility Functions

# %%
def make_grid(input_range, resolution):
    """Create a 2D meshgrid over [lo, hi]^2.

    Args:
        input_range: Tuple (lo, hi) defining the input space bounds.
        resolution: Number of points along each axis.

    Returns:
        grid_x1: (resolution, resolution) array of x1 coordinates.
        grid_x2: (resolution, resolution) array of x2 coordinates.
        grid_points: (resolution^2, 2) tensor of all grid points.
    """
    lo, hi = input_range
    x1 = np.linspace(lo, hi, resolution)
    x2 = np.linspace(lo, hi, resolution)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid_points = torch.tensor(
        np.column_stack([grid_x1.ravel(), grid_x2.ravel()]),
        dtype=torch.float32,
    )
    return grid_x1, grid_x2, grid_points


def compute_conditional_probs(bm, grid_points, device, batch_size=10000):
    """Compute p(c|x) over grid points using batched inference.

    Args:
        bm: BornMachine instance (already on device).
        grid_points: (N, 2) tensor of input points.
        device: Torch device.
        batch_size: Number of points per batch.

    Returns:
        (N, num_classes) tensor of conditional class probabilities.
    """
    all_probs = []
    n = grid_points.shape[0]
    for start in range(0, n, batch_size):
        batch = grid_points[start:start + batch_size].to(device)
        with torch.no_grad():
            probs = bm.class_probabilities(batch)
        all_probs.append(probs.cpu())
        bm.reset()
    return torch.cat(all_probs, dim=0)


def compute_joint_probs(bm, grid_points, device, normalize=True, batch_size=10000):
    """Compute p(x,c) = |psi(x,c)|^2 [/ Z] over grid points.

    For each class c, computes the unnormalized joint probability using the
    generator's sequential contraction. Optionally normalizes by the partition
    function.

    Args:
        bm: BornMachine instance (already on device).
        grid_points: (N, 2) tensor of input points.
        device: Torch device.
        normalize: If True, divide by exp(log_partition_function()).
        batch_size: Number of points per batch.

    Returns:
        (N, num_classes) tensor of (normalized) joint probabilities.
    """
    n = grid_points.shape[0]
    num_classes = bm.out_dim

    # Compute partition function once if normalizing
    if normalize:
        bm.generator.reset()
        with torch.no_grad():
            log_Z = bm.generator.log_partition_function()
        Z = torch.exp(log_Z).item()
        logger.info(f"Partition function Z = {Z:.6f} (log Z = {log_Z:.4f})")
    else:
        Z = 1.0

    all_probs = []
    for c in range(num_classes):
        class_probs = []
        for start in range(0, n, batch_size):
            batch = grid_points[start:start + batch_size].to(device)
            labels = torch.full((batch.shape[0],), c, dtype=torch.long, device=device)
            bm.generator.reset()
            with torch.no_grad():
                prob = bm.generator.unnormalized_prob(batch, labels)
            class_probs.append(prob.cpu())
        all_probs.append(torch.cat(class_probs, dim=0))
        logger.info(f"Computed joint probabilities for class {c}")

    # Stack: (N, num_classes)
    joint = torch.stack(all_probs, dim=1)
    if normalize:
        joint = joint / Z
    return joint


def plot_distributions(
    conditional, joint, grid_x1, grid_x2,
    train_data=None, train_labels=None,
    input_range=(0.0, 1.0),
    save_path=None,
):
    """Plot conditional and joint probability distributions.

    Layout:
        Row 0: p(c|x) per class [heatmaps] + decision boundary [argmax]
        Row 1: p(x,c) per class [heatmaps] + sum_c p(x,c) [marginal]

    Args:
        conditional: (N, num_classes) tensor of p(c|x).
        joint: (N, num_classes) tensor of p(x,c).
        grid_x1: (res, res) meshgrid x1 coordinates.
        grid_x2: (res, res) meshgrid x2 coordinates.
        train_data: Optional (M, 2) tensor of training points.
        train_labels: Optional (M,) tensor of training labels.
        input_range: Tuple (lo, hi) for axis limits.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    num_classes = conditional.shape[1]
    resolution = grid_x1.shape[0]
    n_cols = num_classes + 1  # one per class + summary column

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    cond_np = conditional.numpy()
    joint_np = joint.numpy()

    lo, hi = input_range

    # --- Row 0: p(c|x) ---
    for c in range(num_classes):
        ax = axes[0, c]
        vals = cond_np[:, c].reshape(resolution, resolution)
        pcm = ax.pcolormesh(grid_x1, grid_x2, vals, cmap="viridis",
                            shading="auto", vmin=0.0, vmax=1.0)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"p(c={c}|x)")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        if train_data is not None and train_labels is not None:
            _overlay_data(ax, train_data, train_labels, num_classes)

    # Decision boundary (argmax)
    ax = axes[0, num_classes]
    decision = np.argmax(cond_np, axis=1).reshape(resolution, resolution)
    cmap_discrete = plt.colormaps.get_cmap("tab10").resampled(num_classes)
    pcm = ax.pcolormesh(grid_x1, grid_x2, decision, cmap=cmap_discrete,
                        shading="auto", vmin=-0.5, vmax=num_classes - 0.5)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, ticks=range(num_classes))
    cbar.set_label("Class")
    ax.set_title("Decision boundary\nargmax p(c|x)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    if train_data is not None and train_labels is not None:
        _overlay_data(ax, train_data, train_labels, num_classes)

    # --- Row 1: p(x,c) ---
    # Find global max for consistent color scale across class columns
    joint_max = joint_np.max()

    for c in range(num_classes):
        ax = axes[1, c]
        vals = joint_np[:, c].reshape(resolution, resolution)
        pcm = ax.pcolormesh(grid_x1, grid_x2, vals, cmap="viridis",
                            shading="auto", vmin=0.0, vmax=joint_max)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"p(x, c={c})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        if train_data is not None and train_labels is not None:
            _overlay_data(ax, train_data, train_labels, num_classes)

    # Marginal p(x) = sum_c p(x,c)
    ax = axes[1, num_classes]
    marginal = joint_np.sum(axis=1).reshape(resolution, resolution)
    pcm = ax.pcolormesh(grid_x1, grid_x2, marginal, cmap="viridis", shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(r"$p(x) = \sum_c p(x,c)$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    if train_data is not None and train_labels is not None:
        _overlay_data(ax, train_data, train_labels, num_classes)

    fig.suptitle("Learned Distributions of BornMachine", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    return fig


def _overlay_data(ax, data, labels, num_classes):
    """Overlay training data scatter points on an axis.

    Args:
        ax: Matplotlib axis.
        data: (M, 2) tensor or array of data points.
        labels: (M,) tensor or array of class labels.
        num_classes: Number of distinct classes.
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # Use contrasting markers for visibility on heatmaps
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    for c in range(num_classes):
        mask = labels == c
        marker = markers[c % len(markers)]
        ax.scatter(
            data[mask, 0], data[mask, 1],
            s=3, alpha=0.3, c="white", edgecolors="black",
            linewidths=0.3, marker=marker, zorder=5,
        )


def visualize_from_run_dir(
    run_dir,
    resolution=150,
    normalize_joint=True,
    show_data=True,
    device="cpu",
    save_dir=None,
):
    """High-level convenience: load model + data and produce distribution plots.

    Args:
        run_dir: Path to the Hydra output directory.
        resolution: Grid resolution for heatmaps.
        normalize_joint: Whether to normalize p(x,c) by the partition function.
        show_data: Whether to overlay training data points.
        device: Torch device string.
        save_dir: Directory to save figures. If None, does not save.

    Returns:
        Matplotlib Figure object.
    """
    

    device = torch.device(device)
    run_dir = Path(run_dir)

    # Load config and model
    cfg = load_run_config(run_dir)
    checkpoint_path = find_model_checkpoint(run_dir)
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(device)
    logger.info(f"Loaded model from {checkpoint_path}")

    # Load and prepare data
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)

    # Get training data for overlay
    train_data = datahandler.data["train"] if show_data else None
    train_labels = datahandler.labels["train"] if show_data else None

    # Build grid
    input_range = bm.input_range
    grid_x1, grid_x2, grid_points = make_grid(input_range, resolution)

    # Compute distributions
    logger.info("Computing conditional probabilities p(c|x)...")
    conditional = compute_conditional_probs(bm, grid_points, device)

    logger.info("Computing joint probabilities p(x,c)...")
    joint = compute_joint_probs(bm, grid_points, device, normalize=normalize_joint)

    # Determine save path
    save_path = None
    if save_dir is not None:
        save_path = Path(save_dir) / "distributions.png"

    # Plot
    fig = plot_distributions(
        conditional, joint, grid_x1, grid_x2,
        train_data=train_data, train_labels=train_labels,
        input_range=input_range,
        save_path=save_path,
    )

    return fig


# %% [markdown]
# ## Load Model and Data

# %%
def load_model_and_data():
    """Load trained BornMachine and DataHandler from run directory.

    Returns:
        Tuple of (bornmachine, datahandler, device, cfg).
    """
    from analysis.utils import load_run_config, find_model_checkpoint
    from src.data import DataHandler
    from src.models import BornMachine

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    run_dir = Path(RUN_DIR)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    logger.info(f"Loading config from: {run_dir}")
    cfg = load_run_config(run_dir)

    checkpoint_path = find_model_checkpoint(run_dir)
    logger.info(f"Loading model from: {checkpoint_path}")
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(device)

    # Reconstruct DataHandler
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)

    return bm, datahandler, device, cfg


# %%
print("=" * 60)
print("Loading model and data...")
print("=" * 60)

bm, datahandler, device, cfg = load_model_and_data()

print(f"\nDataset: {cfg.dataset.name}")
print(f"Train samples: {len(datahandler.data['train'])}")
print(f"Number of classes: {datahandler.num_cls}")
print(f"Input range: {bm.input_range}")

# %% [markdown]
# ## Compute Distributions

# %%
print("=" * 60)
print("Computing distributions over input grid...")
print("=" * 60)

input_range = bm.input_range
grid_x1, grid_x2, grid_points = make_grid(input_range, RESOLUTION)
print(f"Grid: {RESOLUTION}x{RESOLUTION} = {grid_points.shape[0]} points")

# %%
print("\nComputing p(c|x)...")
conditional = compute_conditional_probs(bm, grid_points, device)
print(f"  Shape: {conditional.shape}")
print(f"  Range: [{conditional.min():.4f}, {conditional.max():.4f}]")

# %%
print("\nComputing p(x,c)...")
joint = compute_joint_probs(bm, grid_points, device, normalize=NORMALIZE_JOINT)
print(f"  Shape: {joint.shape}")
print(f"  Range: [{joint.min():.6f}, {joint.max():.6f}]")

# %% [markdown]
# ## Plot Distributions

# %%
# Prepare training data overlay
train_data = datahandler.data["train"] if SHOW_DATA else None
train_labels = datahandler.labels["train"] if SHOW_DATA else None

# Setup output directory
save_dir = Path(SAVE_DIR)
if not save_dir.is_absolute():
    save_dir = project_root / save_dir
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "distributions.png"

print(f"\nOutput directory: {save_dir}")

# %%
fig = plot_distributions(
    conditional, joint, grid_x1, grid_x2,
    train_data=train_data, train_labels=train_labels,
    input_range=input_range,
    save_path=save_path,
)
plt.show()

# %%
print("\n" + "=" * 60)
print("Distribution Visualization Complete")
print("=" * 60)
print(f"\nSaved figure: {save_path}")


# %%
if __name__ == "__main__":
    pass  # All cells above execute when run as a script
