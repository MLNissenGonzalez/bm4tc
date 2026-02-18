# %% [markdown]
# # Visualize 2D Toy Datasets
#
# Generates and visualizes 2D toy datasets (moons, circles, spirals)
# at various sizes and noise levels in a grid layout.
#
# **Usage:**
# ```bash
# python -m analysis.visualize_datasets
# ```
#
# Saves output to `analysis/outputs/datasets/`.

# %%
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Handle both script and interactive execution
if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from src.tracking.visualisation import create_2d_scatter

# %%
# =============================================================================
# Dataset specifications
# =============================================================================

@dataclass
class DatasetSpec:
    """Specification for a single dataset to visualize.

    Parameters
    ----------
    name : str
        Dataset name (must contain 'moons', 'circles', or 'spirals').
    size : int
        Number of samples per class.
    noise : float
        Noise level for data generation.
    circ_factor : float or None
        Factor for circles dataset (ignored for others).
    label : str or None
        Display label. If None, auto-generated from fields.
    """
    name: str
    size: int
    noise: float
    circ_factor: Optional[float] = None
    label: Optional[str] = None


def _make_label(spec: DatasetSpec) -> str:
    """Generate a display label for a dataset spec."""
    if spec.label is not None:
        return spec.label
    parts = [spec.name, f"n={2*spec.size}", f"noise={spec.noise}"]
    if spec.circ_factor is not None:
        parts.append(f"cf={spec.circ_factor}")
    return ", ".join(parts)


# Existing variants
EXISTING = [
    # Moons
    DatasetSpec("moons_4k", 4000, 0.05, label="moons_4k (existing)"),
    DatasetSpec("moons_2k", 2000, 0.1, label="moons_2k (existing)"),
    DatasetSpec("moons_sparse", 500, 0.05, label="moons_sparse (existing)"),
    # Circles
    DatasetSpec("circles_4k", 4000, 0.05, circ_factor=0.3, label="circles_4k (existing)"),
    DatasetSpec("circles_2k", 2000, 0.1, circ_factor=0.4, label="circles_2k (existing)"),
    DatasetSpec("circles_sparse", 500, 0.05, circ_factor=0.3, label="circles_sparse (existing)"),
    # Spirals
    DatasetSpec("spirals_4k", 4000, 0.5, label="spirals_4k (existing)"),
    DatasetSpec("spirals_2k", 2000, 0.5, label="spirals_2k (existing)"),
    DatasetSpec("spirals_sparse", 500, 0.5, label="spirals_sparse (existing)"),
]

# Candidate hard variants (500 samples per class, higher noise)
HARD_CANDIDATES = [
    # Moons candidates
    DatasetSpec("moons_hard_15", 500, 0.15, label="moons hard noise=0.15"),
    DatasetSpec("moons_hard_20", 500, 0.20, label="moons hard noise=0.20"),
    DatasetSpec("moons_hard_25", 500, 0.25, label="moons hard noise=0.25"),
    DatasetSpec("moons_hard_30", 500, 0.30, label="moons hard noise=0.30"),
    # Circles candidates
    DatasetSpec("circles_hard_15", 500, 0.15, circ_factor=0.5, label="circles hard n=0.15 cf=0.5"),
    DatasetSpec("circles_hard_20", 500, 0.20, circ_factor=0.5, label="circles hard n=0.20 cf=0.5"),
    DatasetSpec("circles_hard_25", 500, 0.25, circ_factor=0.5, label="circles hard n=0.25 cf=0.5"),
    DatasetSpec("circles_hard_15b", 500, 0.15, circ_factor=0.6, label="circles hard n=0.15 cf=0.6"),
    DatasetSpec("circles_hard_20b", 500, 0.20, circ_factor=0.6, label="circles hard n=0.20 cf=0.6"),
    # Spirals candidates
    DatasetSpec("spirals_hard_08", 500, 0.8, label="spirals hard noise=0.8"),
    DatasetSpec("spirals_hard_10", 500, 1.0, label="spirals hard noise=1.0"),
    DatasetSpec("spirals_hard_12", 500, 1.2, label="spirals hard noise=1.2"),
    DatasetSpec("spirals_hard_15", 500, 1.5, label="spirals hard noise=1.5"),
]

SEED = 25

# Final hard variants (chosen from candidate exploration)
HARD_FINAL = [
    DatasetSpec("moons_hard", 500, 0.25, label="moons_hard (noise=0.25)"),
    DatasetSpec("circles_hard", 500, 0.20, circ_factor=0.5, label="circles_hard (n=0.20, cf=0.5)"),
    DatasetSpec("spirals_hard", 500, 1.0, label="spirals_hard (noise=1.0)"),
]

# %%
# =============================================================================
# Generation and plotting
# =============================================================================


def generate_dataset(spec: DatasetSpec, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D toy dataset from a DatasetSpec.

    Mirrors the logic in src.data.gen_n_load._two_dim_generator to avoid
    circular import through src.data.__init__.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset specification.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (2*size, 2)
        Feature matrix.
    t : np.ndarray, shape (2*size,)
        Labels.
    """
    name = spec.name.lower()
    if "moons" in name:
        X, t = sklearn.datasets.make_moons(
            n_samples=2 * spec.size, noise=spec.noise, random_state=seed)
    elif "circles" in name:
        X, t = sklearn.datasets.make_circles(
            n_samples=2 * spec.size, noise=spec.noise, random_state=seed,
            factor=spec.circ_factor)
    elif "spirals" in name:
        rng = np.random.RandomState(seed)
        theta = np.sqrt(rng.rand(spec.size)) * 2 * np.pi
        r_1 = 2 * theta + np.pi
        data_1 = np.array([np.cos(theta) * r_1, np.sin(theta) * r_1]).T
        x_1 = data_1 + spec.noise * rng.randn(spec.size, 2)
        r_2 = -2 * theta - np.pi
        data_2 = np.array([np.cos(theta) * r_2, np.sin(theta) * r_2]).T
        x_2 = data_2 + spec.noise * rng.randn(spec.size, 2)
        res_1 = np.append(x_1, np.zeros((spec.size, 1)), axis=1)
        res_2 = np.append(x_2, np.ones((spec.size, 1)), axis=1)
        res = np.append(res_1, res_2, axis=0)
        rng.shuffle(res)
        X, t = res[:, :-1], res[:, -1]
    else:
        raise ValueError(f"Unknown dataset type in name: {spec.name}")
    return X, t


def plot_dataset_grid(
    specs: list[DatasetSpec],
    n_cols: int = 4,
    seed: int = SEED,
    suptitle: str = "",
    save_path: Optional[str] = None,
):
    """Plot a grid of 2D scatter plots for the given dataset specs.

    Parameters
    ----------
    specs : list of DatasetSpec
        Datasets to visualize.
    n_cols : int
        Number of columns in the grid.
    seed : int
        Random seed.
    suptitle : str
        Figure super-title.
    save_path : str or None
        If given, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(specs)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, spec in enumerate(specs):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        X, t = generate_dataset(spec, seed=seed)
        label = _make_label(spec)
        create_2d_scatter(X=X, t=t, title=label, ax=ax, show_legend=False)

    # Hide unused axes
    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# %%
# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    out_dir = project_root / "analysis" / "outputs" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating existing dataset variants...")
    print("=" * 60)
    plot_dataset_grid(
        EXISTING, n_cols=3,
        suptitle="Existing 2D Toy Datasets",
        save_path=out_dir / "existing_datasets.png",
    )

    print("\n" + "=" * 60)
    print("Generating hard candidate variants...")
    print("=" * 60)
    plot_dataset_grid(
        HARD_CANDIDATES, n_cols=4,
        suptitle="Hard Candidate Variants (500 spc, higher noise)",
        save_path=out_dir / "hard_candidates.png",
    )

    print("\n" + "=" * 60)
    print("Generating final hard variants...")
    print("=" * 60)
    plot_dataset_grid(
        HARD_FINAL, n_cols=3,
        suptitle="Final Hard Variants (500 spc)",
        save_path=out_dir / "hard_final.png",
    )

    print("\nDone! Check analysis/outputs/datasets/")
    plt.show()
