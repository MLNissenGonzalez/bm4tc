# %% [markdown]
# # Visualize Configured 2D Toy Datasets
#
# Reads every YAML config in `configs/dataset/2Dtoy/`, generates the datasets
# using the exact parameters configured there, and plots them in a grid.
#
# Sorted by type (moons → circles → spirals) then by sample size, so each
# row naturally corresponds to one dataset family.
#
# **Usage:**
# ```bash
# python -m analysis.visualize_datasets
# ```
#
# Saves output to `analysis/outputs/datasets/configured_datasets.png`.

# %%
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from src.tracking.visualisation import create_2d_scatter

# %%
# =============================================================================
# Config loading
# =============================================================================

CONFIG_DIR = project_root / "configs" / "dataset" / "2Dtoy"

_TYPE_ORDER = {"moons": 0, "circles": 1, "spirals": 2}


@dataclass
class DatasetSpec:
    name: str
    size: int
    noise: float
    seed: int = 25
    circ_factor: Optional[float] = None
    split: tuple = field(default_factory=lambda: (0.5, 0.25, 0.25))


def _dataset_type(name: str) -> str:
    for t in ("moons", "circles", "spirals"):
        if t in name.lower():
            return t
    return "other"


def _make_label(spec: DatasetSpec) -> str:
    parts = f"n={2 * spec.size}  noise={spec.noise}"
    if spec.circ_factor is not None:
        parts += f"  cf={spec.circ_factor}"
    return f"{spec.name}\n{parts}"


def load_specs(config_dir: Path) -> list[DatasetSpec]:
    """Load DatasetSpecs from all YAML configs in config_dir."""
    specs = []
    for path in sorted(config_dir.glob("*.yaml")):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        g = cfg["gen_dow_kwargs"]
        specs.append(DatasetSpec(
            name=g["name"],
            size=g["size"],
            noise=g["noise"],
            seed=g.get("seed", 25),
            circ_factor=g.get("circ_factor"),
            split=tuple(cfg.get("split", [0.5, 0.25, 0.25])),
        ))
    specs.sort(key=lambda s: (_TYPE_ORDER.get(_dataset_type(s.name), 99), s.size, s.name))
    return specs


# %%
# =============================================================================
# Generation and plotting
# =============================================================================


def generate_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D toy dataset from a DatasetSpec.

    Mirrors the logic in src.data.gen_n_load._two_dim_generator.
    """
    name = spec.name.lower()
    if "moons" in name:
        X, t = sklearn.datasets.make_moons(
            n_samples=2 * spec.size, noise=spec.noise, random_state=spec.seed)
    elif "circles" in name:
        X, t = sklearn.datasets.make_circles(
            n_samples=2 * spec.size, noise=spec.noise, random_state=spec.seed,
            factor=spec.circ_factor)
    elif "spirals" in name:
        rng = np.random.RandomState(spec.seed)
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
    n_cols: int = 5,
    suptitle: str = "",
    save_path: Optional[str] = None,
):
    """Plot a grid of 2D scatter plots for the given dataset specs."""
    n = len(specs)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, spec in enumerate(specs):
        row, col = divmod(i, n_cols)
        X, t = generate_dataset(spec)
        create_2d_scatter(X=X, t=t, title=_make_label(spec), ax=axes[row, col], show_legend=False)

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

    specs = load_specs(CONFIG_DIR)
    print(f"Found {len(specs)} dataset configs in {CONFIG_DIR.name}/:")
    for s in specs:
        cf = f"  circ_factor={s.circ_factor}" if s.circ_factor is not None else ""
        print(f"  {s.name}: size={s.size} (n_total={2*s.size}), noise={s.noise}{cf}")

    # n_cols = max variants per dataset type → one row per type
    type_counts = Counter(_dataset_type(s.name) for s in specs)
    n_cols = max(type_counts.values())

    plot_dataset_grid(
        specs,
        n_cols=n_cols,
        suptitle=f"Configured 2D Toy Datasets — {CONFIG_DIR.name}/",
        save_path=out_dir / "configured_datasets.png",
    )

    print("\nDone! Check analysis/outputs/datasets/configured_datasets.png")
    plt.show()
