"""Generate a class-conditional sample grid from a trained JEM."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from .device import resolve_device
from .model import JEMMLP
from .sampler import ReplayBuffer, SGLDConfig, SGLDSampler


def _sampler_config(run_dir: Path, extra: dict) -> SGLDConfig:
    """Recover SGLD settings without requiring Hydra run metadata."""
    if "sampler_config" in extra:
        return SGLDConfig(**extra["sampler_config"])

    hydra_config = run_dir / ".hydra/config.yaml"
    if hydra_config.exists():
        cfg = OmegaConf.load(hydra_config)
        return SGLDConfig(**OmegaConf.to_container(cfg.sampler, resolve=True))

    config_root = Path(__file__).parent / "configs"
    values = OmegaConf.to_container(
        OmegaConf.load(config_root / "sampler/default.yaml"),
        resolve=True,
    )
    sweep_name = run_dir.parent.name.rsplit("_", 1)[0]
    sweep_config = config_root / "experiment/seed_sweep" / f"{sweep_name}.yaml"
    if sweep_config.exists():
        cfg = OmegaConf.load(sweep_config)
        params = OmegaConf.select(cfg, "hydra.sweeper.params", default={})
        for field_name in SGLDConfig.__dataclass_fields__:
            key = f"sampler.{field_name}"
            if key in params:
                values[field_name] = params[key]
    return SGLDConfig(**values)


def _image_dim(samples: torch.Tensor) -> int:
    image_dim = math.isqrt(samples.shape[-1])
    if image_dim * image_dim != samples.shape[-1]:
        raise ValueError(
            f"Expected square images, but samples have {samples.shape[-1]} pixels."
        )
    return image_dim


def _load_sampler(run_dir: Path, device):
    model, extra = JEMMLP.load(run_dir / "models/model.pt", device=device)
    sgld_cfg = _sampler_config(run_dir, extra)
    seed = int(run_dir.name) + 1 if run_dir.name.isdigit() else 0
    buffer = ReplayBuffer(
        sgld_cfg.buffer_size,
        model.data_dim,
        model.input_range,
        seed,
    )
    if "replay_buffer" in extra:
        buffer.load_state_dict(extra["replay_buffer"])
    return model, SGLDSampler(sgld_cfg, buffer)


def sample_all_classes(
    run_dir: str | Path,
    *,
    n_per_class: int = 64,
    steps: int = 1000,
    device: str = "auto",
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Return class-conditional JEM samples with shape (classes, samples, pixels)."""
    run_dir = Path(run_dir)
    resolved_device = resolve_device(device)
    model, sampler = _load_sampler(run_dir, resolved_device)
    rows = []
    for class_idx in range(model.out_dim):
        rows.append(
            sampler.sample_fresh(
                model,
                n_per_class,
                resolved_device,
                class_idx=class_idx,
                num_steps=steps,
            )
        )
    return torch.stack(rows).cpu(), model.input_range


def plot_sample_grid(
    samples: torch.Tensor,
    input_range: tuple[float, float],
    *,
    show_per_class: int = 8,
):
    """Plot the first samples from each class, with one labelled row per class."""
    if show_per_class < 1:
        raise ValueError("show_per_class must be at least 1.")
    if show_per_class > samples.shape[1]:
        raise ValueError(
            f"Cannot show {show_per_class} samples per class; only "
            f"{samples.shape[1]} were generated."
        )

    image_dim = _image_dim(samples)
    shown = samples[:, :show_per_class].reshape(
        len(samples), show_per_class, image_dim, image_dim
    )
    fig, axes = plt.subplots(
        len(shown),
        show_per_class,
        figsize=(show_per_class, len(shown)),
        squeeze=False,
    )
    for class_idx in range(len(shown)):
        for sample_idx in range(show_per_class):
            ax = axes[class_idx, sample_idx]
            ax.imshow(
                shown[class_idx, sample_idx],
                cmap="gray",
                vmin=input_range[0],
                vmax=input_range[1],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[class_idx, 0].set_ylabel(
            f"Class {class_idx}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
        )
    fig.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    return fig


def plot_mean_grid(
    samples: torch.Tensor,
    input_range: tuple[float, float],
):
    """Plot the pixel-wise mean of every class in a compact 2×5 grid."""
    if len(samples) > 10:
        raise ValueError("The 2x5 mean grid supports at most 10 classes.")

    lo, hi = input_range
    image_dim = _image_dim(samples)
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(7, 3),
        squeeze=False,
        gridspec_kw={"wspace": 0.02, "hspace": 0.16},
    )
    flat_axes = axes.flat
    for class_idx, ax in enumerate(flat_axes):
        if class_idx < len(samples):
            normalized = (samples[class_idx].float() - lo) / (hi - lo)
            mean_image = normalized.mean(0).reshape(image_dim, image_dim)
            ax.imshow(mean_image, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Class {class_idx}", fontsize=9, pad=2)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
    return fig


def sample_and_plot(
    run_dir: str | Path,
    *,
    n_per_class: int = 64,
    steps: int = 1000,
    save_dir: str | Path | None = None,
    device: str = "auto",
):
    """Plot the per-class mean in the same 2×5 layout as the MPS notebook."""
    samples, input_range = sample_all_classes(
        run_dir,
        n_per_class=n_per_class,
        steps=steps,
        device=device,
    )
    fig = plot_mean_grid(samples, input_range)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "mnist_samples.png", dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--per-class",
        type=int,
        default=64,
        help="number of samples to generate per class (default: 64)",
    )
    parser.add_argument(
        "--show-per-class",
        type=int,
        default=8,
        help="number of generated samples to show per class (default: 8)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--mean-grid",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class must be at least 1")

    if args.mean_grid:
        output = Path(args.output) if args.output else Path(args.run_dir)
        save_dir = None if output.suffix else output
        fig = sample_and_plot(
            args.run_dir,
            n_per_class=args.per_class,
            steps=args.steps,
            save_dir=save_dir,
            device=args.device,
        )
        if output.suffix:
            fig.savefig(output, dpi=150, bbox_inches="tight")
            target = output
        else:
            target = output / "mnist_samples.png"
        print(f"Saved {target}")
        return

    if args.show_per_class < 1:
        parser.error("--show-per-class must be at least 1")
    if args.show_per_class > args.per_class:
        parser.error("--show-per-class cannot exceed --per-class")

    samples, input_range = sample_all_classes(
        args.run_dir,
        n_per_class=args.per_class,
        steps=args.steps,
        device=args.device,
    )

    run_dir = Path(args.run_dir)
    if args.output:
        requested_output = Path(args.output)
        if requested_output.is_dir() or not requested_output.suffix:
            samples_output = requested_output / "jem_samples.png"
            mean_output = requested_output / "jem_samples_mean.png"
        else:
            samples_output = requested_output
            mean_output = requested_output.with_name(
                f"{requested_output.stem}_mean{requested_output.suffix}"
            )
    else:
        samples_output = run_dir / "jem_samples.png"
        mean_output = run_dir / "jem_samples_mean.png"
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    mean_output.parent.mkdir(parents=True, exist_ok=True)

    sample_fig = plot_sample_grid(
        samples,
        input_range,
        show_per_class=args.show_per_class,
    )
    sample_fig.savefig(samples_output, dpi=150, bbox_inches="tight")
    plt.close(sample_fig)

    mean_fig = plot_mean_grid(samples, input_range)
    mean_fig.savefig(mean_output, dpi=150, bbox_inches="tight")
    plt.close(mean_fig)
    print(f"Saved {samples_output}")
    print(f"Saved {mean_output}")


if __name__ == "__main__":
    main()
