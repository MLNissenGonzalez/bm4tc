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


def _load_sampler(run_dir: Path, device):
    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    model, extra = JEMMLP.load(run_dir / "models/model.pt", device=device)
    sgld_cfg = SGLDConfig(**OmegaConf.to_container(cfg.sampler, resolve=True))
    buffer = ReplayBuffer(
        sgld_cfg.buffer_size, model.data_dim, model.input_range, int(cfg.tracking.seed)
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


def sample_and_plot(
    run_dir: str | Path,
    *,
    n_per_class: int = 64,
    steps: int = 1000,
    save_dir: str | Path | None = None,
    device: str = "auto",
):
    """Plot the per-class mean in the same 2×5 layout as the MPS notebook."""
    samples, (lo, hi) = sample_all_classes(
        run_dir,
        n_per_class=n_per_class,
        steps=steps,
        device=device,
    )
    image_dim = math.isqrt(samples.shape[-1])
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for class_idx in range(min(len(samples), 10)):
        ax = axes[class_idx // 5, class_idx % 5]
        normalized = (samples[class_idx].float() - lo) / (hi - lo)
        mean_image = normalized.mean(0).reshape(image_dim, image_dim)
        ax.imshow(mean_image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {class_idx}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Mean sampled MNIST digit per class", fontsize=11)
    fig.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "mnist_samples.png", dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mean-grid", action="store_true")
    args = parser.parse_args()

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

    samples, input_range = sample_all_classes(
        args.run_dir,
        n_per_class=args.per_class,
        steps=args.steps,
        device=args.device,
    )
    image_dim = math.isqrt(samples.shape[-1])
    samples = samples.view(len(samples), args.per_class, image_dim, image_dim)
    fig, axes = plt.subplots(
        len(samples),
        args.per_class,
        figsize=(args.per_class, 10),
        squeeze=False,
    )
    for c in range(len(samples)):
        for j in range(args.per_class):
            axes[c, j].imshow(
                samples[c, j],
                cmap="gray",
                vmin=input_range[0],
                vmax=input_range[1],
            )
            axes[c, j].axis("off")
    fig.tight_layout(pad=0.1)
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "jem_samples.png"
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
