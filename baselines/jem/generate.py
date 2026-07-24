"""Generate a class-conditional sample grid from a trained JEM."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from .device import resolve_device
from .model import JEMMLP
from .sampler import ReplayBuffer, SGLDConfig, SGLDSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    model, extra = JEMMLP.load(run_dir / "models/model.pt", device=device)
    sgld_cfg = SGLDConfig(**OmegaConf.to_container(cfg.sampler, resolve=True))
    buffer = ReplayBuffer(
        sgld_cfg.buffer_size, model.data_dim, model.input_range, int(cfg.tracking.seed)
    )
    if "replay_buffer" in extra:
        buffer.load_state_dict(extra["replay_buffer"])
    sampler = SGLDSampler(sgld_cfg, buffer)

    rows = []
    for class_idx in range(model.out_dim):
        rows.append(
            sampler.sample_fresh(
                model,
                args.per_class,
                device,
                class_idx=class_idx,
                num_steps=args.steps,
            )
        )
    samples = torch.stack(rows).view(model.out_dim, args.per_class, 12, 12)
    fig, axes = plt.subplots(model.out_dim, args.per_class, figsize=(args.per_class, 10))
    for c in range(model.out_dim):
        for j in range(args.per_class):
            axes[c, j].imshow(samples[c, j], cmap="gray", vmin=-1, vmax=1)
            axes[c, j].axis("off")
    fig.tight_layout(pad=0.1)
    output = Path(args.output) if args.output else run_dir / "jem_samples.png"
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
