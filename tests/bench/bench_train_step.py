"""Benchmark: wall time per training step, NAT and AT, legendre d3r20 on MNIST12-sized data.

The baseline for D31 (cache ownership, `ousterhout` Phase 3). Not collected by
pytest; run by hand on the GPU and put the numbers in the commit message:

    python -m tests.bench.bench_train_step [--steps 20] [--epochs 3] [--device cuda]

Random inputs in the embedding's input range; 144 features, 10 classes, batch 256.
The trainers run through their public `train()`; only `_train_epoch` is timed (the
validation pass is not), and the first epoch is discarded as warm-up.
"""
import argparse
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.model import ConditionalBornMachine
from src.train import NLLTrainer, AdversarialTrainer

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
N_FEATURES, N_CLASSES, BATCH = 144, 10, 256


def _config(path: Path, **overrides):
    """A config as Hydra passes it today: plain YAML, no schema (D25 is not in yet)."""
    return OmegaConf.merge(OmegaConf.load(path), overrides)


def _datahandler(cbm, steps: int) -> SimpleNamespace:
    lo, hi = cbm.input_range

    def loader(n, shuffle):
        x = lo + (hi - lo) * torch.rand(n, N_FEATURES)
        y = torch.randint(0, N_CLASSES, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=shuffle)

    return SimpleNamespace(
        classification={"train": loader(steps * BATCH, True), "valid": loader(BATCH, False)},
        data_dim=N_FEATURES,
    )


def bench(regime: str, steps: int, epochs: int, device: torch.device) -> list[float]:
    """Seconds per training step, one value per timed epoch."""
    torch.manual_seed(0)
    born = OmegaConf.load(CONFIGS / "born/legendre/d3r20c64.yaml")  # as Hydra passes it
    cbm = ConditionalBornMachine(born, N_FEATURES, N_CLASSES, device)
    dh = _datahandler(cbm, steps)
    if regime == "nat":
        cfg = _config(CONFIGS / "trainer/nll/default.yaml", alpha=0.0, max_epoch=epochs, save=False)
        trainer = NLLTrainer(cbm, cfg, dh, device)
    else:
        cfg = _config(CONFIGS / "trainer/adversarial/pgd_at.yaml",
                      alpha=0.0, max_epoch=epochs, save=False)
        trainer = AdversarialTrainer(cbm, cfg, dh, device)

    times = []
    epoch = trainer._train_epoch

    def timed(*args, **kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        epoch(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / steps)

    trainer._train_epoch = timed
    trainer.train()
    return times[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20, help="training steps per epoch")
    parser.add_argument("--epochs", type=int, default=3, help="epochs, the first is warm-up")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"legendre d3r20 c64, {N_FEATURES} features, {N_CLASSES} classes, batch {BATCH}, "
          f"{args.steps} steps x {args.epochs - 1} timed epochs, {name}")
    for regime in ("nat", "at"):
        times = bench(regime, args.steps, args.epochs, device)
        ms = [1e3 * t for t in times]
        print(f"  {regime}: {statistics.mean(ms):8.1f} ms/step  "
              f"(min {min(ms):.1f}, max {max(ms):.1f}, {len(ms)} epochs)")


if __name__ == "__main__":
    main()
