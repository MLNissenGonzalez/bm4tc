"""Benchmark: wall time and peak GPU memory per training step, NAT and AT.

The baseline for D31 (cache ownership, `ousterhout` Phase 3), and the memory probe
for a study's shape before a launch (D90: units share a GPU under MPS, so memory
per unit bounds `--per-gpu`). Not collected by pytest; run by hand on the GPU and
put the numbers in the commit message:

    python -m tests.bench.bench_train_step [--steps 20] [--epochs 3] [--device cuda]
    python -m tests.bench.bench_train_step --features 784 --bond-dim 40 --batch 512 \
        --micro-batch 256 --beta 0.5          # full MNIST d3r40, as mnist_capacity trains it

Random inputs in the embedding's input range, 10 classes; the defaults are MNIST12
(144 features), d3r20, batch 256, β = 0. The Trainer runs through its public
`train()` with its defaults (on CUDA the step is captured, D90; `--eager` turns that
off); only `_train_epoch` is timed (the validation pass is not), and the first epoch
is discarded as warm-up. Peak memory covers the whole run, validation and capture
included, without the CUDA context (≈ 0.3-0.5 GB more per process).
"""
import argparse
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
from bm4tc.core.train import Trainer, TrainConfig

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
N_CLASSES = 10


def _config(path: Path, **overrides):
    """A preset merged onto the schema, as Hydra composes it."""
    return OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(path), overrides)


def _datahandler(cbm, n_features: int, batch: int, steps: int) -> SimpleNamespace:
    lo, hi = cbm.input_range

    def loader(n, shuffle):
        inputs = lo + (hi - lo) * torch.rand(n, n_features)
        labels = torch.randint(0, N_CLASSES, (n,))
        return DataLoader(TensorDataset(inputs, labels), batch_size=batch, shuffle=shuffle)

    return SimpleNamespace(
        classification={"train": loader(steps * batch, True), "valid": loader(batch, False)},
        data_dim=n_features,
    )


def bench(regime: str, args, device: torch.device) -> tuple[list[float], int]:
    """Seconds per training step, one value per timed epoch, and the peak memory
    allocated [bytes] (0 on CPU)."""
    torch.manual_seed(0)
    born = CBMConfig(init_kwargs=MPSInitConfig(in_dim=3, bond_dim=args.bond_dim))  # defaults: legendre c64
    cbm = ConditionalBornMachine(born, args.features, N_CLASSES, device)
    dh = _datahandler(cbm, args.features, args.batch, args.steps)
    common = dict(beta=args.beta, max_epoch=args.epochs, batch_size=args.batch,
                  micro_batch_size=args.micro_batch, cuda_graph=not args.eager, save=False)
    if regime == "nat":
        cfg = _config(CONFIGS / "trainer/nat.yaml", **common)
    else:
        cfg = _config(CONFIGS / "trainer/at.yaml", eval_every=1, **common)
    trainer = Trainer(cbm, cfg, dh.classification["train"], dh.classification["valid"], device)

    times = []
    epoch = trainer._train_epoch

    def timed(*epoch_args, **epoch_kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        epoch(*epoch_args, **epoch_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / args.steps)

    trainer._train_epoch = timed
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trainer.train()
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return times[1:], peak


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20, help="training steps per epoch")
    parser.add_argument("--epochs", type=int, default=3, help="epochs, the first is warm-up")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--regime", choices=["nat", "at", "both"], default="both")
    parser.add_argument("--features", type=int, default=144, help="sites (MNIST12 144, MNIST 784)")
    parser.add_argument("--bond-dim", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--eager", action="store_true", help="do not capture the step")
    args = parser.parse_args()
    device = torch.device(args.device)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    mode = "eager" if args.eager or device.type != "cuda" else "captured"
    print(f"legendre d3r{args.bond_dim} c64, {args.features} features, {N_CLASSES} classes, "
          f"batch {args.batch}, micro-batch {args.micro_batch}, β {args.beta}, {mode}, "
          f"{args.steps} steps x {args.epochs - 1} timed epochs, {name}")
    for regime in (["nat", "at"] if args.regime == "both" else [args.regime]):
        times, peak = bench(regime, args, device)
        ms = [1e3 * t for t in times]
        print(f"  {regime}: {statistics.mean(ms):8.1f} ms/step  "
              f"(min {min(ms):.1f}, max {max(ms):.1f}, {len(ms)} epochs), "
              f"peak {peak / 2**30:.2f} GiB")
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
