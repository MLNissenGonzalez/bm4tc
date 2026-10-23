"""Benchmark: wall time and peak GPU memory per training step, NAT and AT.

The baseline for D31 (cache ownership, `ousterhout` Phase 3), and the memory probe
for a study's shape before a launch (D90: units share a GPU under MPS, so memory
per unit bounds `--per-gpu`). Not collected by pytest; run by hand on the GPU and
put the numbers in the commit message:

    python -m tests.bench.bench_train_step [--steps 20] [--epochs 5] [--device cuda]
    python -m tests.bench.bench_train_step --features 784 --bond-dim 40 --batch 512 \
        --micro-batch 256 --beta 0.5          # full MNIST d3r40, as mnist_capacity trains it

Random inputs in the embedding's input range, 10 classes; the defaults are MNIST12
(144 features), d3r20, batch 256, β = 0. The Trainer runs through its public
`train()` with its defaults (on CUDA the step and the validation are captured, D90,
D92; `--eager` turns that off); only `_train_epoch` is timed (the validation pass is
not), and the first two epochs are discarded: epoch 1 runs eagerly, epoch 2
captures (D110). The validation split is 2.5 chunks, as a real one has full
chunks and a last partial one, so both shapes are captured, into the graphs' one
pool (D107). Peak memory covers
the whole run, validation and capture included: allocated, reserved (the
allocator's cache and the graphs' pools), and the process as nvidia-smi sees it at
the end, CUDA context included: that is what one unit takes from the GPU, so
`--per-gpu` is about (GPU memory - other processes) / this (D95; E2a's units held
3.9 GiB each on a cluster GPU where reserved said 2.9).

`--snapshot FILE` pickles the allocator's state after training (the viewer is
`python -m torch.cuda._memory_viz`) and prints what each memory pool holds then:
the graphs' pools (each named by the captures in it: the Trainer's attribute and
the input shape) and the general cache, per stream (the default stream, and the side stream that ran
the warm-ups, D94).
"""
import pickle
import argparse
import os
import statistics
import subprocess
import time
from typing import Optional
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
from bm4tc.core.train import Trainer, TrainConfig

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
N_CLASSES = 10
WARM_EPOCHS = 2   # untimed: epoch 1 eager, epoch 2 captures (D110)


def _config(path: Path, **overrides):
    """A preset merged onto the schema, as Hydra composes it."""
    return OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(path), overrides)


def _datahandler(cbm, n_features: int, batch: int, micro_batch, steps: int) -> SimpleNamespace:
    lo, hi = cbm.input_range

    def loader(n, batch_size, shuffle):
        inputs = lo + (hi - lo) * torch.rand(n, n_features)
        labels = torch.randint(0, N_CLASSES, (n,))
        return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=shuffle)

    # validation in chunks of the micro-batch, as the pipeline builds it (D79):
    # two full chunks and a half one, so both shapes are captured
    chunk = micro_batch or batch
    return SimpleNamespace(
        classification={"train": loader(steps * batch, batch, True),
                        "valid": loader(2 * chunk + chunk // 2, chunk, False)},
        data_dim=n_features,
    )


def process_gpu_memory() -> Optional[int]:
    """This process's GPU memory as the driver reports it [bytes]: the CUDA context
    and everything the allocator holds. None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        pid, mib = (field.strip() for field in line.split(","))
        if pid == str(os.getpid()):
            return int(mib) * 2**20
    return None


def _graphed(trainer: Trainer) -> dict:
    """The Trainer's captured functions by name, after train()."""
    graphed = {"train step": trainer._graphed_train_step}
    if trainer._graphed_attack is not None:
        graphed["train attack step"] = trainer._graphed_attack.step
    evaluation = trainer._evaluation
    graphed["valid clean sums"] = evaluation._clean_sums
    if evaluation._attack is not None:
        graphed["valid attack step"] = evaluation._attack.step
        graphed["valid adversarial sums"] = evaluation._adversarial_sums
    return graphed


def pool_report(trainer: Trainer, snapshot: dict) -> list[tuple[str, int, int]]:
    """(name, reserved, allocated) [bytes] per memory pool in ``snapshot``: the
    graphs' pools, named by the captures in each, then the general cache per
    stream."""
    graphs_in = {}
    for name, graphed in _graphed(trainer).items():
        for key, capture in graphed._captures.items():
            graphs_in.setdefault(capture.graph.pool(), []).append(f"{name} {list(key[0][0])}")
    names = {pool: ", ".join(graphs) for pool, graphs in graphs_in.items()}
    side = trainer._graphed_train_step.graphs.side_stream.cuda_stream
    totals = {}
    for segment in snapshot["segments"]:
        pool = tuple(segment["segment_pool_id"])
        if pool in names:
            name = names[pool]
        elif pool == (0, 0):
            stream = segment["stream"]
            name = ("cache, default stream" if stream == 0 else
                    "cache, side stream (warm-ups)" if stream == side else f"cache, stream {stream}")
        else:
            name = f"pool {pool} (not a Trainer graph)"
        reserved, allocated = totals.get(name, (0, 0))
        totals[name] = (reserved + segment["total_size"], allocated + segment["allocated_size"])
    return [(name, *totals[name]) for name in sorted(totals, key=lambda n: -totals[n][0])]


def bench(regime: str, args, device: torch.device) -> tuple[list[float], int, int, Optional[int], list]:
    """Seconds per training step, one value per timed epoch, the peak memory
    allocated and reserved, the process's GPU memory at the end [bytes]
    (0, 0, None on CPU), and with ``--snapshot`` the pools after training
    (:func:`pool_report`)."""
    torch.manual_seed(0)
    born = CBMConfig(init_kwargs=MPSInitConfig(in_dim=3, bond_dim=args.bond_dim))  # defaults: legendre c64
    cbm = ConditionalBornMachine(born, args.features, N_CLASSES, device)
    dh = _datahandler(cbm, args.features, args.batch, args.micro_batch, args.steps)
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
    if device.type != "cuda":
        return times[WARM_EPOCHS:], 0, 0, None, []
    pools = []
    if args.snapshot is not None:
        snapshot = torch.cuda.memory._snapshot()
        path = args.snapshot.with_name(f"{args.snapshot.stem}_{regime}{args.snapshot.suffix}")
        with open(path, "wb") as f:
            pickle.dump(snapshot, f)
        pools = pool_report(trainer, snapshot)
    return (times[WARM_EPOCHS:], torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device), process_gpu_memory(), pools)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20, help="training steps per epoch")
    parser.add_argument("--epochs", type=int, default=5,
                        help="epochs, the first two untimed (eager, then capture)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--regime", choices=["nat", "at", "both"], default="both")
    parser.add_argument("--features", type=int, default=144, help="sites (MNIST12 144, MNIST 784)")
    parser.add_argument("--bond-dim", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--eager", action="store_true", help="do not capture the step")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="pickle the allocator state after training to FILE_{regime}.pickle "
                             "and print the memory per pool")
    args = parser.parse_args()
    device = torch.device(args.device)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    mode = "eager" if args.eager or device.type != "cuda" else "captured"
    print(f"legendre d3r{args.bond_dim} c64, {args.features} features, {N_CLASSES} classes, "
          f"batch {args.batch}, micro-batch {args.micro_batch}, β {args.beta}, {mode}, "
          f"{args.steps} steps x {args.epochs - WARM_EPOCHS} timed epochs, {name}")
    for regime in (["nat", "at"] if args.regime == "both" else [args.regime]):
        times, allocated, reserved, process, pools = bench(regime, args, device)
        ms = [1e3 * t for t in times]
        unit = "" if process is None else f", {process / 2**30:.2f} GiB per process (nvidia-smi)"
        print(f"  {regime}: {statistics.mean(ms):8.1f} ms/step  "
              f"(min {min(ms):.1f}, max {max(ms):.1f}, {len(ms)} epochs), "
              f"peak {allocated / 2**30:.2f} GiB allocated, {reserved / 2**30:.2f} GiB reserved"
              f"{unit}")
        for name, pool_reserved, pool_allocated in pools:
            print(f"    {pool_reserved / 2**30:6.2f} GiB reserved, {pool_allocated / 2**30:6.2f} "
                  f"allocated: {name}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
