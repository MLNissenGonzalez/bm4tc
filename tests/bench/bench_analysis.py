"""Benchmark: one UQ evaluation (the `uq` part of the analysis) eager vs captured (D92).

PGD and likelihood purification at one budget, log p(x) and the predictions, on
random inputs in the embedding's input range, with the analysis defaults
(PGD-40, purification-40, batch 256). Gibbs is not included (it stays eager).
Not collected by pytest; run by hand on the GPU:

    python -m tests.bench.bench_analysis [--features 144] [--bond-dim 40] [--samples 2048]

The two runs share the model and the data; the captured one includes its warm-up
and capture (three eager batches per shape), as an analysis does.
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.analysis.uq import UQConfig, UQEvaluation
from bm4tc.core.graphs import Graphs
from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig

N_CLASSES = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=int, default=144, help="sites (MNIST12 144, MNIST 784)")
    parser.add_argument("--bond-dim", type=int, default=40)
    parser.add_argument("--samples", type=int, default=2048, help="test samples (MNIST: 10000)")
    parser.add_argument("--batch", type=int, default=256, help="analysis.batch_size")
    parser.add_argument("--steps", type=int, default=40, help="attack and purification steps")
    args = parser.parse_args()
    device = torch.device("cuda")

    torch.manual_seed(0)
    born = CBMConfig(init_kwargs=MPSInitConfig(in_dim=3, bond_dim=args.bond_dim))
    cbm = ConditionalBornMachine(born, args.features, N_CLASSES, device)
    cbm.prepare(device=device)
    lo, hi = cbm.input_range
    inputs = lo + (hi - lo) * torch.rand(args.samples, args.features)
    loader = DataLoader(TensorDataset(inputs, torch.randint(0, N_CLASSES, (args.samples,))),
                        batch_size=args.batch)
    cfg = UQConfig(eps_rel=[0.1], delta_rel=[0.1], percentiles=[5],
                   attack_num_steps=args.steps, num_steps=args.steps, eval_batch_size=args.batch)

    print(f"legendre d3r{args.bond_dim} c64, {args.features} features, {args.samples} samples, "
          f"batch {args.batch}, PGD-{args.steps} + purification-{args.steps} at eps 0.1, "
          f"{torch.cuda.get_device_name(device)}")
    for captured in (False, True):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        results = UQEvaluation(cfg).evaluate(cbm, loader, device, calib_loader=loader,
                                             graphs=Graphs(enabled=captured))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        print(f"  {'captured' if captured else 'eager   '}: {seconds:8.1f} s, "
              f"rob {results.adv_accuracies[0.1]:.3f}, "
              f"purify {results.purification_results[(0.1, 0.1)].accuracy_after_purify:.3f}, "
              f"peak {torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
