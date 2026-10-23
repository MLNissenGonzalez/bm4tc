"""Benchmark: one UQ evaluation (the `uq` part of the analysis) eager vs captured (D92),
or with ``--adaptive`` the adaptive attack (the `adaptive` part, D117).

PGD and likelihood purification at one budget, log p(x) and the predictions, on
random inputs in the embedding's input range, with the analysis defaults
(PGD-40, purification-40, batch 256). Gibbs is not included (it stays eager).
``--adaptive``: one restart of the adaptive attack, from the clean point (so
captured equals eager), through each deterministic defence of the part:
likelihood purification, the detector on its output, the detector alone.
Not collected by pytest; run by hand on the GPU:

    python -m tests.bench.bench_analysis [--features 144] [--bond-dim 40] [--samples 2048]
    python -m tests.bench.bench_analysis --adaptive [--adaptive-steps 100] [--samples 256]

The two runs share the model and the data; the captured one includes its warm-up
and capture (three eager batches per shape), as an analysis does.
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.analysis.adaptive import AdaptivePGD, Defense
from bm4tc.analysis.purification import LikelihoodPurification
from bm4tc.analysis.uq import UQConfig, UQEvaluation
from bm4tc.core.graphs import Graphs
from bm4tc.core.interface import log_px
from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig

N_CLASSES = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=int, default=144, help="sites (MNIST12 144, MNIST 784)")
    parser.add_argument("--bond-dim", type=int, default=40)
    parser.add_argument("--samples", type=int, default=2048, help="test samples (MNIST: 10000)")
    parser.add_argument("--batch", type=int, default=256, help="analysis.batch_size")
    parser.add_argument("--steps", type=int, default=40, help="attack and purification steps")
    parser.add_argument("--adaptive", action="store_true", help="the adaptive attack instead")
    parser.add_argument("--adaptive-steps", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda")

    torch.manual_seed(0)
    born = CBMConfig(init_kwargs=MPSInitConfig(in_dim=3, bond_dim=args.bond_dim))
    cbm = ConditionalBornMachine(born, args.features, N_CLASSES, device)
    cbm.prepare(device=device)
    lo, hi = cbm.input_range
    inputs = lo + (hi - lo) * torch.rand(args.samples, args.features)
    labels = torch.randint(0, N_CLASSES, (args.samples,))
    if args.adaptive:
        return _adaptive(cbm, inputs, labels, args, device)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=args.batch)
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


def _adaptive(cbm, inputs, labels, args, device):
    lo, hi = cbm.input_range
    eps_abs, delta_abs = 0.1 * (hi - lo), 0.1 * (hi - lo)
    with torch.no_grad():
        tau = float(log_px(cbm, inputs[:args.batch].to(device)).quantile(0.05))
    attack = AdaptivePGD(steps=args.adaptive_steps, restarts=1)
    print(f"legendre d3r{args.bond_dim} c64, {args.features} features, {args.samples} samples, "
          f"batch {args.batch}, adaptive-{args.adaptive_steps} (1 restart) at eps 0.1, "
          f"purification-{args.steps} at delta 0.1, {torch.cuda.get_device_name(device)}")
    advs = {}
    for captured in (False, True):
        graphs = Graphs(enabled=captured)
        purifier = LikelihoodPurification(num_steps=args.steps).graphed(cbm, graphs)
        radius = graphs.scalar(delta_abs, device)
        purify = lambda x: purifier(x, radius)[0]
        defences = {"purify": Defense(cbm, purify),
                    "purify_sel": Defense(cbm, purify, tau, detect_on="purified"),
                    "rob_sel": Defense(cbm, threshold=tau)}
        for name, defense in defences.items():
            graphed = attack.graphed(defense, graphs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start, out, right = time.perf_counter(), [], []
            for i in range(0, args.samples, args.batch):
                x = inputs[i:i + args.batch].to(device)
                y = labels[i:i + args.batch].to(device)
                out.append(graphed(x, y, graphs.scalar(eps_abs, device), starts=[x]))
                right.append(defense.rates(out[-1], y)[0])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            adv = torch.cat(out)
            same = "" if not captured else \
                f", {'equal to' if torch.equal(adv, advs[name]) else 'DIFFERS from'} eager"
            advs[name] = adv
            print(f"  {'captured' if captured else 'eager   '} {name:10s}: {seconds:8.1f} s, "
                  f"right {float(torch.cat(right).mean()):.3f}{same}, "
                  f"peak {torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved")
        del graphs, purifier, defences, graphed
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
