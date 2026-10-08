"""Profile: where one MPS training step spends its time (the per-step overhead,
docs/plan.md "Efficiency"). Not collected by pytest; run by hand:

    python -m tests.bench.profile_train_step [--sites 144] [--bond 20] [--batch 512]
        [--beta 0.5] [--steps 10] [--device cuda] [--trace out.json]

Times the parts of a step separately (amplitudes, log Z, backward, optimizer) with
device synchronisation, counts the CUDA kernel launches and the Python-side
operator calls per step, and prints the top operators by CPU self time
(torch.profiler). ``--trace`` writes a Chrome trace (chrome://tracing, Perfetto).
"""
import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn, device, reps):
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps * 1e3, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=144)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--in-dim", type=int, default=3)
    ap.add_argument("--bond", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trace", default=None)
    a = ap.parse_args()
    device = torch.device(a.device)

    torch.manual_seed(0)
    born = CBMConfig(init_kwargs=MPSInitConfig(in_dim=a.in_dim, bond_dim=a.bond))
    cbm = ConditionalBornMachine(born, a.sites, a.classes, device)
    cbm.prepare(device=device)
    lo, hi = cbm.input_range
    x = (lo + (hi - lo) * torch.rand(a.batch, a.sites)).to(device)
    y = torch.randint(0, a.classes, (a.batch,), device=device)
    opt = torch.optim.Adam(cbm.parameters(), lr=1e-4)

    def step():
        opt.zero_grad()
        loss = cbm.mixed_nll(x, y, beta=a.beta)
        loss.backward()
        opt.step()
        return loss

    for _ in range(3):                       # warm-up (tracing, allocator, cuBLAS)
        step()

    # The parts, each repeated on the current parameters.
    ms_amp, _ = _timed(lambda: cbm.log_amp_sq(x), device, a.steps)
    ms_logz, _ = _timed(lambda: cbm.log_Z(recompute=True), device, a.steps)

    def fwd_bwd():
        opt.zero_grad()
        cbm.mixed_nll(x, y, beta=a.beta).backward()
    ms_fb, _ = _timed(fwd_bwd, device, a.steps)
    ms_opt, _ = _timed(opt.step, device, a.steps)
    ms_step, _ = _timed(step, device, a.steps)

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(a.steps):
            step()
        _sync(device)
    events = prof.key_averages()
    launches = sum(e.count for e in events if e.key in ("cudaLaunchKernel", "cuLaunchKernel"))
    aten_calls = sum(e.count for e in events if e.key.startswith("aten::"))
    cuda_ms = sum(e.self_cuda_time_total for e in events) / 1e3 / a.steps if device.type == "cuda" else float("nan")

    print(f"\n{a.sites} sites, d{a.in_dim}r{a.bond}, batch {a.batch}, beta {a.beta}, {device}")
    print(f"  step              {ms_step:8.1f} ms")
    print(f"    log|psi|^2 fwd  {ms_amp:8.1f} ms")
    print(f"    log Z (grad)    {ms_logz:8.1f} ms")
    print(f"    fwd + bwd       {ms_fb:8.1f} ms")
    print(f"    optimizer       {ms_opt:8.1f} ms")
    print(f"  per step: {launches / a.steps:.0f} kernel launches, {aten_calls / a.steps:.0f} aten calls, "
          f"GPU busy {cuda_ms:.1f} ms of {ms_step:.1f} ms")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
    if a.trace:
        prof.export_chrome_trace(a.trace)
        print(f"trace: {a.trace}")


if __name__ == "__main__":
    main()
