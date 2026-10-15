# Compute estimates for Phase 8 (2026-10-07)

Measured on the dev laptop's RTX 2080 (8 GB, 16 CPU threads) with short real runs:
the pipeline's own `_fit` and analysis parts on the study configs at commit
`c4994868`. Every number is an **upper bound**: every run goes to `max_epoch`
(early stopping only shortens it). The raw measurements are in the
[appendix](#appendix-measurements).

## Takeaways

*Eager estimates from 2026-10-07: their per-step costs are superseded by the sections
after them (CUDA graphs, D101/D103 for Gibbs, D104 for the eager forward).*

1. **About 3,500 GPU-hours without Gibbs** (one unit per GPU, 2080 speed). Full
   MNIST is about 70% of that (`mnist_capacity`, `mnist_nat`, `mnist_at`), and MPS
   on MNIST 12×12 is about 20%. Spirals (≈ 12 h in all) and JEM (≈ 240 h) hardly
   count.
2. **HPO is 71% of the training cost.** It is 30 trials per cell against 5 seeds,
   and every trial runs the full `max_epoch`. Fewer trials, shorter trials, or HPO
   on fewer cells cut the bill more than anything else that needs no code.
3. **Training is limited by Python overhead, not the GPU.** Epoch time does not
   depend on the bond dimension (MNIST 12×12: 20 s at r = 10, 20 and 40). It grows
   with the number of sites and the number of steps per epoch. One unit keeps one
   CPU core busy and leaves the GPU mostly idle. Two consequences:
   - **Several units per GPU are almost free.** 3 per GPU give 2.8× the
     throughput, 6 give 4×. The limits are **CPU threads** (G17G01: 40, G21G01: 80)
     and **GPU memory**.
   - **A larger batch is faster almost one for one.** 4× the batch (2048 vs 512)
     gave 3.6× faster epochs. But it changes the optimisation (4× fewer steps), so
     every result changes and the HPO has to absorb it. It needs a decision and a
     check before Phase 8, not after.
4. **Analysis costs as much as training for MPS.** PGD-40 at 5 budgets plus
   likelihood purification, then the same again for the joint attack, takes
   ≈ 40 min per run on MNIST 12×12 and ≈ 3.6 h on full MNIST. That is ≈ 330
   GPU-h in all.
5. **Gibbs purification on full MNIST was not affordable as written; it is now
   (D101, D103: ≈ 50 s per run on full MNIST, ≈ 9 s on MNIST 12×12; see
   [Gibbs purification](#gibbs-purification-since-d101-and-d103-2026-10-14)).** Each
   one-feature update evaluates (batch × 96 bins) candidates with a full forward
   pass over all n sites, so one sweep costs O(n²). On MNIST 12×12 it takes 447 s
   for 24 points (≈ 1.3 h per run at 250 points). Full MNIST is ≈ 30× slower per
   batch: **≈ 40–80 GPU-h per run**, so 1,000–2,000 GPU-h for all of `mnist_nat`.
   The fix is to cache the left and right environments, which makes each update cost
   one site instead of n (see `docs/plan.md`). Until then, run Gibbs only where the
   paper reads it: `mnist_sweep_purify` uses `mnist_nat` α = 0.01 only.
6. **Memory decides where full-MNIST runs can go.** At batch 512, d3r20 needs
   2.5 GB, d3r40 ≈ 10.4 GB and d3r80 ≈ 41 GB (memory grows with batch × r²).
   - d3r40 fits only the RTX 6000s (G21G01, 2 per GPU). It is too tight for a 1080 Ti.
   - **d3r80 fits no GPU at batch 512.** Drop it from `mnist_capacity`, or give it
     batch 128 (≈ 10 GB, but 4× slower and a different batch from its neighbours).
   - Gibbs on full MNIST also runs out of memory on 8 GB at its configured batch of
     24 points.
7. **The PGD-5 pilot matters for cost.** Adversarial training is 6.4× the cost of
   NAT per epoch with PGD-10. `mnist_at` alone is ≈ 980 GPU-h, the most expensive
   study. PGD-5 would roughly halve that.

## What it means for wall time

- **Total, roughly:** two GPU nodes (G17G01 + G21G01: 16 GPUs, 120 threads) at
  4 units per GPU give about 50 single-unit equivalents. 3,500 GPU-h then takes
  **≈ 3 days**, plus packing losses, so about a week if the GPUs are mostly free.
  The two 1080 Ti nodes with 8 GPUs each (G18G01, G18G02) add the same again.
- **Critical path:** `mnist_capacity` → `mnist_nat` → `mnist_at`. One `mnist_at`
  trial is 100 epochs × 680 s ≈ 19 h on its own. The chain is ≥ 2–3 days even
  with unlimited slots.
- **First check on the cluster:** these runs are limited by CPU speed, and the
  cluster's CPUs may be slower per thread than the laptop's. Time one probe on a
  node (MNIST 12×12 d3r20 NAT, 3 epochs) before planning in detail.

## Recommended order

1. **Decide the batch size** (512 or larger) and the **HPO budget**. They change
   every number, pilots included.
2. **Fix efficiency where the numbers don't change**, on the Phase 9 track (D30):
   - Gibbs environment caching, checked bit-identical to the current code (done:
     D101, D103).
   - Per-step overhead (CUDA graphs or `torch.compile`, fewer small launches).
   - Analysis batch size: it changes PGD's random draws, so it needs a re-pin.
3. **Pilots** (PGD-5 vs 10; norm-control target) can run on the cluster at the
   same time as step 2, once step 1 is settled. They only train, so Gibbs caching
   and overhead fixes don't change their answers.

## Since CUDA graphs (2026-10-08 to 2026-10-13; not folded into the estimates yet)

The estimates above are eager and per unit. These measurements replace their
per-step costs; re-estimating the totals is open (plan.md, Efficiency).

**Eager, where the time went** (*stale since D104: the forward no longer resets;
now 130 ms per step, the log|ψ|² forward 19 ms, see
[Since D104](#since-d104-the-forward-keeps-its-nodes-2026-10-14)*;
`python -m tests.bench.profile_train_step`, laptop
RTX 2080, 144 sites, d3r20, batch 512, β = 0.5, 2026-10-08): 170 ms per step, of
which the log|ψ|² forward is 54 ms, log Z 25 ms, forward + backward 167 ms, Adam
3 ms. Per step ≈ 9,900 kernel launches and 34,000 aten calls; the GPU is busy 74 ms
of the 170, on tiny (r × r) kernels (`bmm`, `div`, `vector_norm`, `where`,
`masked_fill` once per site, forward and backward). Launching alone costs ≈ 44 ms
of CPU per step. Epoch time does not depend on the bond dimension; it grows with
sites × steps. On G21G01 (busy node) an eager NAT step took 440 ms, ≈ 2.4× the
laptop, CPU-bound.

**Captured step** (prototypes `pilots/graph_*.py`, untracked, 2026-10-08; integrated
as D90): bit-identical to eager for NAT β ∈ {0, 0.5, 1}, d3r20/d3r40, 144 and 784
sites, micro-batched, and AT PGD-5/10 with a fixed start (a random start draws
from other RNG offsets). Per unit: G21G01 8.7× (NAT d3r20), 9.9× (AT PGD-10);
laptop 3.9× / 4.5×; full MNIST (laptop) 4.0× NAT d3r20, 3.7× NAT d3r40 (micro
128), 3.3× AT d3r20. Capture takes 0.3–8 s once. (*The ratios are against the eager
step before D104, stale: the eager step is faster now, the captured one unchanged.*)

**MPS** (several captured units per GPU), G21G01, steps/s per GPU at 1/2/4/6/8
units: NAT graph 20.5/36.5/58.4/77.6/88.3 (eager 2.4/5.0/10.0/13.2/16.8); AT
PGD-10 graph 3.6/5.7/7.8/8.9/9.5 (eager 0.4/0.8/1.4/2.0/2.6). So ≈ 5–6× NAT and
≈ 4–5× AT per GPU against 6 eager units. (*Eager columns stale since D104.*)

**What is left per step is GPU time:** NAT d3r20 10.5k kernels (30 ms) + 15 ms
gaps; AT 58k kernels (242 ms) + 22 ms gaps. The largest single cost (≈ 1/3) is the
complex ÷ real division of the renormalisation (chain `[B, r] / [B, 1]`, log Z's
tk `renormalize`), forward and backward, on PyTorch's slow mixed-dtype kernel
(7.3 µs). Casting the divisor to complex does not help; multiplying a real view by
1/n halves it but is not bit-identical (≈ 1e-7).

**Memory per unit** (`tests.bench.bench_train_step`, captured, β = 0.5, RTX 6000;
*stale since D104, re-measure on the cluster: at E2a's shape the RTX 2080 shows
+0.8 GiB reserved per process, 3.69 → 4.49 GiB, allocated unchanged*):

| Shape | ms/step | Memory |
|---|---|---|
| MNIST12 d3r40 batch 512 NAT (E2a) | 70 | 3.89 GiB per process on G21G01, 3.69 on the RTX 2080 (D95: CUDA context and both validation shapes included) |
| full MNIST d3r40 micro 256 NAT | 595 | 5.34 GiB allocated (2026-10-09, before D92/D95: no context, eager validation) |
| full MNIST d3r40 micro 256 NAT, real run (E2b, G21G01, MPS) | | ≈ 9.3 GiB per process (train); its analysis ≥ 10.7 GiB |
| full MNIST d3r40 micro 256 AT PGD-5 | 2,600 | 7.58 GiB, likewise |
| full MNIST d3r80 micro 128 NAT | | does not fit: validation needs ≈ 4.8 GiB beside the step's 16 GiB pool |

`--per-gpu` ≈ (GPU memory − other processes) / the per-process figure; E2a runs 4
per GPU on G21G01. (*This table is superseded by the cluster check of D107 + D110,
next section.*)

## Cluster check of D107 + D110 (2026-10-15)

G21G01, GPU 5 alone, worktree `runs/5e41a7d3`, one process at a time; logs in
`$R/d110_checks/` (copied to `pilots/cluster/d110_checks/`, untracked).
`tests/unit/test_graphs.py` passes 19/19. `bench_train_step --beta 0.5 --bond-dim
{40,80} --batch 512`, 20 steps × 5 epochs (epoch 1 eager, 2 captures, 3–5 timed),
GiB:

| Shape | ms/step captured (eager) | Peak allocated | Peak reserved | Reserved at end | Per process (end, nvidia-smi) | Per process before (`da7185e8`) |
|---|---|---|---|---|---|---|
| MNIST12 d3r40 NAT | 63.8 | 1.82 | 2.87 | 2.87 | 3.20 | 4.57 |
| MNIST12 d3r40 AT | 283 | 2.75 | 2.93 | 2.93 | 3.42 | 10.92 |
| full MNIST d3r40 micro 256 NAT | 545 (1,824) | 5.03 | 7.80 | 5.49 | 6.71 (peak ≈ 9.0) | 10.46 |
| full MNIST d3r40 micro 256 AT | 2,486 (6,682) | 7.56 | 8.06 | 8.06 | 10.32 | out of memory |
| full MNIST d3r80 micro 128 NAT | 1,431 | 10.46 | 16.04 | 11.51 | 13.62 (peak ≈ 18.2) | out of memory |
| full MNIST d3r80 micro 128 AT | 6,311 | 15.38 | 16.44 | 16.43 | 19.93 | not tried |

Eager at the full-MNIST d3r40 shapes: NAT 5.02 / 5.34 / 5.54 GiB (allocated,
reserved, per process), AT 7.55 / 7.89 / 8.10. Captured is 3.3× (NAT) and 2.7× (AT)
faster per step.

- **Reserved now follows allocated**, except for a transient at the first capture in
  NAT: `torch.cuda.graph` empties the cache when it starts, but the epoch-1 eager
  tensors still held by tk's nodes are freed only once the capture has replaced them,
  so the pool and those leftovers coexist for a moment (full MNIST NAT 7.80 GiB peak
  vs 5.49 at the end; d3r80 16.04 vs 11.51). "Peak ≈" in the table adds the
  process's non-allocator memory (per process − reserved at end) to the peak
  reserved. Open: dropping those references before the first capture would remove
  it (not done; plan.md).
- **Outside the allocator**, a process holds 0.3–0.5 GiB at 144 sites and 1.2–3.5 GiB
  at 784 (CUDA context and the graphs' executables, which grow with the number of
  kernels captured).

`bench_analysis --features 784` (d3r40, 2048 points, PGD-40 + purification-40 at
ε = 0.1): the same numbers captured and eager (rob 0.104, purify 0.104).

| `analysis.batch_size` | Eager | Captured |
|---|---|---|
| 64 | 1,472 s, 1.52 GiB reserved | 394 s, 2.99 GiB |
| 256 | 434 s, 7.59 GiB | 170 s, 10.38 GiB |

Captured ≈ 0.04 GiB per point, plus ≈ 0.5; add ≈ 1–2 GiB per process outside the
allocator.

**`--per-gpu` on G21G01** (22 GB usable, the GPU otherwise empty; the largest unit
of a study decides, since every slot is sized alike):

| Study shape | Largest unit | `--per-gpu` |
|---|---|---|
| MNIST12 d3r40 (NAT, AT) | train ≤ 3.4 GiB; analysis at 144 sites not re-measured since D107 (≤ 4 GiB expected) | 5 |
| full MNIST d3r40, `analysis.batch_size` 128 (the studies' setting since D111) | AT train 10.3 GiB, analysis ≈ 6.5 | 2 |
| full MNIST d3r40, `analysis.batch_size` 256 (the default before D111) | analysis ≈ 11.5–12.5 GiB | 1 (2 may fail when two analyses meet) |
| full MNIST d3r80 | AT train 19.9 GiB; analysis at 128 unmeasured (≈ 20 GiB by B·n·r²) | 1 |

## Gibbs purification since D101 and D103 (2026-10-14)

`GibbsPurification` through the API both versions share
(`purify_snapshots`, 96 bins, step 0.05), RTX 2080, random model (time does not
depend on the weights). Before: `3b6afd73` (every candidate a full-chain
contraction, O(n²) per sweep). D101: `954fa692` (one `sample` call per sweep, the
ket alone). D103: the nodes kept between calls. Each from its own worktree.

One sweep (peak memory above the model):

| Shape | Before (batch 24) | D101, batch 24 | D101, batch 64 |
|---|---|---|---|
| spirals d10r6, 512 points, batch 256 | 0.02 s, 31 MiB | 0.02 s, 10 MiB | – |
| MNIST 12×12 d3r40, per point | 285 ms, 4.2 GiB | 21 ms, 79 MiB | 7.9 ms, 209 MiB |
| MNIST 12×12 d3r20, per point | 282 ms, 1.1 GiB | – | 7.8 ms, 131 MiB |
| full MNIST d3r40, per point | out of memory (also at batch 8) | 152 ms, 428 MiB | 57 ms, 1.1 GiB |
| full MNIST d3r80, per point | out of memory (also at batch 4) | – | 57 ms, 2.6 GiB |

A batch costs the same at 24 or 64 points (0.5 s on MNIST 12×12, 3.6 s on full
MNIST): the time is tensorkrowch's Python work per operation, not arithmetic, so the
largest batch that fits is the cheapest per point.

Ten sweeps (snapshots 1, 3, 6, 10, the analysis default), batch 64, log p(x)
included; the draws of D101 and D103 are identical:

| Shape | D101 | D103 | Per run (250 points) |
|---|---|---|---|
| MNIST 12×12 d3r40, 256 points | 18.6 s | 8.6 s, 229 MiB | ≈ 9 s |
| full MNIST d3r40, 128 points | 69.4 s | 25.4 s, 1.2 GiB | ≈ 50 s |
| full MNIST d3r80, 128 points | 69.5 s | 26.1 s, 2.6 GiB | ≈ 52 s |
| spirals d10r6, 512 points (batch 256) | 0.2 s | 0.1 s | – |

The totals include the final log p(x) forward, eager, which is faster since D104
(so they are slightly high). Before D100, the same run took ≈ 12 min on MNIST 12×12 (285 ms per point and sweep)
and did not fit on full MNIST, where takeaway 5 estimated 40–80 GPU-h per run. A
D103 call that reuses its nodes costs ≈ 40% of one that builds them; the first sweep
after each forward builds them (the adaptive attack classifies between purifications,
so there only sweeps 2..k of a k-sweep purification reuse).

## Gibbs purification since D106 (2026-10-15)

D106 replaces D101's open-leg sweep: an environment is double only where it contains
a summed site (in a sweep, the class), and each step of the sweep is a few fixed
nodes whose tensors are reset from site to site. RTX 2080, random model (legendre
d3, complex64), 96-bin windows of step 0.05, batch 64; before = HEAD `918b7c34`
(D103) from a worktree, after = D106. The draws are identical (same checksums).

One sweep, `born.sample` alone (peak above the model; held = what stays allocated
until the next call):

| Shape | D103: first / later call | D106: first / later call |
|---|---|---|
| MNIST 12×12 d3r40 | 0.53 / 0.19 s, peak 205–226 MiB, held 205 MiB | 0.37 / 0.20 s, peak 83–92 MiB, held 15 MiB |
| full MNIST d3r40 | 2.51 / 1.02 s, peak 1.1–1.2 GiB, held 1.1 GiB | 1.15 / 1.11 s, peak 430–439 MiB, held 15 MiB |
| full MNIST d3r80 | 2.49 / 1.02 s, peak 1.9–2.0 GiB, held 1.9 GiB | 1.15 / 1.12 s, peak 1.3–1.4 GiB, held 53 MiB |

Most of D106's peak is the double right environments kept for the sweep, left of
the class: batch × n/2 × r² complex values (64 × 392 × 80² × 8 B ≈ 1.3 GB at r80,
0.3 GB at r40). Nothing else grows with the number of sites.

`GibbsPurification.purify_snapshots`, log p(x) forward included: one sweep of 64
points (first call) MNIST 12×12 r40 0.84 → 0.59 s, full MNIST r40 4.16 → 1.98 s,
r80 4.23 → 1.94 s; ten sweeps (snapshots 1, 3, 6, 10) MNIST 12×12 r40 (256 points)
8.2 → 8.5 s, full MNIST r40 / r80 (128 points) 24.6 → 23.9 s / 24.7 → 24.2 s. Peak
in the ten-sweep run: 238 → 136 MiB, 1.30 → 0.73 GiB, 2.82 → 2.89 GiB; at r80 it is
the log p(x) forward's, not the sampler's. So the per-process memory of an analysis
does not grow (E2b's and E3's settings unchanged); between Gibbs calls the sampler
now holds megabytes instead of 1–2 GiB.

Open (plan.md, "Sampling" step 6): float32 rounding of double layers. On random
models these runs gave no negative weights; trained models are not checked yet.

## Since D104: the forward keeps its nodes (2026-10-14)

`ConditionalBornMachine.forward` calls `contract()` on the nodes of its previous call
instead of resetting (D104). Laptop RTX 2080, from a worktree of the commit before
(`594d1193`) and the change; the same numbers out.

| Measurement | Before | After |
|---|---|---|
| `profile_train_step` (MNIST 12×12 d3r20, B 512, β 0.5, eager): step | 173.8 ms | 130.0 ms |
| — of which the log\|ψ\|² forward | 54.1 ms | 19.4 ms |
| — log Z (grad), optimizer | 24.0, 3.2 ms | 23.6, 3.0 ms |
| — kernel launches per step, GPU busy | 9,862, 73.7 ms | 9,862, 73.8 ms |
| `bench_train_step` E2a's shape (MNIST 12×12 d3r40, B 512, β 0.5, NAT), captured | 61.7 ms/step, 1.82 GiB allocated, 3.44 reserved, 3.69 per process | 61.8 ms/step, 1.82 / 4.24 / 4.49 GiB |
| the same, eager | 178.1 ms/step, 1.82 / 1.95 / 2.09 GiB | 140.5 ms/step, 1.82 / 2.81 / 2.95 GiB |
| `bench_analysis` (MNIST 12×12 d3r40, 2048 points, PGD-40 + purification-40), eager | 81.0 s, 0.99 GiB reserved | 53.1 s, 0.99 GiB |
| the same, captured | 47.2 s, 2.98 GiB | 39.3 s, 2.93 GiB |

The eager forward no longer pays for building and deleting its nodes; the GPU work is
the same (same kernels), so captured training is unchanged. **Open:** reserved memory
grows by ≈ 0.8 GiB per process at E2a's shape although the allocated peak does not;
not the freeing strategy (freeing in the call or after it reserve the same), and a
bare train and validate loop shows none, so something in the Trainer's path
(warm-up stream, validation shapes) caches more. It decides `--per-gpu` (E2a:
4 × 4.5 GiB still fits 22 GB). Full MNIST d3r40 micro 256 and AT at E2a's shape do
not fit the 8 GB laptop (before and after alike): cluster.

## Since D112: the data zipped into the running vector (2026-10-15)

`log_amp_sq` contracts with `zipup_input=True` (D112): autograd keeps (B, r) vectors
per site instead of a (B, r, r) matrix. Laptop RTX 2080, `bench_train_step --beta 0.5
--bond-dim 40 --batch 512`, captured, from a worktree of the commit before (`9fd8a22a`)
and the change. Memory: allocated peak / reserved / per process (nvidia-smi).

| Shape | Before | After |
|---|---|---|
| MNIST 12×12 NAT | 61.7 ms/step, 1.82 / 2.87 / 3.12 GiB | 49.0 ms/step, 0.13 / 0.20 / 0.46 GiB |
| MNIST 12×12 AT | 303.0 ms, 2.75 / 2.93 / 3.31 | 182.1 ms, 0.18 / 0.27 / 0.67 |
| full MNIST micro 128 NAT | 887.9 ms, 2.64 / 4.08 / 4.66 | 880.6 ms, 0.33 / 0.50 / 2.26 |
| full MNIST micro 128 AT | 3635.9 ms, 3.90 / 5.93 / 7.59 | 3464.0 ms, 0.40 / 0.56 / 3.56 |
| full MNIST, no micro-batches, NAT | out of memory (8 GB) | 256.8 ms, 0.66 / 0.85 / 1.49 |
| full MNIST, no micro-batches, AT | out of memory | 1293.6 ms, 0.86 / 1.12 / 2.53 |

The allocated peak drops 7–14×; per process, what remains is mostly the CUDA context
and the graphs. At full MNIST the step is launch-bound, so micro-batches now cost
3–3.5× in time and buy nothing. Where each path wins (a 33-shape sweep of the
amplitude contraction alone, D112): in memory always, for d < r; in captured time from
B·r² ≈ 3·10⁵ up, 1.1–1.3× slower below; eager ≈ 1.3× slower. The cluster numbers
(`--per-gpu`) are not measured yet.

## Appendix: measurements

RTX 2080, torch 2.1.0, one unit alone unless stated. Epoch time is the mean over
epochs 2 onwards and includes validation; an AT epoch includes its share of the
PGD validation done every 5 epochs.

### Training, seconds per epoch

|Dataset|Model|Arch|Regime|α|Batch|s/epoch|Peak GB|
|---|---|---|---|---|---|---|---|
|spirals|MPS|d4r3 / d6r4 / d10r6 / d30r18|NAT|0|256|0.14 / 0.10 / 0.11 / 0.17|0.02|
|spirals|MPS|d10r6|NAT|1|256|0.13|0.02|
|spirals|MPS (hermite)|d10r6|NAT|0|256|0.12|0.02|
|spirals|MPS|d10r6|AT (PGD-10)|0|256|0.86|0.02|
|mnist12|MPS|d3r10 / d3r20 / d3r40|NAT|0|512|20.1 / 19.9 / 20.2|0.13 / 0.47 / 1.81|
|mnist12|MPS|d3r40|NAT|1|512|20.2|1.81|
|mnist12|MPS|d3r20|NAT|0|**2048**|**5.5**|1.80|
|mnist12|MPS|d3r40|NAT|0|2048|OOM (8 GB)|—|
|mnist12|MPS|d3r20 / d3r40|AT (PGD-10)|0|512|128 / 127|0.47 / 1.81|
|mnist12|JEM|d3r20|NAT|0|512|0.41|0.03|
|mnist12|JEM|d3r40|NAT|0.01, SGLD 40 / 20 steps|512|8.0 / 6.3|0.06|
|mnist12|JEM|d3r40|AT (PGD-10)|0 / 0.01 (SGLD 40)|512|2.2 / 10.3|0.06|
|mnist|MPS|d3r10 / d3r20|NAT|0|512|107 / 106|0.65 / 2.48|
|mnist|MPS|d3r40|NAT|0|512|OOM (8 GB), ≈ 10.4 GB extrapolated|—|
|mnist|MPS|d3r40|NAT|0|128|423|2.60|
|mnist|MPS|d3r80|NAT|0|512|OOM: one allocation of 19 GB|—|
|mnist|MPS|d3r40|AT (PGD-10)|0|512|OOM (8 GB)|—|
|mnist|JEM|d3r40|NAT|0.01, SGLD 40|512|11.9|0.19|

Several units on one GPU (mnist12 d3r20 NAT, 3 epochs each):

|Units|s/epoch each|Throughput vs 1|
|---|---|---|
|1|19.9|1.0×|
|3|21.3|2.8×|
|6|30.2|4.0×|

### Analysis per seed run

Timed on a test subset and scaled linearly to the test split (2,000 spirals,
7,000 MNIST points).

|Model|uq (PGD-40 × 5 budgets, detection, likelihood purification)|uq_joint|Gibbs / SGLD (250 points, sweeps 1–10)|
|---|---|---|---|
|spirals d10r6 / d30r18|16 s / 22 s|16 s / 22 s|—|
|mnist12 MPS d3r20|21 min|20 min|1.3 h (447 s per 24 points)|
|mnist MPS d3r20|≈ 1.8 h (scaled by the epoch ratio 5.3)|≈ 1.8 h|≈ 40–80 h (scaled; OOM at Gibbs batch 24 on 8 GB; a timing run at batch 12 was stopped after ~1 h)|
|mnist12 JEM d3r40|23 s|16 s|33 s (SGLD)|
|mnist JEM d3r40|29 s|—|—|

### Units per study and upper-bound GPU-hours (2080, one unit per GPU)

`mnist_nat`/`mnist_at` assume the arch costs as much as d3r20, `mnist_capacity`
counts r10–r40 only, and `jem_mnist_*` are the not-yet-written copies of the
`jem_mnist12_*` studies. Gibbs is left out (see takeaway 5).

|Study|Cells|Trials + seeds per cell|HPO h|Seed runs h|Analysis h|Total h|
|---|---|---|---|---|---|---|
|spirals_nat|6|30 + 5|2|0|0|2|
|spirals_capacity|8|30 + 5|3|0|0|4|
|spirals_embedding|10|30 + 5|3|0|0|4|
|spirals_at|2|20 + 5|2|0|0|2|
|mnist12_nat|18|30 + 5|300|50|60|410|
|mnist12_at|4|20 + 5|284|71|13|369|
|jem_mnist12_nat|12|30 + 5|61|10|1|72|
|jem_mnist12_at|4|20 + 5|14|3|0|18|
|mnist_capacity|6|30 + 5|530|88|107|725|
|mnist_nat|6|30 + 5|530|88|107|725|
|jem_mnist_nat|12|30 + 5|102|17|1|120|
|mnist_at|2|20 + 5|754|188|36|978|
|jem_mnist_at|4|20 + 5|20|5|0|26|
|**Total**||| **2,604**|**524**|**327**|**≈ 3,455**|

Adding d3r80 at batch 128 to `mnist_capacity` would cost another ≈ 820 h
(2 cells × 35 runs × 100 epochs × 423 s).

### Side finding

When a Gibbs or SGLD analysis part runs, every budget logs
`Detection/attack failed (eps_rel=…): ; skipping`. The sweep part runs with no
detection percentiles, so `next(iter(det.values()))` raises an empty
`StopIteration` (`bm4tc/analysis/uq.py:611`). The attack results are stored
before that line, so the numbers are unaffected. It only adds noise to the log.
