# Compute estimates for Phase 8 (2026-10-07)

Measured on the dev laptop's RTX 2080 (8 GB, 16 CPU threads) with short real runs:
the pipeline's own `_fit` and analysis parts on the study configs at commit
`c4994868`. Every number is an **upper bound**: every run goes to `max_epoch`
(early stopping only shortens it). The raw measurements are in the
[appendix](#appendix-measurements).

## Takeaways

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
5. **Gibbs purification on full MNIST is not affordable as written.** Each
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
   - Gibbs environment caching, checked bit-identical to the current code.
   - Per-step overhead (CUDA graphs or `torch.compile`, fewer small launches).
   - Analysis batch size: it changes PGD's random draws, so it needs a re-pin.
3. **Pilots** (PGD-5 vs 10; norm-control target) can run on the cluster at the
   same time as step 2, once step 1 is settled. They only train, so Gibbs caching
   and overhead fixes don't change their answers.

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
`StopIteration` (`bm4tc/analysis/uq.py:556`). The attack results are stored
before that line, so the numbers are unaffected. It only adds noise to the log.
