# Plan: what is left

The `ousterhout` refactor (2026-10-06 to 2026-10-07) simplified bm4tc before the
experiments are re-run for the journal version of the rejected NeurIPS submission
27079. Phases 0–7 are done: seam tests and benchmark, deletions, one metrics module,
one Trainer, model-owned caches, studies and `run.json`, the `bm4tc/{core,analysis,
pipeline}` package with the CLI, JEM in the pipeline, figures from paper manifests,
and these docs. How each was decided is in [decisions.md](decisions.md) (D1–D77); the
per-phase plan is in the git history (`plan.md` before this file).

## Phase 8: reproduce the journal paper

Run the studies on the lab HPC, then `python -m bm4tc figures paper`.

Order (warm studies need their `warm_from` study trained):

1. Pilots that set defaults (one-off, by hand):
   - PGD-5 vs PGD-10 AT training on `mnist12` d3r40 α=0 (D43): if PGD-5 loses more
     than ~1 point of robust accuracy at PGD-40 evaluation, keep 10; otherwise set
     `num_steps: 5` in `configs/trainer/at.yaml`.
   - The norm-control target for cold MNIST runs, 0 vs n·ln d / 2 (D59).
2. `spirals_nat`, `spirals_capacity`, `spirals_embedding`, `mnist12_nat`,
   `mnist_capacity`, `jem_mnist12_nat` (independent; cold).
3. `spirals_at`, `mnist12_at`, `jem_mnist12_at` (warm).
4. Set `arch` in `mnist_nat` / `mnist_at` from `mnist_capacity` (D44's two
   criteria), write `jem_mnist_{nat,at}` (copies of the `jem_mnist12_*` studies on
   `mnist`, `arch` = the chosen MPS), then run `mnist_nat`, `jem_mnist_nat`,
   `mnist_at`, `jem_mnist_at`.
5. `python -m bm4tc figures paper`; commit `figures/journal/`.

Still to provide for Phase 8:
- A `run paper`-style launcher across studies (the executor already runs a DAG;
  today one launch runs one study, D66).
- **HPO budget (D81):** 15 trials per tuned hparam with median pruning (50%
  warm-up). The HPO share of [compute.md](compute.md)'s ≈ 3,500 GPU-h (71% at 30
  trials, unpruned) drops to roughly 40% of what it was.
- **Batch size (D79, D82):** one batch size per study for every capacity; the
  full-MNIST studies micro-batch d3r40 (256) and d3r80 (128) through `archs:`, and
  analyse those runs in chunks of the same size. Gibbs (`sweep_purify.gibbs.batch_size`
  24) still runs out of memory on full MNIST: part of the Gibbs rework below.
- **AT (D80):** cw = 0, the lr is the only tuned hparam (done in the four AT
  studies). Warm AT tunes the lr in [1e-5, 1e-2] (D81). AT α grid
  {0, 0.01, 0.1} (D82).
- **Gibbs purification on full MNIST costs ≈ 40–80 GPU-h per run** (see below).
  Fix it first, or enable `analysis.sweep_purify` only where the paper reads it
  (`paper.yaml`'s `mnist_sweep_purify`: `mnist_nat` α = 0.01).
- Time one probe on a cluster node first. Training is limited by CPU speed, and
  the cluster's CPU threads may be slower than the laptop's.

## Efficiency (before the expensive studies; Phase 9 track, D30)

Measured in [compute.md](compute.md).

- **Gibbs purification is O(n²) per sweep.** `GibbsPurification.purify_snapshots`
  (`bm4tc/analysis/purification.py`) builds (batch × num_bins) candidates for each
  feature and evaluates each one with a full forward pass through all n sites.
  - Cost: MNIST 12×12 447 s per 24 points; full MNIST ≈ 30× that per batch, and it
    runs out of memory at Gibbs batch 24 on 8 GB.
  - Fix: cache the left and right environments for the batch, so a candidate
    costs L_{k-1} · A_k(x) · R_{k+1}, then update the left environment after
    feature k is resampled (the sampler already does this from left to right). That
    is about n× cheaper.
  - It is tensor-network maths, so it belongs in `ConditionalBornMachine` on
    tensorkrowch nodes. It must reproduce the current snapshots bit for bit (or to
    floating-point tolerance, with the seam re-pinned).
- **Python overhead dominates training and analysis.** Epoch time does not depend
  on the bond dimension; it grows with the number of sites × steps. One unit
  keeps one CPU core busy and leaves the GPU mostly idle (6 units per GPU = 4×
  throughput). Candidates: fewer, larger kernels per step (tensorkrowch
  stacking, CUDA graphs, `torch.compile`), and a larger `analysis.batch_size`
  (changes PGD's random draws, so it needs a re-pin).
- Minor: the Gibbs/SGLD analysis part logs a spurious
  `Detection/attack failed: ; skipping` at every budget (`bm4tc/analysis/uq.py:556`:
  `next(iter(det.values()))` on empty percentiles). Harmless; guard it.

## Phase 9 (separate track): tensorkrowch

Verify each upstream candidate in [decisions.md](decisions.md#tensorkrowch-upstream-candidates-tensorkrowch-116)
against the current tensorkrowch source; prototype as patches in a fork, run the
bm4tc suite against the fork, then propose upstream. Changes to the bm4tc numerical
core (`ConditionalBornMachine` contractions, `log_partition_function`, accumulate,
sampling) happen here and nowhere else (D30).

## Open items

- **Time series** (D6): datasets, grids and studies (NAT and AT; JEM if wanted).
- **Study settings** were copied from the old sweeps (epochs, patience, HPO spaces);
  review before Phase 8.
- **Trainer unification** for MPS and JEM: noted, not decided (see the note under
  the decisions table).
- **Adaptive attacks** a journal reviewer will ask for (rebuttal): BPDA/EOT through
  Gibbs purification, unrolled PGD through gradient purification. Not implemented.
- **Agent notes on the other laptop:** `CLAUDE.md`, `DEFERRED.md` and `.claude/`
  there may hold the old plan and issue list; move what still matters into GitHub
  issues or this file.
- `tests/unit/test_privacy.py::test_membership_inference_runs_end_to_end` fails when
  only `tests/unit tests/integration` run (order-dependent on the session `cbm`
  fixture); passes in the full suite.
