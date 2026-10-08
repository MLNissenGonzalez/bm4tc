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

1. Pilots that set defaults (D83). Pilot A done: n·ln d / 2 (D84, applied). Pilot B
   (`pilot_pgd10` + `pilot_pgd5`, PGD-5 vs 10, D43) running; apply its verdict to
   `configs/trainer/at.yaml`, and decide the AT clean-accuracy question with it.
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
- **HPO workers per cell = the launch's slots** (capped at the cell's trials). With
  ≥ 15 slots every trial of a cell starts at once: TPE never sees a finished trial
  and the pruner never acts, so the HPO is random search. Cap the workers per cell
  (e.g. at ~1/3 of the trials) before Phase 8; until then, give a study ~8 slots.
- `run <study> --cell X` stops at `select`, which needs every cell's HPO. Let
  select (and run) work on the launched cells only.
- **Cluster speed:** `bench_train_step` on G21G01 (GPU shared with 4 pilot units, busy
  node): 440 ms per NAT step vs ≈ 180 on the laptop, so ≈ 2.4x slower per unit;
  mostly the CPU (single-thread bound). Double the wall times in compute.md.
- Time one probe on a cluster node first. Training is limited by CPU speed, and
  the cluster's CPU threads may be slower than the laptop's.

## Beta, the dimension-corrected objective (implemented, D86; pilots to plan)

Implemented on 2026-10-08 (D86): the code optimises
$L_\beta = (1-\beta)\,L_\text{dis} + \beta\,L_\text{gen}/N$ in nats per variable,
logs `loss_dis` [nats/label], `loss_x` [nats/feature], `log_px` per feature, the
`norm/*` diagnostics per site, and divides the norm penalty by $N$. The theory is in
Martin's notes (`docs/interpolation.md`, untracked on purpose). What is left: E0, the
beta pilots (E2), then the Phase 8 ladder and AT grid from them.

Paper sentence (the translation table is in the theory note, §7):

> We weight the two terms by $\beta \in [0, 1]$,
> $L_\beta = (1-\beta)\,L_\text{dis} + \beta\,L_\text{gen}/N$, where $N = n + 1$ counts the
> modelled variables, so that $L_\beta$ is a weighted mean of per-variable negative
> log-likelihoods. Up to a positive factor this is the objective of [TPM], with
> $\beta/(1-\beta) = N\alpha/(1-\alpha)$; TPM's $\alpha \in \{0.01, 0.1, 0.5\}$ on 12×12
> MNIST are $\beta \approx \{0.59, 0.94, 0.993\}$.

### Evidence so far

- MNIST 12×12, pilot A (d3r40, target $n \ln d / 2$): test accuracy 98.1 / 76.2 / 73.8% at
  $\beta = 0$ / 0.94 / 1 ($\alpha$ = 0 / 0.1 / 1); test $L_\text{gen}$ +62 / -138 / -142 nats.
- MNIST 12×12, TPM (d3r20): classification "markedly degrades beyond $\alpha \approx 0.01$"
  ($\beta = 0.59$, clean accuracy > 95%); at $\alpha = 0.5$ ($\beta = 0.993$) 56%.
- Spirals (d10r6), TPM: clean accuracy near-perfect at every $\alpha$: no accuracy knee,
  only a density knee.

So the MNIST 12×12 accuracy knee lies in $\beta \in (0.59, 0.94)$. The density knee is
unmeasured. TPM's whole MNIST sweep sat at $\beta \ge 0.59$.

### Experiments

**E0. The valid/test gap of pilot A (before E2).** Martin (2026-10-08): the seed runs
converged; `objective/valid` and the train objective sit near their asymptote of about
-16 at $\alpha = 0.1$. But the test objective of the same runs is about -13.1, and at
$\alpha = 1$ valid $L_\text{gen}$ is -182 to -193 while test is -137 to -143: a gap of
about 45 nats per image, on splits drawn at random from one pool (`train_test_split`,
the scaler fitted on train). At $\alpha = 0$ valid and test $L_\text{dis}$ agree
(0.057 vs 0.067). Find the cause before reading any test $L_\text{gen}$: evaluate one
saved checkpoint on valid and test with the same code path. If valid gives -185 there,
the splits differ (or valid leaks from train); if it gives -140, the analysis path
(checkpoint load, log Z, chunks) differs from training-time validation.

**E2. Locate the knees on MNIST and test the correction** (to discuss). Pilot-style
studies: 1 seed, d3r20, `max_epoch` 100, target $n \ln d / 2$, analysis at
$\epsilon = 0.1$ only, no joint attack; grid
$\beta \in \{0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 1\}$ (a factor 3
in the weight ratio per step, 12 cells).
- E2a `pilot_beta12`, MNIST 12×12: lr-only HPO over `{log: [1e-4, 1e-2]}`, 6 trials, no
  pruning (random search, all trials at once); ≤ 110 unit-h, one night.
- E2b `pilot_beta28`, full MNIST: no HPO, the lr of E2a's cell at the same $\beta$
  (`hparams_from`, which needs $\beta$ implemented); 12 runs × ≤ 7 h, one night.
- E2c `pilot_beta_spirals`, d10r6, normal HPO: minutes.

Read per $\beta$: test acc, $L_\text{dis}$, $\bar\ell$, rob and purify at 0.1, detection
q5; plot against $\ln r(\beta)$ and against $\ln r(\alpha)$. Knees: where acc and
$\bar\ell$ cross the middle of their range; $\beta^*$: the best purified accuracy.
Decisions: the exponent (the shift between 12×12 and 28×28 is 0 for $p = 1$, +0.85 for
$p = \tfrac12$, +1.7 for $p = 0$, blurred by about ±0.5; keep $N$ unless both knees
shift by more than ~1); the Phase 8 ladder (≤ 7 cells, ≥ 2 between the knees); the AT
grid $\{0, \beta^*\}$ with pilot B. Prediction: with $p = 1$, D82's $\alpha = 0.01$ on
full MNIST ($\beta = 0.89$) is at or past the accuracy knee.

**E1 (optional, for the paper).** On E2 checkpoints, the per-sample ratio
$\mathbb{E}\lVert\nabla \log p(x)\rVert^2 / \mathbb{E}\lVert\nabla \log p(c \mid x)\rVert^2$
for 12×12 and 28×28 (the theory note §4.1 predicts ×5.4); depends on the MPS gauge, so
supporting only.

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

### Sampling code
Ask Martin to clone fork of tensorkrowch with development branch into 0git/. 
non-finished sampling implementation on tensorkrowch on the develop branch
- maybe resuse that implementation in tensorkrowch/models/mps/... sampling
- it tries to be general and apply to fully sample form the learned distribution to conditional sampling for things like imputation or purification. 
- look at the code, compare it with the sampling implementation in ConditionalBornMachine.
- create a comparison, what one code is capable of what the other does not.
- then discussion with questions on what to adapt from the tensorkrowch code and what not to 

fork: https://github.com/MLNissenGonzalez/tensorkrowch/tree/develop
## Phase 9 (separate track): tensorkrowch

Verify each upstream candidate in [decisions.md](decisions.md#tensorkrowch-upstream-candidates-tensorkrowch-116)
against the current tensorkrowch source; prototype as patches in a fork, run the
bm4tc suite against the fork, then propose upstream. Changes to the bm4tc numerical
core (`ConditionalBornMachine` contractions, `log_partition_function`, accumulate,
sampling) happen here and nowhere else (D30).

## Open items

- **randn_eye rescale misses the edge cores:** `ConditionalBornMachine.__init__` multiplies
  `self.tensors` by 1/φ₀, but the two edge entries are contracted copies (tensorkrowch
  candidates in decisions.md), so ψ is rescaled by φ₀^−(n−2), not φ₀^−n. Harmless (the norm
  penalty absorbs a constant); fixing it changes every initialisation, so it needs a re-pin.

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
