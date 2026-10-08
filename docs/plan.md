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

## Planned: β, a dimension-corrected objective (option C; to assess, then implement)

Martin's choice (2026-10-08): replace α by **one** parameter β everywhere (studies,
trainers, metrics, figures), with the generative term in **nats per feature**. Written
up for a fresh session to **critically assess the reasoning first** (the open
questions at the end), then implement. Nothing below is decided until that review;
record the outcome as a D-number. **Reviewed 2026-10-08 in [beta.md](beta.md):** the
verdict on the six questions, an alternative to the rescaled loss (C′), and the
experiments E0–E2 that fix the ladder.

### Why α is not comparable across datasets

Notation: x has n features (data sites), c is the class; the MPS has N = n + 1 sites.
L_dis = −log p(c|x), L_gen = −log p(x, c), L_marg = −log p(x). Today

    L(α) = (1−α)·L_dis + α·L_gen.

Since p(x, c) = p(x)·p(c|x), **L_gen = L_marg + L_dis**, so exactly

    L(α) = L_dis + α·L_marg.

α weighs the marginal density against the classifier. By the chain rule over
features, L_marg = Σ_{i=1..n} −log p(x_i | x_<i): a sum of n per-feature terms. L_dis
is a single term of order 1 (≤ log C at initialisation). Their gradients behave alike:
∇L_marg is a sum of n per-feature gradients, so its norm grows with n (linearly if the
per-feature gradients add coherently, like √n if they are independent). The two terms
balance where α·|∇L_marg| ≈ |∇L_dis|, i.e. **α\* ∝ 1/n** (between 1/n and 1/√n):
α\* ≈ 0.3 on spirals (n = 2), ~7e-3 on MNIST 12×12 (n = 144), ~1e-3 on full MNIST
(n = 784), up to the unknown constant. Evidence: pilot A (D84), MNIST 12×12 d3r40, test
accuracy 98% at α = 0 but 77% at α = 0.1, i.e. α = 0.1 is already almost purely
generative there, while on spirals α = 0.1 is a moderate mix. (Comparing loss *values*,
~140 nats vs ~0.07, is not the argument: values carry constants with no gradient, and
L_marg is negative here. The scaling argument is about gradients.)

### The proposal

    L_β = (1−β)·L_dis + β·L_gen / N,   β ∈ [0, 1].

L_gen / N is the generative NLL in **nats per feature** (bits/dim, as generative models
report it). Substituting L_gen = L_dis + L_marg:

    L_β = (1 − β + β/N)·L_dis + (β/N)·L_marg = (1 − β + β/N) · L(α(β)),

    α(β) = β / (N − β(N−1)),     β(α) = α·N / (1 + α(N−1)).

- **The endpoints are kept:** β = 0 ⇔ α = 0 (discriminative), β = 1 ⇔ α = 1 (generative).
- **The middle is dimension-corrected:** β = ½ ⇔ α = 1/(N+1), the classifier loss
  weighs as much as the mean per-feature generative NLL, on every dataset.
- **Optimisation:** L_β is a positive multiple of L(α(β)), so with Adam (nearly scale
  invariant) and a tuned lr it trains like α(β). The only thing that changes in
  practice is the scale of the loss, which matters for anything added to it (the norm
  penalty, below) and for logged values.

The current α ladder read in β (one α means very different things per dataset):

|α|MNIST 12×12 (N = 145)|full MNIST (N = 785)|spirals (N = 3)|
|---|---|---|---|
|1e-5|0.0014|0.0078|3e-5|
|1e-4|0.014|0.073|3e-4|
|1e-3|0.13|0.44|0.003|
|1e-2|0.59|0.89|0.03|
|1e-1|0.94|0.99|0.25|

Proposed **one ladder for every dataset: β ∈ {0, 0.01, 0.1, 0.5, 0.9, 0.99, 1}**
(spirals then needs no ladder of its own, D84); AT grid e.g. β ∈ {0, 0.1, 0.5}.
Figures put all datasets on one β axis.

### Implementation (once the review agrees)

A clean break (D12): `alpha` disappears from code, configs, names and keys. Nothing of
Phase 8 has run except the pilots, which stay as they are (α runs).

1. `ConditionalBornMachine.mixed_nll(x, y, beta)`: with the current terms,
   L_β = −(1−β+β/N)·log|ψ(x,c)|² + (1−β)·log Σ_c |ψ(x,c)|² + (β/N)·log Z
   (keep the endpoint gating: β = 0 never calls log Z; β = 1 drops the Σ_c term).
2. `objective.mix(dis, gen, beta)` → (1−β)·dis + β·gen/N; `evaluate` and the logged
   `objective/*` follow. Metric keys (`pipeline/metrics.py`): decide whether `loss_gen`
   becomes per feature or a new key (e.g. `nll_gen_per_feature`) sits beside it.
3. MPS `Trainer` AT path: (1−β)·L_dis(x_adv) + (β/N)·L_gen(x) (cw = 0, D80); the
   two-forward trick `s·mixed_nll(x, α/s)` needs re-deriving for β.
4. JEM (`core/jem/train.py`, `mix`): the same form; N = data_dim + 1 for both models
   (one definition of "feature" across models).
5. `TrainConfig.alpha` → `beta`; `Cell.alpha` → `beta`, cell names `b0.5` (D14
   prefixes); study grids, `defaults.yaml` ladder, AT grids; `paper.yaml`
   (`x: alpha`, `where: {alpha: …}`); `results.csv` identity column; figures' axis label.
6. **Norm control:** the soft penalty strength·(log Z − target)² is added to the loss,
   whose scale changes by (1−β+β/N) (≈ 1/N near β = 1). Either scale the penalty by the
   same factor (keeps today's strengths' meaning) or re-tune; pilot A's strengths
   were set in α units.
7. Tests: every α in tests (≈ 150 occurrences, mostly `test_trainer.py`, `test_cbm.py`,
   `test_figures.py`); a unit test that L_β = (1−β+β/N)·L(α(β)) numerically, and the
   endpoints. **Re-pin the seams** in a commit of its own (objective scale changes).
8. Docs: GUIDE (objective, vocabulary), README formula, decisions superseded: D45/D84
   ladder, D82 AT grid, D71 JEM objective form, D80 (objective form).

### For the reviewing session: check the reasoning critically

1. **Is per-feature normalisation the right balance?** The derivation shows α\* ∝ 1/n
   only up to a constant and between 1/n and 1/√n. Measure it: |∇L_dis| and
   |∇L_marg| (parameter-gradient norms) on spirals, MNIST 12×12 and full MNIST, at
   initialisation and on a trained NAT model. If the ratio scales like √n, normalise by
   √N instead, or say plainly that β only removes the leading dimension dependence.
2. **N or n?** The class site is a site of the MPS but not a feature of x. N keeps
   β = 1 ⇔ "mean NLL per modelled variable"; n is what "per feature" says. Pick one and
   use it for both models.
3. **Scale invariance:** Adam is invariant to the loss scale only up to ε; the norm
   penalty is not (item 6). Is anything else scale-dependent (the patience threshold,
   gradient clipping, the collapse diagnostics, the AT objective/valid selection)?
4. **JEM:** its generative term is a contrastive estimate without log Z; per-feature
   normalisation is still meaningful for its gradient, but its logged value is not a
   true NLL.
5. **The paper:** TPM used α. Reporting β in the journal needs one sentence of
   translation (the formula above) and maybe a table of the TPM α values in β.
6. **Ladder:** do {0, 0.01, 0.1, 0.5, 0.9, 0.99, 1} cover the interesting region for
   full MNIST (where pilot-A-like accuracy drops start)? A quick spirals/MNIST 12×12
   sweep over β before Phase 8 would show it.

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
