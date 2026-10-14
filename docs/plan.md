# Plan: where bm4tc stands and what is left

The `ousterhout` branch rebuilds bm4tc (main = the code behind the rejected NeurIPS
submission 27079, tag `pre-ousterhout`) for the journal version. The refactor
(Phases 0–7) is done; Phase 8 runs the paper's studies on the lab HPC and is in its
pilots. Every decision is in [decisions.md](decisions.md) (D1–D99, chronological, the
source of truth); the section below groups them by theme. Measured costs are in
[compute.md](compute.md).

## Now (2026-10-13, evening)

- **E2a is read** ([Pilots](#beta--mnist-1212-e2a-ran-2026-10-13-on-f9e7cb0c)):
  accuracy knee β ≈ 0.7, density knee β ≈ 0.03–0.1, β\* = 0.25 between them (on
  transfer numbers). **E2b** (`pilot_beta28`, full MNIST, E2a's lr per β, 150
  epochs) is written, not launched.
- **E2c is read** ([Pilots](#pilots-evidence-so-far)): no accuracy knee, a density
  knee at the bottom of the grid, a later purification knee, β\* ∈ [0.9, 1], and
  purified accuracy above the data-only ceiling.
- **Adaptive attacks and ceilings are in** (D96, D97, `docs/adaptive.md`): the
  `adaptive` analysis part, the selective ceiling, the population ceiling on spirals.
  A first **E3** run (`pilot_adaptive_spirals`, stopped) showed the detector attack
  working, but BPDA through likelihood purification (and Gibbs at 6 sweeps) weaker
  than PGD followed by the purifier (`docs/adaptive.md` §5). Fixed by starting the
  adaptive attack from the PGD example (D98); E3 trimmed (q5, Gibbs {1, 6}, EOT 8),
  not yet rerun.

Next, in order:

1. **Launch E2b** on G21G01 (a worktree at the current commit). The ladder waits
   for its knee shift; the AT grid {0, β\*} waits for E3's verdict on purification
   under adaptive attacks (below, E2a).
2. **The sampling code (D100, D101, D103): done.** `SamplingMPS`, develop's sampler on
   tk 1.1.6, keeps its nodes between calls; Gibbs and the figures use it; the
   tensorkrowch patch is a local branch, not pushed
   ([Sampling](#sampling-tensorkrowch-develops-sampler-on-116-d100)).
3. **Run E3** (trimmed, with D98; ~1.5–2 h on the eGPU, launch it detached so the
   terminal can close), read it against the ceilings, then the saturation ablation
   (steps, restarts, EOT) that sets `analysis.adaptive`'s defaults.
4. Before Phase 8 (below): cap HPO workers per cell, `select`/`run --cell` on a
   subset, a multi-study launcher.
5. Phase 8 studies.

## What the branch changed, by theme

- **Package and docs.** Three areas, `bm4tc/{core,analysis,pipeline}`; `core` imports
  neither of the others, `analysis` only `core` (D32, D68). README, GUIDE, AGENTS
  (D27, D77). Clean break from old run dirs and configs (D12). Tensor-network maths
  stays on tensorkrowch nodes; the numerical core changes only on the Phase 9 track
  (D30).
- **Study-based config tree.** `defaults.yaml` (every constant once), `dataset/`,
  `studies/` (one grid per dataset × regime × model, appendix variants as their own
  studies), `hparams/<study>.yaml` written only by `select` (D16, D19, D20, D29, D34,
  D36, D57); Hydra compose API and an enforced schema (D25, D56).
- **Runs found by manifest.** Each run writes `run.json`; dirs are study-rooted and
  named by identity axes only; nothing parses a path (D14, D17, D58). Checkpoints
  and `prune` (D22, D67); dataset caches named by their settings, created under a
  lock (D76, D78).
- **One CLI, one job pool.** `python -m bm4tc {hpo,select,train,analyse,run,status,
  figures,prune}` (D23, D33); every unit is a subprocess in a `--gpus × --per-gpu`
  pool (D38, D66). HPO: one Optuna journal per cell, 15 trials per tuned hparam,
  median pruning; `select` takes the argmin of `objective/valid` (D13, D21, D62, D81);
  CUDA failures and interrupted trials are rerun, not counted (D93, D95). W&B for
  live curves only (D39, D47). Runs on the lab HPC by hand, no agents there (D37).
- **One Trainer, one selection rule.** NAT is AT without the attack (D10, D15,
  D50, D52–D54); select on the objective on validation (D8); metric keys
  `quantity/split[/budget]`, built only in `pipeline/metrics.py` (D11, D48, D64);
  micro-batches for memory (D79).
- **β objective.** $L_\beta = (1-\beta)L_\text{dis} + \beta L_\text{gen}/N$ in nats per
  variable replaces α (D86); cold MNIST pulls log Z to n·ln d / 2 (D84). The ladder
  waits for E2.
- **The model.** complex64 and the production defaults (D4, D60); one log-space
  amplitude contraction (D89); frozen obc boundaries (D87; β > 0 runs before it
  optimised a corrupted objective); caches stamped by the parameters' version (D31,
  D55).
- **Adversarial training.** Split objective only (D18); PGD-AT with a curriculum
  (D40, D43, D61), PGD-5, MNIST AT 150 epochs (D88); no `clean_weight` (D80, D91);
  one budget grid for training and analysis (D3, D65).
- **Analysis.** Cached, separately seeded parts (D63); detection calibrated on valid
  (D2); data-only robustness ceilings for two classes, with rejection and on the
  spirals law (D97); adaptive attacks through purifier and detector (D96, D98); MIA out of
  the pipeline (D1).
- **JEM baseline** in the same pipeline, parameter-matched by `arch`, with SGLD
  purification (D24, D35, D69–D74).
- **Figures from paper manifests** (`configs/papers/`, D28, D75).
- **CUDA graphs and MPS for efficiency.** The training step, validation and the
  analysis's per-batch work replay from CUDA graphs; several units share a GPU
  through NVIDIA MPS (D90, D92, D94). ≈ 9–10× per unit, 4–6× per GPU (compute.md).
- **Pinning.** Seam tests pin end-point metrics and curves; changes re-pin in their
  own commit (D26, D51, D85).
- **Grids and pilots.** D5, D7, D9, D42, D44, D46, D59, D82, D83, D84, D88; the
  D45 α ladder and D82 α grid are superseded by β (D86), pending E2.

## Phase 8: reproduce the journal paper

Run the studies on the lab HPC, then `python -m bm4tc figures paper`. Warm studies
need their `warm_from` study trained.

1. Pilots that set defaults (D83): A (log Z target, D84) and B (PGD-5, D88) done;
   E2 (β) running, see below.
2. `spirals_nat`, `spirals_capacity`, `spirals_embedding`, `mnist12_nat`,
   `mnist_capacity`, `jem_mnist12_nat` (independent; cold).
3. `spirals_at`, `mnist12_at`, `jem_mnist12_at` (warm).
4. Set `arch` in `mnist_nat` / `mnist_at` from `mnist_capacity` (D44's two
   criteria), write `jem_mnist_{nat,at}` (copies of the `jem_mnist12_*` studies on
   `mnist`, `arch` = the chosen MPS), then run `mnist_nat`, `jem_mnist_nat`,
   `mnist_at`, `jem_mnist_at`.
5. `python -m bm4tc figures paper`; commit `figures/journal/`.

**To do before the studies run:**
- **β grids from E2:** the ladder in `defaults.yaml` (now {0, 0.01, 0.1, 0.5, 0.9,
  0.99, 1}) and the AT grid (now {0, 0.1, 0.5} in every AT study; planned {0, β\*}).
- **Cap HPO workers per cell.** Today workers = the launch's slots (capped at the
  cell's trials): with ≥ 15 slots every trial starts at once, TPE never sees a
  finished trial and the pruner never acts. Cap at ~1/3 of the trials; until then
  give a study ~8 slots. Also fixes E2c's 16/15 trials (parallel workers race on
  the count).
- **`select`/`run --cell` on a subset:** `run <study> --cell X` stops at `select`,
  which needs every cell's HPO.
- **A multi-study launcher** (`run paper`-style; the executor already runs a DAG,
  one launch runs one study, D66).
- **Analysis units need more GPU memory than training units**, but `--per-gpu`
  sizes every slot alike. E2a's relaunch (`--per-gpu 4`) lost 3 of 12 analyses to
  out of memory in a `uq` capture (log p(x), purification): an analyse unit held
  6.2–6.4 GiB beside trains at 3.9 GiB. E2b (full MNIST d3r40 micro 256, 2 units
  per GPU) lost its first 4 analyses and one train start the same way: a train holds
  ≈ 9.3 GiB per process (7.9 of it reserved by the allocator, unallocated), an
  analysis ≥ 10.7 GiB; the unit that starts next to a running one is the one that
  fails. Phase 8's `mnist_nat`/`mnist_at` would hit this on every launch. Options:
  measure the analysis peak in the bench and size on it, a per-stage `--per-gpu`, or
  free each part's captured graphs before the next part.
- **Staggered unit starts**, so a launch does not put every warm-up peak on a GPU at
  once; at E2a's launch two units on one GPU failed with "CUDA-capable device(s)
  is/are busy or unavailable" (MPS refusing clients that connect together; retried
  by D93).
- **Gibbs purification on full MNIST** (≈ 40–80 GPU-h per run, out of memory at
  batch 24): the rework under Efficiency, or enable `analysis.sweep_purify` only
  where the paper reads it (`paper.yaml`'s `mnist_sweep_purify`).
- **Full-MNIST d3r80 (micro 128) does not fit**: validation needs ≈ 4.8 GiB beside
  the step's 16 GiB pool. Smaller validation chunks or micro 64; `mnist_capacity`
  decides.
- **Adaptive attacks:** implemented (D96, D98); E3 and its saturation ablation set
  `analysis.adaptive`'s defaults (Now, step 3).
- **Re-estimate compute.md** with graphs, MPS and the HPO budget (D81 cuts the HPO
  share to ≈ 40% of the 3,500 GPU-h estimate's).
- **Review the study settings** copied from the old sweeps (epochs, patience, HPO
  spaces).

## Pilots: evidence so far

### Beta: the objective (D86)

The code optimises $L_\beta = (1-\beta)\,L_\text{dis} + \beta\,L_\text{gen}/N$ and logs
`loss_dis` [nats/label], `loss_x` [nats/feature], `log_px` per feature and the
`norm/*` diagnostics per site; the norm penalty is divided by $N$. The theory is in
Martin's note `docs/interpolation.md` (untracked on purpose). Paper sentence (the
translation table is in the note, §7):

> We weight the two terms by $\beta \in [0, 1]$,
> $L_\beta = (1-\beta)\,L_\text{dis} + \beta\,L_\text{gen}/N$, where $N = n + 1$ counts the
> modelled variables, so that $L_\beta$ is a weighted mean of per-variable negative
> log-likelihoods. Up to a positive factor this is the objective of [TPM], with
> $\beta/(1-\beta) = N\alpha/(1-\alpha)$; TPM's $\alpha \in \{0.01, 0.1, 0.5\}$ on 12×12
> MNIST are $\beta \approx \{0.59, 0.94, 0.993\}$.

**E2, the β pilots.** 1 seed, analysis at ε = 0.1 only, no joint attack; grid
β ∈ {0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 1} (a factor 3 in
the weight ratio per step).
- E2a `pilot_beta12`, MNIST 12×12 d3r40, 100 epochs, target n·ln d / 2, lr-only HPO
  `{log: [1e-4, 1e-2]}`, 6 trials, random search: **done**, below.
- E2b `pilot_beta28`, full MNIST: no HPO, the lr of E2a's cell at the same β
  (`hparams_from`; `configs/hparams/pilot_beta12.yaml` is the cluster's `select`
  output, tracked so a worktree has it), 150 epochs; 12 runs × ≤ 10 h. Written, not
  launched.
- E2c `pilot_beta_spirals`, d10r6, 300 epochs, normal HPO: **done**, below.

Read per β: test acc, $L_\text{dis}$, $L_x$, rob and purify at 0.1, detection q5,
against $\ln r(\beta)$ and $\ln r(\alpha)$. Knees: where acc and $L_x$ cross the
middle of their range (but see E2c: β = 0's $L_x$ is untrained); β\*: the best
purified accuracy. Decisions from E2: the exponent (the knee shift between 12×12 and
28×28 is 0 for $p = 1$, +0.85 for $p = \tfrac12$, +1.7 for $p = 0$, blurred by about
±0.5; keep $N$ unless both knees shift by more than ~1); the Phase 8 ladder (≤ 7
cells, ≥ 2 between the knees); the AT grid {0, β\*}. Prediction: with $p = 1$, D82's
α = 0.01 on full MNIST (β = 0.89) is at or past the accuracy knee.

### Beta / MNIST 12×12

- Pilot A (d3r40, target n·ln d / 2): test accuracy 98.1 / 76.2 / 73.8% at
  β = 0 / 0.94 / 1 (α = 0 / 0.1 / 1); test $L_\text{gen}$ +62 / −138 / −142 nats.
- TPM (d3r20): classification "markedly degrades beyond α ≈ 0.01" (β = 0.59, clean
  accuracy > 95%); 56% at α = 0.5 (β = 0.993).

So the accuracy knee lies in β ∈ (0.59, 0.94); the density knee is unmeasured (TPM's
sweep sat at β ≥ 0.59). E2a measures both.

### Beta / MNIST 12×12 (E2a, ran 2026-10-13 on `f9e7cb0c`)

Outputs in `pilots/cluster/pilot_beta12/` (untracked). G21G01, GPUs 5–7, 4 units per
GPU with MPS. ε = 0.1 (relative), purification radius 0.1, detection at the clean
5th percentile (false-positive rate 0.049–0.054: calibrated). $L_x$ in nats per pixel
(× 144 per image).

| β | lr | acc | L_dis | L_x | rob | det q5 | err_passed | purify | recovery |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.6e-4 | 0.980 | 0.074 | +0.334 | 0.282 | 1.000 | – | 0.558 | 0.385 |
| 0.001 | 1.7e-4 | 0.982 | 0.067 | +0.189 | 0.279 | 1.000 | – | 0.501 | 0.308 |
| 0.01 | 5.6e-4 | 0.978 | 0.080 | −0.071 | 0.334 | 0.999 | 0.63 | 0.620 | 0.430 |
| 0.03 | 1.7e-4 | 0.980 | 0.069 | −0.311 | 0.365 | 0.993 | 0.48 | 0.644 | 0.440 |
| 0.1 | 1.7e-4 | 0.977 | 0.076 | −0.574 | 0.454 | 0.966 | 0.74 | 0.790 | 0.616 |
| 0.25 | 1.7e-4 | 0.967 | 0.106 | −0.807 | 0.522 | 0.915 | 0.39 | 0.829 | 0.644 |
| 0.5 | 1.7e-4 | 0.946 | 0.181 | −0.900 | 0.475 | 0.856 | 0.37 | 0.777 | 0.578 |
| 0.75 | 5.6e-4 | 0.844 | 0.475 | −0.950 | 0.277 | 0.860 | 0.63 | 0.637 | 0.501 |
| 0.9 | 5.6e-4 | 0.843 | 0.497 | −0.968 | 0.261 | 0.867 | 0.52 | 0.633 | 0.506 |
| 0.97 | 5.6e-4 | 0.828 | 0.550 | −0.984 | 0.269 | 0.855 | 0.48 | 0.621 | 0.484 |
| 0.99 | 5.6e-4 | 0.793 | 0.640 | −0.990 | 0.275 | 0.835 | 0.42 | 0.597 | 0.448 |
| 1 | 5.6e-4 | 0.728 | 0.792 | −0.993 | 0.261 | 0.850 | 0.44 | 0.547 | 0.389 |

- **Accuracy knee at β ≈ 0.7** (the middle of 0.98 → 0.73), between 0.5 and 0.75;
  flat at 0.98 up to β = 0.1. As TPM (β ≈ 0.59) and pilot A at β = 1 (0.738). The
  selected lr flips from 1.7e-4 to 5.6e-4 at the same place, but it is not the cause:
  in the HPO trials at β ≥ 0.75 the lower lr is worse in objective and accuracy
  (0.807 vs 0.842 at 0.75), at β = 0.5 better (0.949 vs 0.868).
- **Density knee at β ≈ 0.03–0.1.** β = 0's $L_x$ is untrained (as E2c); $L_x$
  crosses the middle of the range from β = 0 at ≈ 0.03 and half its β = 1 value at
  ≈ 0.08; above 0.5 it gains only 0.09 nat per pixel.
- **β\* = 0.25, between the knees:** purified accuracy (0.829), recovery (0.644) and
  plain PGD robustness (0.522 vs 0.28 at β = 0) all peak there and fall back past the
  accuracy knee. Spirals' β\* ∈ [0.9, 1] came from having no accuracy knee. The top
  is flat (purify 0.79 / 0.83 / 0.78 at β = 0.1 / 0.25 / 0.5).
- **Detection says little at ε = 0.1:** the untrained density at β = 0 flags every
  attack, and detection falls to ≈ 0.85 as β grows. An L∞ step of 0.2 on every pixel
  is easy for any density to see; the attack does not try to keep log p high.
- **Caveats.** (1) Purification and detection are transfer numbers (the attack sees
  neither); at β = 0 purifying with an untrained density already lifts 0.28 to 0.56,
  so about half the purification number is the transfer gap, ≈ +0.27 is the density's.
  (2) The rise of rob at β = 0.25 (PGD-40, one restart) is not checked for
  saturation. (3) Every selected run with β ≥ 0.03 stopped at the 100-epoch cap with
  its best epoch at 97–100; at β ≥ 0.75 accuracy still climbs ≈ 1 point per 10 epochs
  (β = 1: 0.643 at epoch 50, 0.722 at 100), so the accuracy knee may move right a
  little. (4) The lr search drew the same 6 values in every cell (1.7e-4 twice), and
  1.7e-4, the lowest drawn, won in most cells.

Decided with Martin (2026-10-14): the read above; E2b at 150 epochs. The pilot does
not enter the paper, so its analysis is not strengthened for its own sake. One
decision depends on the transfer numbers and is costly to undo: β\* sets the AT
grid, i.e. AT training. It waits for E3 (spirals, against the ceilings): if
purification holds under BPDA with the PGD warm start, the AT grid is {0, β\*}; if
not, the `adaptive` part runs on a few E2a checkpoints (β = 0, 0.1, 0.25, 0.5; no
retraining) before the grid is fixed. A 12×12 ladder with ≥ 2 cells between the
knees would be {0, 0.01, 0.1, 0.25, 0.5, 0.9, 1}; the Phase 8 ladder waits for E2b's
knee shift.

### Beta / Spirals (E2c, ran 2026-10-09 on `d127eaad`)

Outputs in `pilots/cluster/pilot_beta_spirals/` (untracked). Captured training step,
eager validation: sound despite D94's bug (nothing released the stray buffers).
2000 test points, so one standard error is ≈ ±0.011 on a rate near 0.6. ε = 0.1 (relative;
0.2 absolute on legendre's [−1, 1]), purification radius 0.1, detection at the
clean 5th percentile.

| β | acc | L_dis | L_x | rob | det q5 | err_passed | purify | recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 0.0002 | 0.680 | 0.575 | 0.42 | 0.250 | 0.601 | 0.08 |
| 0.001 | 1.000 | 0.0002 | 0.463 | 0.546 | 0.50 | 0.255 | 0.579 | 0.08 |
| 0.01 | 1.000 | 0.0004 | 0.353 | 0.496 | 0.75 | 0.282 | 0.603 | 0.22 |
| 0.03 | 1.000 | 0.0007 | 0.336 | 0.488 | 0.78 | 0.333 | 0.602 | 0.23 |
| 0.1 | 1.000 | 0.0015 | 0.311 | 0.564 | 0.72 | 0.136 | 0.675 | 0.26 |
| 0.25 | 1.000 | 0.0036 | 0.296 | 0.469 | 0.85 | 0.226 | 0.633 | 0.32 |
| 0.5 | 1.000 | 0.0080 | 0.283 | 0.517 | 0.83 | 0.149 | 0.693 | 0.40 |
| 0.75 | 1.000 | 0.0160 | 0.273 | 0.543 | 0.80 | 0.050 | 0.759 | 0.49 |
| 0.9 | 0.9995 | 0.0215 | 0.265 | 0.532 | 0.81 | 0.011 | 0.809 | 0.61 |
| 0.97 | 0.9995 | 0.0254 | 0.267 | 0.512 | 0.80 | 0.008 | 0.815 | 0.65 |
| 0.99 | 0.9995 | 0.0268 | 0.266 | 0.515 | 0.80 | 0.010 | 0.810 | 0.64 |
| 1 | 0.9995 | 0.0312 | 0.258 | 0.542 | 0.81 | 0.005 | 0.818 | 0.64 |

The data-only ceiling is 0.676 at every β; the clean false-positive rate is
0.045–0.054 (q5 is calibrated); purifying clean inputs keeps accuracy at 1.0.

- **No accuracy knee** (as TPM): accuracy loses one test point; $L_\text{dis}$ grows
  150× but stays ≤ 0.03 nats.
- **The density knee is at the bottom of the grid:** $L_x$ crosses the middle of its
  range at β = 0.001 and has made most of its gain by 0.01. At β = 0 $L_x$ is
  untrained (its valid value wanders 0.50–0.69 over the epochs), so "the middle of
  the range" has no fixed low end.
- **Detection follows the density knee:** 0.42 → 0.75 by β = 0.01, ≈ 0.80 from 0.25.
- **Purification has a later knee:** 0.60 → 0.82, the middle near β ≈ 0.5–0.75,
  although $L_x$ only drops 0.31 → 0.26 from β = 0.1 to 1. β\* ∈ [0.9, 1] (0.809–0.818,
  within noise), where err_passed is also ≈ 1%.
- **rob has no trend in β** (0.47–0.58: several standard errors, from each cell's
  selected lr) and stays under the ceiling.
- **Purified accuracy (0.82) is above the ceiling (0.676).** Likelihood purification
  is deterministic (no random start), so "purify, then classify" is a classifier
  and the ceiling binds it against a worst-case attack. The PGD attacks the bare
  classifier only, so the purification numbers overstate robustness by ≥ 0.14 here.
  MNIST has no ceiling (two classes only) to show this.
- HPO: the sampler draws the same lr sequence in every cell (e.g. trial 13's
  0.01285 won at β = 0.01, 0.97, 0.99), so neighbouring cells can share curves.
  Four runs (β = 0.01, 0.1, 0.5, 0.9) stopped at the 300-epoch cap with their best
  epoch past 250.

To decide: the knee rule for E2a (proposed: within half of the β = 1 value, rather
than the middle of the range from β = 0); adaptive attacks before the AT grid,
since β\* is chosen on purified accuracy.

### Done and optional experiments

- **E0, done (2026-10-08):** the 45-nat valid/test $L_\text{gen}$ gap of the α
  seed runs was the evaluation path, removed by D87. One checkpoint evaluated with
  one code path gives train, valid and test within 0.7 nat; the logged gap equalled
  training-time log Z minus the checkpoint's (at α = 1, 4–9 nats unexplained, not
  chased).
- **E1, optional (for the paper):** on E2 checkpoints, the per-sample ratio
  $\mathbb{E}\lVert\nabla \log p(x)\rVert^2 / \mathbb{E}\lVert\nabla \log p(c \mid x)\rVert^2$
  for 12×12 and 28×28 (the theory note §4.1 predicts ×5.4); depends on the MPS
  gauge, so supporting only.

## Efficiency (Phase 9 track, D30)

The training step, validation and the analysis's per-batch work are captured
(D90, D92, D94); the measurements and memory per unit are in compute.md. What is
left:

- **The forward stores B·n·r² (noted, not scheduled).** `log_amp_sq` (training,
  validation, log p(x)) runs tk 1.1.6's `contract`: every site is contracted with its
  data first, into a (B, r, r) matrix, and all are kept (autograd keeps them for the
  backward). Measured: 153 MiB per point at 784 sites, r160 (exactly n·r²·8 bytes),
  38 MiB per point at r80 with 10 classes. Environment first, as develop's `norm`
  zips up and the Gibbs sweep does (D101): the running vector absorbs the site
  (B, d, r), then the data (B, r), so B·n·r·d is stored, ≈ 27× less at r80, d 3; the
  time stays B·r²·d per site. It may be what puts d3r80 micro 128 out of memory.
  Core path (D30): the bar below applies.
- **The forward without `reset` (Martin, 2026-10-14; prototyped, decision pending).**
  `ConditionalBornMachine.forward` resets before and after every call: tk's `forward`
  replays `_seq_ops` once result nodes exist, which would skip the Python-side
  `_accumulate_log_norm` in `_inline_contraction`. Calling `contract(renormalize=True,
  inline_mats=True)` directly, with no reset, runs the Python code every call (so the
  accumulator too) while every operation reuses its successor (`div` by a tensor is
  keyed on the node alone). Prototype (scratch): `log_amp_sq` and every gradient
  bit-identical to the reset path over changing batch sizes and optimizer steps, CUDA
  and CPU; no node added after the first call. Without more, one call's intermediates
  stay held between calls (254 MiB at MNIST 12×12 d3r20 B 512, 645 MiB at full MNIST
  d3r40 B 64) and the peak doubles; with tk's trace bookkeeping (first call under
  `_tracing`, then `_traced`, so later calls free each intermediate after its last
  use) nothing is held and the full-MNIST peak drops (674 → 612 MiB). RTX 2080:
  no-grad forward 35 → 13 ms (MNIST 12×12 d3r20 B 512), 236 → 78 ms (full MNIST
  d3r40 B 64); forward + backward 94 → 56 ms (MNIST 12×12 B 512). Captured steps run
  no Python, so the gain is in eager work (validation outside graphs, analysis, log
  p(x), CPU). The tk renormalisation node is not needed. Open for the change: the
  hand-over between the forward's nodes and those `sample` keeps (one structure key
  for the network), `test_graphs.py` and the seams, the callers that reset (OOM
  recovery, `initialize`, the Trainer).
- **Gaps D94 left:** no test pins memory per unit at a study's shape (the bench does,
  by hand); replays do not bump version counters, so `_params_key()` is stale after
  a capture, and an eager `log_normalizer()` or `log_Z(recompute=False)` during
  captured training would serve an old log Z (nothing does today). Before each
  launch, a cluster check from the study's worktree: `pytest tests/unit/test_graphs.py`
  plus `bench_train_step` at the study's shape.
- **Then GPU time, in order:**
  1. A custom `torch.autograd.Function` for the renormalised chain (and the log Z
     zip-up): the forward saves the per-site vectors and norms, the backward is one
     hand-written reverse sweep. Removes most of the backward's launches; same
     algorithm, so it should match to float rounding.
  2. Fewer operations per site: renormalise every k sites (k bounded by overflow),
     drop the `where`/`masked_fill` guards where the norm cannot be zero.
  3. The renormalising division on a real view times 1/n (≈ 2× on that kernel, not
     bit-identical). `torch.compile` is not an option on torch 2.1 with complex
     tensors.

  A custom Function works on raw tensors, outside tensorkrowch's nodes: under D30 it
  belongs in tensorkrowch, as a renormalize that returns the log norm (tk's
  `renormalize` keeps gradients but discards the norm, so log Z computes each norm
  twice), unless Martin makes an exception.
- **Ruled out** (prototypes, 2026-10-08; both match log|ψ|² to ≈ 2e-6): a log-depth
  tree contraction with per-level renormalisation (1.5× faster at d3r20, 2.6×
  slower at d3r40: matrix-matrix r³ against the chain's r²); all transfer matrices
  from one einsum, then a matvec chain renormalised every 8 sites (3.6–14× slower:
  backward through slices of one tensor writes a full-size gradient per slice).
- **The bar** (D30): the new contraction matches log|ψ|², log Z and their gradients
  on random models (overflowing ones included) to float tolerance; the seams re-pin
  in their own commit; the gain is measured with `bench_train_step` and
  `profile_train_step` before and after, from a `git worktree` of HEAD.
- Minor: Gibbs/SGLD analysis parts log a spurious `Detection/attack failed: ;
  skipping` at every budget (`bm4tc/analysis/uq.py:611`, `next(iter(det.values()))`
  on empty percentiles). Harmless; guard it.

## Sampling: tensorkrowch develop's sampler on 1.1.6 (D100)

Compared on 2026-10-14: develop's `sample(in_condition=x)` is one Gibbs sweep (O(n)),
the same distributions as `GibbsPurification` (O(n²·bins)). `SamplingMPS`
(`bm4tc/core/sampling.py`) carries develop's `condition` and `sample` to tk 1.1.6;
`ConditionalBornMachine` subclasses it. Steps:

1. **Done:** the port as it is, develop's tests, the CBM checks
   (`tests/unit/test_sampling.py`).
2. **Done:** efficiency. (a) a domain per sample (the windows); (b–d) at each
   sampled site, everything but its physical edge is contracted first (the site's
   reduced density matrix), candidates are quadratic forms on it; ket and bra stay
   single-layer chains wherever the variables are fixed; the bra layer is built in
   the network (develop's copy-and-move renamed nodes quadratically). One windowed
   sweep, 96 bins, RTX 2080 (`GibbsPurification` → `SamplingMPS.sample`):

   | shape | before | after |
   |---|---|---|
   | spirals, B 256 | 0.02 s, 910 MiB | 0.010 s, 7 MiB |
   | MNIST 12×12 r40, B 24 | 6.85 s, 4.3 GiB | 0.71 s, 137 MiB |
   | MNIST 12×12 r40, B 256 | – | 0.74 s, 1.4 GiB |
   | MNIST 12×12 r40, B 24, 1024 bins | – | 0.77 s, 322 MiB |
   | MNIST 28×28 r40 / r80, B 24 | out of memory | 6.5 s / 6.7 s, 0.7 / 1.3 GiB |

   Time is tensorkrowch's Python overhead per operation (≈ 0.2 ms, ~25 operations
   and ~30 new nodes per site), not batch or bins; `reset` deleted nodes in time
   quadratic in nodes of one name (2.3 s of 8.4 s profiled at 784 sites). Step 4
   keeps the nodes instead.
3. **Done (D101):** `GibbsPurification` runs each sweep as one `born.sample(in_condition=x,
   domain=windows)`, the figures' class samples are `condition` + `sample`, the old
   sampler is deleted; rows with no usable weight are drawn uniformly with one
   warning per call; `gibbs.batch_size` 64. The Gibbs golden reproduces unchanged.
   A sweep (`in_condition`) keeps the ket only: with every variable but the sampled
   one fixed, the density is ρ = Σ_c w_c w_c† from the ket and its conjugate, so no
   bra layer and memory per point linear in r (784 sites, 10 classes: 17.5 / 30.5 /
   ≈ 57 MiB at r40 / r80 / r160). One sweep over 512 points at batch 64: MNIST 12×12
   r40 3.95 s, full MNIST r40 / r80 29 s (1.1 / 2.7 GiB peak with the log p(x)
   forward).
4. **Done (D103):** measured before (`3b6afd73`, before D100) and after, from
   worktrees (compute.md, "Gibbs purification"), and `sample` keeps its nodes
   between calls: tensorkrowch caches each operation's result node as a successor
   of its operands, so a repeated sweep on the same nodes only recomputes tensors
   (no new nodes, no reset; ≈ 60% of a sweep was building nodes and resetting).
   10 sweeps, batch 64: MNIST 12×12 r40 18.6 → 8.6 s, full MNIST r40 / r80
   69.4 → 25.4 s / 69.5 → 26.1 s (128 points), same draws.
5. **Done:** the develop patch, branch `sampling-efficiency` in `../tensorkrowch`
   (local, not pushed), one commit per bm4tc step (2a, 2b–2d, step 3, D103), each
   `sampling.py` translated back to develop's names (the translation of step 1
   reproduces develop exactly) with bm4tc's tests in develop's style: `0a079fa`,
   `9a08f59`, `a23003e`, `e514b02`. tensorkrowch's `tests/` in the bm4tc env (torch
   2.1): only the failures develop has there too (`to`, `to_contracted`,
   deparameterize; `entropy` and one canonicalize test fail at random on develop as
   well); doctests pass. Documented in `e514b02`, not changed (Martin: sampler only): connecting
   through a result node's inherited edge connects the leaf (`reattach_edges`
   first); `@` refuses batch-only contractions (`einsum` works); `einsum` needs
   repeated non-batch subscripts to be connected edges; duplicate axis names
   become `input_0`/`input_1`; moving nodes between networks and `reset` rename in
   time quadratic in same-named nodes. Once a tensorkrowch release has it,
   `SamplingMPS` can go (bm4tc stays on 1.1.6 until then).

Out of scope until decided: finer grids or exact continuous draws, a
Metropolis–Hastings correction for the windows, temperature (they change the
purifier).

## Phase 9 (separate track): tensorkrowch

Verify each upstream candidate in [decisions.md](decisions.md#tensorkrowch-upstream-candidates-tensorkrowch-116)
against the current tensorkrowch source; prototype as patches in a fork, run the
bm4tc suite against the fork, then propose upstream. Changes to the bm4tc numerical
core (`ConditionalBornMachine` contractions, `log_partition_function`, accumulate,
sampling) happen here and nowhere else (D30).

## Ideas for the benchmark project (not for this paper)

Kept here until they move to `~/0git/tn4dd/` (benchmarking ways to train tensor
networks generatively).

- **Hard renormalisation during training.** The removed `norm_control.hard_every`
  (D90) rescaled all cores every k steps so log Z returned to the target, a
  projection against the soft penalty strength·(log Z − target)²/N; it does not fit a
  captured step. `renormalize_` stays (figures use it). Compare hard projection,
  the soft penalty and neither (log-space accumulation only) on overflow, stability
  and the generative NLL.
- **DMRG sweeps** (from the personal laptop's `DEFERRED.md` §1).

## Open items

- **Adaptive attacks** a reviewer will ask for: implemented for likelihood
  purification, Gibbs/SGLD and the detector (D96), started from the transfer attack
  (D98); E3 to run (`docs/adaptive.md` §5).
- **Time series** (D6): datasets, grids and studies (NAT and AT; JEM if wanted).
- **Trainer unification** for MPS and JEM: noted, not decided (decisions.md, under
  the table).
- **randn_eye rescale misses the edge cores:** `ConditionalBornMachine.__init__`
  multiplies `self.tensors` by 1/φ₀, but the edge entries are contracted copies, so
  ψ is rescaled by φ₀^−(n−2), not φ₀^−n. Harmless (the norm penalty absorbs a
  constant); fixing it changes every initialisation, so it needs a re-pin.
- **main's spirals AT `rob` above the ceiling** (0.929 / 0.829 against 0.679 / 0.609
  at abs 0.2 / 0.3; real-dtype runs from before D87; AGENTS.md's laptop comparison):
  check on `spirals_at` once it runs. E2c's NAT `rob` is below the ceiling.
- **The Chebyshev comment** at `bm4tc/core/embeddings.py:250` cites
  `reference/chebyshev_boundary_born_machines.md`, which exists only untracked on
  the personal laptop.
