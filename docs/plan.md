# Plan: where bm4tc stands and what is left

The `ousterhout` branch rebuilds bm4tc (main = the code behind the rejected NeurIPS
submission 27079, tag `pre-ousterhout`) for the journal version. The refactor
(Phases 0–7) is done; Phase 8 runs the paper's studies on the lab HPC and is in its
pilots. Every decision is in [decisions.md](decisions.md) (D1–D110, chronological, the
source of truth); the section below groups them by theme. Measured costs are in
[compute.md](compute.md).

## Now (2026-10-13, evening)

- **E2a is read** ([Pilots](#beta--mnist-1212-e2a-ran-2026-10-13-on-f9e7cb0c)):
  accuracy knee β ≈ 0.7, density knee β ≈ 0.03–0.1, β\* = 0.25 between them (on
  transfer numbers).
- **E2b is read** ([Pilots](#beta--full-mnist-e2b-trained-on-05c44cf8-analysed-on-da7185e8),
  2026-10-15): accuracy knee β ≈ 0.68, a shift of ≈ −0.25 from 12×12, so p = 1 (keep
  N). The density knee is unreadable (early stopping at low β). E2a's lr does not
  transfer: 5.6e-4 fails at β = 0 *and* β = 0.01 (0.884, 0.861), and 1.7e-4 reaches
  0.98, then blows up around epoch 45. On 2026-10-14, β = 0 was kept as a result, with
  the generative term read as a regulariser. That rested on β = 0.01 having "trained
  well" at the same lr, which the analysis contradicts. Decided: keep N, the ladder
  {0, 0.01, 0.1, 0.25, 0.5, 0.9, 1} (D108), full MNIST tunes its own lr, the
  regulariser reading dropped.
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

1. **The cluster check of D107 + D110: passed (2026-10-15)**, run by the agent over
   SSH (D109) on G21G01 GPU 5 at `5e41a7d3`: `test_graphs` 19/19; every shape fits,
   full MNIST d3r40 AT at 10.3 GiB per process (out of memory before), d3r80 micro
   128 NAT 13.6 / AT 19.9 GiB; the 784-site analysis gives the same numbers captured
   and eager. Numbers and `--per-gpu` per study shape: compute.md, "Cluster check of
   D107 + D110". D94's rule is met: AT may run with `trainer.cuda_graph` from
   `d2aca0e5` on. Open: `analysis.batch_size` for full MNIST (the default 256 takes
   ≈ 12 GiB per analysis, so `--per-gpu` 1; 64–128 allows 2). E2b's decisions are taken (D108). D104's cluster checks passed
   on G21G01 (`test_graphs` 16/16; the 784-site analysis gives the same numbers
   captured and eager, 53 s vs 178 s). The AT grid {0, β\*} waits for E3's verdict on
   purification under adaptive attacks (below, E2a).
2. **The sampling code (D100, D101, D103) and the reset-free forward (D104): done.**
   `SamplingMPS`, develop's sampler on tk 1.1.6, keeps its nodes between calls; Gibbs
   and the figures use it; the tensorkrowch patch is branch `sampling-efficiency` on
   Martin's fork. D104 needs its cluster checks (AGENTS.md) before a launch
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
  live curves only (D39, D47). Runs on the lab HPC by hand, no agents there (D37);
  the agent drives it from the lab laptop over SSH with the same commands (D109).
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
- **β grids from E2:** the ladder in `defaults.yaml` is set ({0, 0.01, 0.1, 0.25,
  0.5, 0.9, 1}, D108); the AT grid (now {0, 0.1, 0.5} in every AT study; planned
  {0, β\*}) waits for E3.
- **Cap HPO workers per cell.** Today workers = the launch's slots (capped at the
  cell's trials): with ≥ 15 slots every trial starts at once, TPE never sees a
  finished trial and the pruner never acts. Cap at ~1/3 of the trials; until then
  give a study ~8 slots. Also fixes E2c's 16/15 trials (parallel workers race on
  the count).
- **`select`/`run --cell` on a subset:** `run <study> --cell X` stops at `select`,
  which needs every cell's HPO.
- **A multi-study launcher** (`run paper`-style; the executor already runs a DAG,
  one launch runs one study, D66).
- **A one-command deploy, outside bm4tc (D109; planned later).** Today the agent
  runs `docs/hpc_bm4tc.md`'s commands over SSH (AGENTS.md, "Environment"; a probe
  with screen, env and exit code passed on G21G01 on 2026-10-15). A "deploy" button
  would help: a worktree of a pushed commit (`runs/<sha>`, clean sha in `run.json`),
  launched in a named screen, reporting `status` and the failed units' logs. **It
  belongs in a general tool, not in bm4tc**: bm4tc stays independent of any HPC or
  launcher; it needs only a git checkout of a fixed commit and `BM4TC_DATA_ROOT`.
  The candidate is a `deploy` mode in hpc-greyc's `cluster-run`, which today copies
  a script and its local modules into a job directory (fine for one-off scripts,
  unsuitable for studies).
- **Analysis units need more GPU memory than training units**, but `--per-gpu`
  sizes every slot alike. E2a's relaunch (`--per-gpu 4`) lost 3 of 12 analyses to
  out of memory in a `uq` capture (log p(x), purification): an analyse unit held
  6.2–6.4 GiB beside trains at 3.9 GiB. E2b (full MNIST d3r40 micro 256, 2 units
  per GPU) lost its first 4 analyses and one train start the same way: a train holds
  ≈ 9.3 GiB per process (7.9 of it reserved by the allocator, unallocated), an
  analysis ≥ 10.7 GiB; the unit that starts next to a running one is the one that
  fails. Phase 8's `mnist_nat`/`mnist_at` would hit this on every launch. Since
  D105 the attack and purification are captured per step: at 784 sites, batch 64,
  the bench peaks at 2.55 / 5.05 GiB reserved captured at batch 32 / 64 (0.77 /
  1.52 GiB eager) on the RTX 2080, ≈ 0.08 GiB per sample, each step graph keeping a
  pool of its own; `pilot_beta28` analyses at batch 64. Graphs that never run at
  once (attack step, purification step, log p(x), prediction) could share one pool
  (`torch.cuda.graph(pool=...)`; their replays already return copies). **D107 does
  this** (one pool per `Graphs`): 2.99 GiB at batch 64 (eager 1.52); the analysis
  keeps per-shape warm-ups (D110), so ≈ 2× eager. On G21G01 (2026-10-15): 2.99 GiB
  reserved at batch 64, 10.38 at 256 (eager 1.52, 7.59), 394 vs 170 s for 2048
  points; `--per-gpu` per study shape in compute.md. Still open: a per-stage
  `--per-gpu`, or a smaller `analysis.batch_size` for full MNIST, since an analysis
  at batch 256 is larger than any d3r40 train.
- **Graph memory: reserved far above allocated, worst for AT: fixed by D107 + D110,
  checked on G21G01 (2026-10-15; compute.md, "Cluster check of D107 + D110").** One
  transient is left: at the first capture, the epoch-1 eager tensors held by tk's
  nodes coexist with the new pool (full MNIST NAT 7.80 GiB peak reserved vs 5.49 at
  the end; d3r80 NAT 16.04 vs 11.51). Dropping those references before the first
  capture would remove it. Per-pool accounting (`bench_train_step --snapshot`)
  showed reserved = the sum of private pools (D107: one pool per `Graphs`) and eager
  warm-ups after a capture adding their peak to the pool's (D110: the Trainer warms
  everything up in epoch 1, then captures). eGPU: full-MNIST AT micro 128 5.92 GiB
  reserved (eager 5.88), NAT chunk 256 5.50 (eager 5.34). Before, D104's check on G21G01 at `da7185e8` (D104 + D105),
  `bench_train_step`, β 0.5, batch 512, per process (nvidia-smi):
  MNIST12 d3r40 NAT 4.57 GiB (1.82 allocated at peak, 4.24 reserved; E2a measured
  3.89 before D104), AT 10.92 GiB (2.76 allocated, 10.42 reserved); full MNIST
  d3r40 micro 256 NAT 10.46 GiB (5.01 / 9.24), AT out of memory in the first
  validation's attack warm-up (3.63 GiB allocated, 15.33 reserved but unallocated).
  Suspects: one private pool per graph (D92) holds each graph's peak at once, and
  D105 added graphs (the attack step in training, another in validation at the
  valid batch shape) where AT's step was one graph; warm-ups on the side stream;
  D104's +0.8 GiB. To measure first: the same bench at `f56365de` (D104 without
  D105), then a shared pool for the graphs of one `Graphs` (they never run at once;
  inputs are copied in, outputs cloned out), checked with D94's allocation tests
  (tk's kept nodes, D104, may hold capture-time memory). Measured instead per pool;
  the attribution to D104 vs D105 was not needed.
- **Staggered unit starts**, so a launch does not put every warm-up peak on a GPU at
  once; at E2a's launch two units on one GPU failed with "CUDA-capable device(s)
  is/are busy or unavailable" (MPS refusing clients that connect together; retried
  by D93).
- **Gibbs purification on full MNIST** (≈ 40–80 GPU-h per run, out of memory at
  batch 24): the rework under Efficiency, or enable `analysis.sweep_purify` only
  where the paper reads it (`paper.yaml`'s `mnist_sweep_purify`).
- **Full-MNIST d3r80 (micro 128) did not fit**: validation needed ≈ 4.8 GiB beside
  the step's 16 GiB pool, exactly what D107 + D110 remove. Recheck in the cluster
  check; `mnist_capacity` decides.
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
  output, tracked so a worktree has it), 150 epochs; 12 runs × ≤ 10 h: **done**, below.
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

### Beta / full MNIST (E2b, trained on `05c44cf8`, analysed on `da7185e8`)

Outputs in `pilots/cluster/pilot_beta28/` (untracked). G21G01, 2 units per GPU with
MPS; analyses at batch 64, ≈ 28 min each. Same settings as E2a (ε = 0.1 relative,
purification radius 0.1, q5 calibrated: false-positive rate 0.046–0.054), no HPO:
each cell takes E2a's lr at the same β. Patience 50, cap 150 epochs. $L_x$ in nats
per pixel (× 784 per image). "valid acc" is at epochs 50 / 100 / last.

| β | lr | best/last | acc | L_dis | L_x | rob | det q5 | err_passed | purify | recovery | valid acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.6e-4 | 35/86 | 0.884 | 0.371 | +0.440 | 0.042 | 0.969 | 0.99 | 0.718 | 0.706 | 0.84 / – / 0.80 |
| 0.001 | 1.7e-4 | 41/92 | 0.979 | 0.073 | +0.542 | 0.091 | 1.000 | – | 0.530 | 0.483 | 0.86 / – / 0.92 |
| 0.01 | 5.6e-4 | 22/73 | 0.861 | 0.449 | +0.073 | 0.045 | 0.899 | 0.99 | 0.679 | 0.664 | 0.82 / – / 0.77 |
| 0.03 | 1.7e-4 | 23/74 | 0.979 | 0.082 | +0.414 | 0.142 | 1.000 | – | 0.570 | 0.499 | 0.96 / – / 0.93 |
| 0.1 | 1.7e-4 | 41/92 | 0.978 | 0.082 | −0.013 | 0.128 | 0.981 | 0.90 | 0.753 | 0.716 | 0.94 / – / 0.96 |
| 0.25 | 1.7e-4 | 115/150 | 0.915 | 0.271 | −0.840 | 0.018 | 0.828 | 0.99 | 0.665 | 0.659 | 0.88 / 0.91 / 0.87 |
| 0.5 | 1.7e-4 | 135/150 | 0.886 | 0.379 | −1.002 | 0.050 | 0.831 | 0.98 | 0.681 | 0.664 | 0.90 / 0.88 / 0.87 |
| 0.75 | 5.6e-4 | 72/123 | 0.688 | 0.916 | −1.132 | 0.072 | 0.754 | 0.94 | 0.468 | 0.427 | 0.67 / 0.66 / 0.65 |
| 0.9 | 5.6e-4 | 94/145 | 0.651 | 0.996 | −1.147 | 0.110 | 0.783 | 0.90 | 0.475 | 0.411 | 0.62 / 0.64 / 0.64 |
| 0.97 | 5.6e-4 | 126/150 | 0.634 | 1.069 | −1.151 | 0.163 | 0.788 | 0.81 | 0.452 | 0.350 | 0.58 / 0.61 / 0.62 |
| 0.99 | 5.6e-4 | 144/150 | 0.625 | 1.074 | −1.153 | 0.246 | 0.788 | 0.69 | 0.461 | 0.292 | 0.46 / 0.58 / 0.61 |
| 1 | 5.6e-4 | 147/150 | 0.521 | 1.324 | −1.153 | 0.295 | 0.780 | 0.60 | 0.417 | 0.182 | 0.39 / 0.48 / 0.53 |

- **E2a's lr does not transfer to 784 sites.** The two low-β cells at 5.6e-4
  (β = 0, 0.01) never pass 0.88 valid accuracy, while their neighbours at 1.7e-4
  (β = 0.001, 0.03, 0.1) reach 0.98 by epoch 12–25. The cells at 1.7e-4 then blow
  up: the *training* objective jumps (β = 0.001: 0.04 → 0.61 at epochs 44–48;
  β = 0.1: 0.04 → 0.20 at the same epochs, same seed and data order), and accuracy
  recovers only partly (0.98 → 0.79 → 0.93). This is optimisation, not
  overfitting. No NaN and no norm excursion (log Z and log |ψ|² stay at their target).
  So **β = 0's 0.884 is the lr, not the absence of a generative term**: β = 0.01 at
  the same lr does as badly (0.861). The regulariser hypothesis (Now, 2026-10-14) has
  no support from E2b.
- **Accuracy knee at β ≈ 0.68** (ln r ≈ 0.75): the middle of 0.979 → 0.521 lies
  between β = 0.5 (0.886) and 0.75 (0.688). E2a's knee was β ≈ 0.73 (ln r ≈ 1.0),
  so the **shift is ≈ −0.25** (±0.5 from the grid and the lr confound). Predictions:
  0 for p = 1, +0.85 for p = ½, +1.7 for p = 0. **p = 1 (keep N)**, and D82's
  α = 0.01 (β = 0.89) is past the knee (0.651), as predicted. The bottom is not
  converged (β = 1 still climbs at epoch 150, best epoch 147). A higher bottom would
  move the middle left, not right. Accuracy also falls earlier than at 12×12
  (0.915 / 0.886 at β = 0.25 / 0.5, against 0.967 / 0.946), on plateaus, not
  still climbing.
- **The density knee is not readable.** At β ≤ 0.1, selection on `objective/valid`
  (≈ L_dis) picks epochs 22–41, before the blow-ups, when $L_x$ has barely moved
  ($L_x$ at β = 0.1: −0.01 selected, −0.52 at epoch 92). The selected $L_x$ crosses
  the middle of its range at β ≈ 0.15 (12×12: 0.03–0.1), a shift of ≈ +1.5 that
  comes from early stopping. Only one knee shifts, and that one is confounded, so
  the rule (keep N unless both shift by more than ~1) keeps N.
- **Robustness at ε = 0.1 almost vanishes at 784 pixels:** rob 0.02–0.14 for the
  good classifiers (12×12: 0.28–0.52), rising to 0.30 at β = 1 (at 0.52 accuracy).
  Purification lifts β = 0.1 from 0.13 to 0.753 (recovery 0.716), but β = 0.001
  (density untrained, $L_x$ +0.54) already reaches 0.530. So about 0.4 of that lift
  is the transfer gap and ≈ +0.2 is the density's. Among well-trained cells,
  purify peaks at β = 0.1 (0.753), against 0.665 / 0.681 at β = 0.25 / 0.5 on worse
  classifiers. β\* ∈ [0.1, 0.25] fits E2a's flat top, but this is weak evidence.
  The 5.6e-4 cells' purify (0.68–0.72) beats their 1.7e-4 neighbours' while their
  accuracy is worse: transfer from a badly trained classifier differs.
- **Detection** as in E2a: ≈ 1.0 with an untrained density, 0.75–0.83 above β = 0.25.
  err_passed ≈ 0.9–0.99 at β ≤ 0.75: almost every undetected attack succeeds.

Decided with Martin (2026-10-15): **keep N** (p = 1); **the ladder
{0, 0.01, 0.1, 0.25, 0.5, 0.9, 1}** for every dataset (D108, `defaults.yaml`), with
0.25 and 0.5 between the knees at both sizes; **full MNIST tunes its own lr**:
`mnist_capacity` and `mnist_nat` already search the default `{log: [1e-5, 1e-1]}`,
only the pilot borrowed E2a's lr; **the regulariser reading of β = 0 is dropped** (the
lr explains it). Open: `mnist_nat` caps at 100 epochs, and E2b's β ≥ 0.75 cells
still climbed at 150.

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
- **The forward without `reset`: done (D104), cluster check open.** `forward` calls
  `contract()` on its kept nodes, recorded like tk's `trace`; bit-identical, eager step
  174 → 130 ms, eager analysis 81 → 53 s, captured training unchanged (compute.md,
  "Since D104"). Open: reserved memory +0.8 GiB per process at E2a's shape (allocated
  unchanged, cause not found); measure on G21G01 before packing units (handoff.md).
- **A tk `renormalize` operation that keeps the norm (deep, hard; reduces complexity).**
  tk's `renormalize` discards the norm; bm4tc's `_inline_contraction` override
  computes it in Python and accumulates it beside the network. An operation that
  returns the log-norm as a node of its own would put the accumulation on tk's
  computational path (trace, successors, replay through `forward`), so the override,
  the accumulator and the special forward could go. A tensorkrowch change, upstream
  candidate.
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
6. **Done (D106, 2026-10-15): double layers only where something is summed, fixed
   sampling nodes.** Martin rejected step 3's open leg (memory d_out^k for k output
   sites). An environment is double only if it contains a summed site; each step of
   the sweep is a few fixed virtual nodes whose tensors are set from site to site,
   so memory does not grow with n except the right environments kept for the sweep
   (B·r² each, in Gibbs purification only left of the class). Same draws; the
   sampler holds megabytes between calls instead of 1–2 GiB (compute.md, "Gibbs
   purification since D106"). The tk PR branch `sample-density` has the same code
   (`../tensorkrowch/_notes/note_jose_sample_density.md`).
   **Open, not to be solved on the fly: float32 precision of two layers.** A
   two-layer environment is PSD in exact arithmetic, but its float32 rounding error
   is not: a double layer sums ket × bra products, so its error grows like the
   square of the contraction's cancellation (κ = |l|·|A x|·|r| / |ψ|, median 10³,
   max 8·10³ at 784 sites), against linear for one layer (amplitude, then |·|²).
   Measured (random MPS, obc, class of 10 in the middle, poly degree 2, renormalize;
   scratch scripts, not kept):
   - site densities with every input fixed, worst error relative to the row's
     largest weight: 144 sites r40, one layer 1.5e-5, two layers 2.2e-4; 784 sites
     r40 7.6e-4 vs 3.5e-2; r80 3.6e-4 vs 2.6e-2; float64 ≤ 2e-11.
   - `sample(in_condition)` at 784 sites r40, 4 samples, 32 candidates in a narrow
     window (0.8–1.0), worst negative weight / row maximum over 3 seeds: tk develop
     (two layers) −2.7e-3; the rebuilt branch −8e-2, raising "negative beyond
     numerical tolerance" in 2 seeds; the open-leg single layer (step 3) ≥ −5e-6;
     float64 always clean. That branch (two layers everywhere) was 10–30× worse
     than develop here, likely because it forms the site's density matrix in the
     embedding basis first and evaluates each candidate as a quadratic form, which
     cancels when the whole window is improbable (develop puts the candidate in
     before the environments, at memory B·candidates·r²). D106 (one of the two
     environments double, one layer elsewhere): −8e-4, no raise; with 24 samples
     and per-sample windows around one value (the case that raised on that
     branch, −11.5%) −4.6%, no raise.
   Possible fixes, undecided: float64 where two layers are used; environments
   stored as a square root (ρ = F F†, a QR per site, about B·r³·d); routing
   negative rows to the uniform draw. **Still to check: float32 on trained models**
   (none on the laptop; random full-MNIST models at r40 / r80 ran ten sweeps with no
   negative weight beyond tolerance). Check before relying on Gibbs results from
   D106 on (E3): e.g. one purification at the analysis settings on a trained full
   MNIST checkpoint on the cluster, watching for "negative beyond numerical
   tolerance".

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
