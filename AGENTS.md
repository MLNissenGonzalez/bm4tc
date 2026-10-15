# Notes for coding agents

Read [README.md](README.md), then [GUIDE.md](GUIDE.md) (concepts, pipeline, code
map). Design decisions D1–D108 are in [docs/decisions.md](docs/decisions.md), the
source of truth; what is left to do is in [docs/plan.md](docs/plan.md). The code
follows Ousterhout's *A Philosophy of Software Design*
([docs/ousterhout.md](docs/ousterhout.md)): deep modules, information hidden in one
place, no pass-through layers.

## Working with Martin

- **He decides.** For a choice, propose 2–3 options with a recommended one and the
  trade-off; push back when an idea looks wrong. For interface or architecture
  choices, first show each option in chat (a code sketch, its effect on pinned
  numbers, the trade-off), then ask.
- Record each new decision as the next D-number in `docs/decisions.md`.
- Commit at each coherent step with a detailed message. **Never push without asking.**
- Write a handoff (a short state + next step + open items file, deleted once folded
  in) only when the context is about to fill or Martin asks.
- Paper sources and rebuttal drafts in `_paper/` are untracked on purpose: reference
  only, keep them out of git.

## Rules for changes

- **Pin before you change behaviour** (D26, D51). The seam tests (`tests/e2e/`) pin
  end-point metrics and per-epoch training curves. A refactor reproduces them
  exactly. A deliberate change re-pins in its own commit, with the reason. Take
  "before" numbers from a `git worktree` of HEAD, not from a background run of code
  you are editing.
- **tensorkrowch first** (D30): tensor-network maths stays in `ConditionalBornMachine`
  on tensorkrowch nodes and contractions, not in plain torch elsewhere. The numerical
  core (contractions, `log_partition_function`, accumulate, sampling) changes only on
  the Phase 9 track.
- **The import rule** (D32): `bm4tc.core` imports neither `analysis` nor `pipeline`;
  `bm4tc.analysis` imports only `core`. A test enforces it.
- Metric keys are built only in `bm4tc/pipeline/metrics.py`; run paths only in
  `bm4tc/pipeline/runs.py`; nothing parses a path (runs are found by `run.json`).
- Clean break (D12): no compatibility shims for old run dirs, configs or CSVs; the
  tag `pre-ousterhout` has the old code.

## Environment

- Conda env `bm4tc` from `environment.yml` (pinned). There is no bare `python` on the
  dev laptop: `conda run --cwd $PWD -n bm4tc pytest -q -p no:logging`
  (`-p no:logging` keeps failure output readable), likewise
  `conda run ... python -m bm4tc ...`. Ad-hoc scripts outside the repo need
  `PYTHONPATH=$PWD`.
- The full suite takes ~3 min: run it in the background and wait for it once.
- Point `BM4TC_DATA_ROOT` at a scratch directory for experiments, to keep datasets
  and outputs out of the repo.
- Local GPU: an RTX 2080 (8 GB); tests must fit in it.
- **Runs happen on the lab HPC**: SSH, jobs started by hand in screen, several GPUs per
  node, **no AI agents there** (D37). Everything must be operable from short commands
  and diagnosable from `status` and the log files.
  `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. The HPC env needs recreating
  from `environment.yml`.
- `git add` with a pathspec that no longer exists aborts the whole add.

## State (2026-10-15)

Branch `ousterhout`, pushed to GitHub up to `da7185e8` (D105, E2b's analysis batch).
Phases 0–7 of the refactor are done; `main` is untouched and the local tag
`pre-ousterhout` marks it. Phase 8 (journal studies on the HPC) is in its pilots;
`docs/plan.md` has the details, `docs/compute.md` the (pre-graph) cost estimates and
the measurements since (stale numbers are marked there).

Two sessions work in parallel: **E2b and graph capture** (handoff below) and the
**tensorkrowch sampling rebuild** (its notes in `../tensorkrowch/_notes/`, plan.md
"Sampling"; it has uncommitted work in `bm4tc/core/sampling.py` and
`tests/unit/test_sampling.py`: leave those files to it).

**Start here (E2b and graph capture):**
1. **Read E2b.** Its analyses finished on G21G01 (launched from `da7185e8`, batch
   64). Copy it, then read it like E2a (plan.md, "Beta / MNIST 12×12"): the knee shift
   between 12×12 and 28×28 decides the exponent of N and the Phase 8 ladder (plan.md,
   "E2, the β pilots").
2. **The graph memory problem** (below): measure first, then fix. AT cannot launch at
   full MNIST until it is solved, and every `--per-gpu` depends on it.
3. **E3** and the rest (below, "Next").

### Handoff: E2b and graph capture (2026-10-15)

**E2b** (`pilot_beta28`: full MNIST d3r40 micro 256, E2a's 12 β with E2a's lr per β,
150 epochs, one seed). Trained from `05c44cf8` (β = 0.9 in a second launch);
analysed from `da7185e8` with `analysis.batch_size: 64`. Copy it with

    H=nisseng261@N315L-G21G01.ressource.unicaen.fr; R=/ceph/chercheurs/nisseng261/bm4tc
    rsync -av --exclude '.logs/' --exclude '.replaced/' --exclude 'models/' --exclude 'wandb/' \
        $H:$R/outputs/pilot_beta28/ pilots/cluster/pilot_beta28/

β = 0 trained to 0.887 only (valid accuracy 0.87 by epoch 21, then swinging
0.79–0.87; no NaN or collapse) while β = 0.01, with the same lr, trained well. Martin:
keep it as a result, not retrained; β > 0 may act as a regulariser, a future study can
test that in the high-capacity regime (plan.md). Read the accuracy knee's top from
β = 0.001–0.1. Ask Martin for `$BM4TC_DATA_ROOT/d104_checks/*.log` (his D104 checks on
G21G01 at `da7185e8`; the train benches are in the table below): `graphs.log`
(test_graphs on the cluster) and `analysis784.log` were not seen here.

**What happened with graph capture.** Every E2b analysis died capturing the 40-step
PGD: out of memory beside another unit, a segmentation fault alone on a 22 GB GPU. A
graph of the whole loop holds steps × sites contractions (784 × 40 ≈ 31,000; E2a's
144 × 40 worked). **D105** captures one step and replays it (`GraphedPGD`,
`GraphedPurification`), in the analysis, the Trainer's validation and AT training (the
Trainer attacks every micro-batch before the step; `Trainer._adversarials`). Eager
numbers and every seam unchanged; captured = eager bit for bit with a fixed start. On
the eGPU at 784 sites the analysis now runs (48 s vs 143 s eager) at 2.55 / 5.05 GiB
reserved for batch 32 / 64 (0.77 / 1.52 eager): ≈ 0.08 GiB per sample.

**What is still wrong: reserved memory far above allocated, worst for AT.**
`bench_train_step` on G21G01 at `da7185e8` (D104 + D105), β 0.5, batch 512, per
process (nvidia-smi):

| Shape | Allocated (peak) | Reserved | Per process |
|---|---|---|---|
| MNIST12 d3r40 NAT | 1.82 GiB | 4.24 GiB | 4.57 GiB (E2a, before D104: 3.89) |
| MNIST12 d3r40 AT | 2.76 GiB | 10.42 GiB | 10.92 GiB |
| full MNIST d3r40 micro 256 NAT | 5.01 GiB | 9.24 GiB | 10.46 GiB (E2b's real trains: ≈ 9.3) |
| full MNIST d3r40 micro 256 AT | 3.63 GiB | 15.33 GiB unallocated | out of memory in the first validation's attack warm-up |

Why, as far as the code shows (not yet measured):
- **One private pool per graph** (D92): every graph keeps its own peak reserved, so
  reserved ≈ the sum of the graphs' peaks, not the largest.
- **D105 added graphs to AT.** Before, AT's step (PGD-5 on each chunk + loss +
  backward + optimizer) was one graph. Now: the training step, the training attack's
  step (`Trainer._graphed_attack`), the validation's clean sums, the validation's
  attack step (`Evaluation._attack`, a second `GraphedPGD` on the same model) and the
  forward on the attacked batch. Validation runs in chunks of the micro-batch (D79),
  so at full MNIST the training and validation attacks capture **the same shape
  (256) twice, in two pools**, and the last, smaller validation chunk adds a third
  shape per graph.
- Warm-ups run eagerly on the side stream (D94); the allocator keeps their blocks
  cached for that stream.
- D104 adds ≈ 0.8 GiB per process (tk's kept nodes).

**Measurements to take first** (cluster, one free GPU; `bench_train_step` prints
allocated, reserved and per-process memory):
1. **Attribute D104 vs D105:** the same four shapes at `f56365de` (D104, no D105) and
   at `05c44cf8` (neither). Whatever D105 added is the graph structure; whatever D104
   added is tk's kept nodes.
2. **Per-graph accounting:** after training and one validation, the allocator's
   snapshot (`torch.cuda.memory._snapshot()`, segments carry their pool id), summed per
   pool, mapped to the `Graphed` objects (train step, training attack step, clean
   sums, validation attack step, adversarial sums; each per shape). A bench option
   that pickles the snapshot is fine (`tests/bench/`); the viewer is
   `torch.cuda.memory._memory_viz`.
3. **Eager baseline:** `--eager` at the same shapes, to separate what capture costs
   from what the computation needs.
4. Still open from D104's list: d3r80 micro 128 (did it fit?), and Gibbs on full
   MNIST in a real analysis (≈ 50 s per run expected).

**Possible solutions** (to propose to Martin with numbers; recommended first):
- **A. One pool per `Graphs`** (`torch.cuda.graph_pool_handle()` passed to every
  capture of one task). The graphs of a task never run at once, inputs are copied in
  and outputs cloned out (D92), so reserved becomes ≈ the largest graph's peak
  instead of the sum. Risk: memory that outlives a capture, allocated inside it
  (tk's kept nodes and successors, D104; the log Z cache), could be reused by the
  next capture. Check with D94's allocation tests (`test_graphs.py`: every allocation
  of a capture in its pool) extended to two graphs, plus captured = eager.
- **B. One `GraphedPGD` for training and validation.** The Trainer passes its attack
  graph to `Evaluation`: same model, same graphs, same chunk shape at full MNIST, so
  one capture and one pool instead of two. Cheap; AT only.
- **C. Several steps per graph** instead of one: a graph of k steps with
  k × sites under a bound (PGD-5 training at 784 sites ≈ 3,900 site-steps worked
  before D105). Fewer graphs and launches, still bounded; adds a rule to tune.
- **D. Free a task's graphs after use:** drop validation's captures (and
  `empty_cache`) after each validation, re-capturing next time (seconds per
  validation). Trades capture time for memory; simplest fallback for validation.
- **E. Eager AT validation** (`eval_every` 5: a fifth of the epochs): no validation
  graphs at all. Costs validation time, no risk.
- **F. Smaller graphs at the source:** tk's environment-first forward (B·n·r·d stored
  instead of B·n·r², the tensorkrowch track) lowers every graph's peak; not a fix for
  the duplication.

Expected order: measurements 1–3, then B (small, certain), then A (largest gain, needs
the allocation tests), D or E only if A fails. Record the choice as D106.

**Rule from D94, still binding:** studies may run with `trainer.cuda_graph` on only
from the D94 commit on, after the cluster check; never on `5c2db9cd`..`fc9d452e`.
Martin brings tk intuition, not CUDA expertise: explain the GPU side plainly. tk's
source is in `../tensorkrowch` (branch `develop`; our work on `sampling-efficiency`).

Done before (D78–D87, see decisions.md): micro-batches, AT with cw = 0, the HPO budget,
pilot A (log Z target n·ln d / 2), **β replaces α** (D86; Martin's theory note
`docs/interpolation.md`, untracked), and **D87** (frozen obc boundaries: β > 0 runs
before it optimised a corrupted objective).

Done on 2026-10-09:
- **D88, pilot B:** PGD-5 in `configs/trainer/at.yaml`; the MNIST AT studies train 150
  epochs with patience 8.
- **D89:** one amplitude contraction, `log_amp_sq`, always accumulating the norm in log
  space (`born.accumulate`, the raw path and `amplitudes()` deleted). Old checkpoints
  whose config has `accumulate` no longer load (D12).
- **D90, CUDA graphs + MPS:** the training step has no host syncs (`mixed_nll` returns
  `(objective, log_amp_sq)`; norm statistics stay on the GPU, `NormTracker` syncs once
  per epoch; the reset retry and `hard_every` are gone). `bm4tc/core/graphs.py`
  (`Graphed`) captures it; `trainer.cuda_graph` (default on, CUDA only); Adam with
  `capturable=True` on CUDA; the AT radius is a device scalar. `run ... --mps` starts one
  MPS daemon per GPU (`bm4tc/pipeline/mps.py`, left running). Captured AT with a random
  start differs from eager (RNG offsets) but is reproducible.
- **On G21G01:** `tests/unit/test_graphs.py` passes 7/7 (a failed capture poisons the
  process in torch 2.1, so that test runs in a child process). M0 memory per unit
  (`tests.bench.bench_train_step`, which now takes the shape and prints peak memory):
  MNIST12 d3r40 1.87 GiB (8 units per GPU), full MNIST d3r40 micro 256 NAT 5.3 GiB,
  AT 7.6 GiB; **d3r80 micro 128 runs out of memory in the eager validation**, because
  the training graph's private pool stays reserved (compute.md, "Since CUDA graphs").
- **β pilots launched** (`configs/studies/pilot_beta12.yaml` = E2a, MNIST12 d3r40, 12 β,
  6 random lr trials, 1 seed; `pilot_beta_spirals.yaml` = E2c), both with `--mps
  --per-gpu 8`, worktree `runs/d127eaad`; E2a on GPUs 0,5 (overnight). E2c showed one
  cell with 16/15 trials (parallel workers race on the trial count; fix with the HPO
  worker cap).

Cluster: G21G01 (8× RTX 6000, 22 GB usable; 80 threads). New studies need a worktree at
the current commit. MPS gotcha: the daemon gets the physical `CUDA_VISIBLE_DEVICES`,
clients `CUDA_VISIBLE_DEVICES=0`; `--mps` handles it. Martin's notes `docs/hpc_*.md`,
`docs/interpolation.md` and `pilots/` are untracked on purpose; `docs/hpc_bm4tc.md`
still shows launches without `--mps`.

Laptop: the eGPU (RTX 2080, 8 GB) is back since 2026-10-13: the CUDA tests run here
(full suite with the GPU: ≈ 3.5 min).

Done on 2026-10-09, late (CPU suite green; on CUDA, D92's captured validation crashed until D94):
- **D91:** `clean_weight` deleted; AT is `(1-β)·L_dis(x_adv) + (β/N)·L_gen(x)`, AT
  validation attacks every sample, no `n_rob`. The AT seams were re-pinned at cw = 0
  first, then reproduced exactly.
- **D92, captured evaluation (its crash fixed by D94):** `Graphs` (one pool per graph,
  replays return copies, one warm-up stream: that fixed the 3× warm-up memory),
  `Evaluation` (the Trainer's validation, captured: crashes the training graph), and the
  analysis's per-batch work (PGD, joint PGD, likelihood purification, log p(x),
  prediction) captured for the MPS on CUDA (≈ 7× per batch, bit-identical to eager).
  Gibbs and JEM stay eager. d3r80 (micro 128) does not fit yet.
- **D93:** an HPO trial that ends in a CUDA error is retried, not counted, and stops its
  worker. E2a's two launches were lost (out of memory, then the shared pool); it
  relaunches with `--replace`.

Done on 2026-10-13 (eGPU):
- **D95:** `--replace` always starts an HPO over; trials left running by a stopped
  launch are run again, not counted; `bench_train_step` validates like a real run
  and prints the process's GPU memory (E2a's unit: 3.9 GiB on G21G01).
- **D94:** the crash's cause (decisions.md) and its fix, one side stream per `Graphs` for
  warm-ups and captures. New seams in `tests/unit/test_graphs.py`: every allocation
  of a capture on one stream (tk-free and through the Trainer), and the E2a-shape run
  captured vs eager number for number.

Done on 2026-10-13 (evening):
- **D96, adaptive attacks** (`bm4tc/analysis/adaptive.py`, part `adaptive`, off by
  default): `Defense` (purifier, classifier, detector at τ_q on the input or the
  purified input) and `AdaptivePGD` (BPDA + EOT through the purifier, selective PGD
  through the detector, best iterate, restarts). New keys: `purify_adapt`,
  `rob_sel[_adapt]`, `purify_sel[_adapt]`, `purify_gibbs_adapt`. No grid attack: Martin
  wants universal attacks, not 2-D special cases.
- **D97, ceilings:** selective (the ceiling at ε/2), and the population ceiling on the
  spirals law (a certified max-flow bracket, `DataHandler.class_masses`,
  `population_ceiling`), logged as `rob_ceiling[_sel]/pop/{eps}`. Spirals at ε = 0.1:
  test 0.676, population [0.651, 0.654]. MNIST has no useful ceiling.
  `docs/adaptive.md` has the theory, the curve and the first pilot.
- **E3** (`configs/studies/pilot_adaptive_spirals.yaml`; it needs an untracked copy of
  E2c's hparams in `configs/hparams/pilot_beta_spirals.yaml`, from
  `pilots/cluster/pilot_beta_spirals/hparams.yaml`). A first run (stopped) gave, at
  β = 0 and ε = 0.05: the detector attack as strong as PGD or stronger; **BPDA through
  likelihood purification weaker than PGD then purify** (0.9925 vs 0.978), and Gibbs
  at 6 sweeps too. Sign-step purification is a staircase that snaps onto the density
  ridge, so the identity backward misleads (`docs/adaptive.md` §3, §5).
- **D98:** the adaptive attack's first restart starts from the PGD example, so it
  never does worse than the transfer attack. E3 is trimmed (q5, Gibbs sweeps {1, 6},
  EOT 8: ~1.5–2 h on the eGPU) and **not yet rerun**.
- The eGPU dropped off the bus once mid-run ("unspecified launch failure" in three
  processes at once, `nvidia-smi`: no devices): a driver or link problem, not the
  code. **Never suggest unplugging it then: that has crashed the laptop.** Reboot.

Done on 2026-10-14 (session `ousterhout-sampling-efficiency`):
- **D100–D101, sampling on tk nodes:** `SamplingMPS` (`bm4tc/core/sampling.py`) carries
  tk develop's `condition`/`sample` to 1.1.6 and makes them efficient (a domain per
  sample, each site's density matrix first, one layer for a sweep, uniform draw where
  no weight is usable). `GibbsPurification` is one `born.sample(in_condition=x,
  domain=windows)` per sweep; the figures' class samples are `condition` + `sample`;
  the old sampler is deleted. **D102:** the privacy attack's worst case is never below
  chance; the shared test model is seeded.
- **D103:** `sample` keeps its tk nodes between calls (tk caches each operation's result
  as a successor; a repeated call only recomputes tensors). Gibbs per analysis run
  (250 points, 10 sweeps): ≈ 9 s MNIST 12×12, ≈ 50 s full MNIST (did not fit before).
- **D104:** `ConditionalBornMachine.forward` calls `contract()` on its kept nodes
  instead of resetting, recorded like tk's `trace`; `_nodes_key` says whether the
  forward or a `sample` structure holds the network. Bit-identical; eager step 174 →
  130 ms, eager analysis 81 → 53 s, captured training unchanged; reserved +0.8 GiB per
  process (cluster check above). tk's `reset` reorders `parameters()`: compare
  parameters by name across a reset.
- **tensorkrowch:** branch `sampling-efficiency` on Martin's fork, one commit per bm4tc
  sampling step in develop's names, and `TK_NOTES.md` (tk internals: successors,
  forward's replay, trace's inverse memory, reset, node and edge rules, 1.1.6 vs
  develop, candidates). Core issues documented there, not changed.

Done on 2026-10-14 (session `E2b-and-AT-graph-capture`):
- **E2a read** (plan.md): accuracy knee β ≈ 0.7, density knee β ≈ 0.03–0.1, β\* = 0.25
  on transfer numbers; the AT grid waits for E3, the ladder for E2b.
- **D99:** `select` waits for the HPO workers to end, not to succeed (executor
  `Unit.after`); a CUDA-failed worker no longer skips select and every train.
- **E2b written and run** (`pilot_beta28`, 150 epochs; `configs/hparams/pilot_beta12.yaml`
  tracked). Full-MNIST units' real memory in compute.md and plan.md.
- **D105:** attacks and likelihood purification captured one step at a time (above).

**Next (in this order):**
1. E2b's read and the graph memory problem (handoff above).
2. **E3:** run it detached (`setsid nohup ... &`, so the terminal can close; scratch
   `BM4TC_DATA_ROOT`), read it against the ceilings (`docs/adaptive.md` §5 lists its
   questions), then the saturation ablation that sets `analysis.adaptive`'s defaults.
   Gibbs is cheap now (D101, D103).
3. **tensorkrowch (its own session, in progress):** a norm-keeping `renormalize` (the log-norm as a
   node; would remove bm4tc's `_inline_contraction` override, the accumulator and the
   special forward), a public "contract with reuse", `reset`'s quadratic renaming and
   its reordering of `parameters()`, `@` on batch-only edges, `einsum`'s connected-edge
   rule, edge inheritance, the environment-first forward (B·n·r·d stored instead of
   B·n·r²); then what goes upstream and how (PR to the fork's develop, then to
   tensorkrowch).
4. The rest of the efficiency list (plan.md): cap HPO workers per cell (also the 16/15
   race, seen again once in `test_run_in_parallel_then_status`); `select`/`run --cell`
   on a subset; a multi-study launcher; re-estimate `docs/compute.md`.

Other open items (time series, notes on the other laptop) are in `docs/plan.md`.

## Personal laptop vs lab laptop (2026-10-12)

This compares the personal laptop's local state with the lab laptop's state (this
branch), for the former plan.md open item "Agent notes on the other laptop" (done; what
still matters is now in plan.md). It records
differences only; it does not judge either side against the plan.

The personal laptop's checkout (`~/0git/bm4tc`) is on `main` at `aa697706`, the merge
base of this branch, with nothing unpushed. What it has that the lab laptop does not
is all in git-ignored files:

- `CLAUDE.md`: the agent guide for `main`, last edited around 2026-08-02 to
  2026-09-18.
- `DEFERRED.md`: two deferred features.
- `.claude/settings.local.json`: permissions only.
- `reference/`: the same material as `_paper/` here (NeurIPS v2 `.tex`, reviews,
  rebuttal drafts, `chebyshev_boundary_born_machines.md`, old analysis scripts).
- `notebooks/archive/`: 2025 notebooks.
- Backup branches `backup-behemoth`, `backup-functional-version` and `paper`, and an
  old stash on `local_luck`. These were not reviewed.

The personal laptop has no conda and only an MX150 (2 GB). Its `CLAUDE.md` says to
use a full env path and never `conda run`; that path does not exist there any more.

| Personal laptop (`main` + local `CLAUDE.md`) | Lab laptop (`ousterhout`) |
|---|---|
| Model selection: NAT on `mixed_loss` for every α (decided 2026-07-30, not implemented), AT on `at_loss` (2026-08-02) | One rule for every regime: argmin `objective/valid` (D8, D11, D48). `stop_crit` is gone |
| AT in the `gen_on_clean` split form, with `clean_weight` tuned | Split objective only (D18); cw = 0 (D80), then deleted (D91) |
| α ladders; `warm_`/`cold_` prefixes in dir names, with a bare name not meaning warm | β instead of α (D86), ladder after E2; `init: cold\|warm` per study (D19); identity in `run.json` (D17, D58) |
| Budgets authored as `eps_rel`/`delta_rel`, `_abs` derived | The same (D3, D14, D48, D65) |
| Notebooks report budgets with `REPORT_UNIT = "abs"`, because José wants absolute [-1, 1] budgets in the paper | Notebooks deleted (D75). `bm4tc/pipeline/figures/` and `configs/papers/` have no rel/abs switch; the rebuttal quotes ε = 0.1 in [0,1] pixel units (rel) |
| `born.accumulate` tied to complex64, behind a flag | On by default (D60), then the only path (D89) |
| Norm control: `hard_every` 0, soft 0.1, target 0 | Target n·ln d / 2 for cold MNIST (D84); penalty ÷N (D86); `hard_every` removed and kept as a benchmark-project idea (D90) |
| Chebyshev restricted to ±0.99, explained in `reference/chebyshev_boundary_born_machines.md` | The same restriction. The comment at `bm4tc/core/embeddings.py:250` cites that note, which exists only untracked on the personal laptop |
| Known bug, open: spirals AT `rob` (0.929 / 0.829) above the data-only ceiling (0.679 / 0.609 at abs 0.2 / 0.3), on real-dtype runs from before D87 | `analysis/ceiling.py` and the `clean` part compute the ceiling. Now an open item in plan.md |
| Known bug, open: attacks don't clamp to `input_range` | `attacks.py` clamps (lines 102, 140). `main` already clamps since `73c9d692` (2026-07-29), so the note there is out of date |
| Known bug, fixed: Gibbs weight overflow | The same fix (`draw_from_grid_log`) |
| Flat and consolidated output layouts, `analysis/batch.py`, `tools/*` | Deleted (D12, D49, D58, D77) |
| `DEFERRED.md` §2: mixed NLL for AT | AT at β > 0 (D82, D86, D91) |
| `DEFERRED.md` §1: DMRG sweeps | Not mentioned. D27 asked for DEFERRED.md's content to move into issues or the docs |

**Working rules.** The personal laptop's user-global `~/.claude/CLAUDE.md` has rules
that are not in this repo:

- Use plan mode before writing code.
- Add tests with every new feature.
- When a bug surfaces, stop and discuss it with Martin before debugging.
- Diagnostic scripts and monkey-patches never land in the codebase.
- Vocabulary: "Born Machine" is the probability rule, not a model class. Copied-input
  models are "polar networks", never "polynomial models".

These agree with "Working with Martin" above, and "polynomial model" appears nowhere
on this branch.

**Rebuttal material (personal laptop `reference/`, lab laptop `_paper/`) against
this branch.**

Covered by this branch:
- More datasets: mnist12, mnist, and time series (D6, still open).
- A baseline: JEM (D24, D35, D69–D74), parameter-matched by `arch` (D70). The
  rebuttal compared JEM at r20 with the MPS at r40.
- Accuracy vs coverage: D2 and the `coverage` figure kind.
- Langevin purification (MyWa): `sgld` (D72).
- Compute and bond-dimension scaling (2eMz Q5): `mnist_capacity` and `compute.md`.

Not covered:
- BPDA/EOT adaptive attacks (in plan.md's open items).
- Purification at a smaller or unknown radius (2eMz Q4). `purify_delta` is `[0.1]`
  only; Gibbs is radius-agnostic.
- Pretraining on unlabelled data (A5V3 Q3).

The rebuttal's MPS table came from code from before D2, D70 and D87. That includes
its comparison of MPS against JEM purification, 0.783 vs 0.536 at α = 0.01.
