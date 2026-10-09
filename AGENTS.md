# Notes for coding agents

Read [README.md](README.md), then [GUIDE.md](GUIDE.md) (concepts, pipeline, code
map). Design decisions D1–D92 are in [docs/decisions.md](docs/decisions.md), the
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

## State (2026-10-09, evening)

Branch `ousterhout`, pushed to GitHub up to `d127eaad` (the commits after it are
local). Phases 0–7 of the refactor are done; `main` is untouched and the local tag
`pre-ousterhout` marks it. Phase 8 (journal studies on the HPC) is in its pilots;
`docs/plan.md` has the details, `docs/compute.md` the (pre-graph) cost estimates.

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
  the training graph's private pool stays reserved (plan.md, "Integrated (D90)").
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

Laptop: the eGPU is away until Tuesday 2026-10-13; GPU checks are commands for Martin
to run on the cluster.

Done on 2026-10-09, late (CPU-tested; CUDA parts unchecked):
- **D91:** `clean_weight` deleted; AT is `(1-β)·L_dis(x_adv) + (β/N)·L_gen(x)`, AT
  validation attacks every sample, no `n_rob`. The AT seams were re-pinned at cw = 0
  first, then reproduced exactly.
- **D92, captured evaluation:** `Graphs` (one pool for graphs that never run together;
  replays return copies; `empty_cache()` after each capture), `Evaluation` (the
  Trainer validates in its step's graphs), and the analysis's per-batch work (PGD, joint
  PGD, likelihood purification, log p(x), prediction) captured for the MPS on CUDA.
  Gibbs and JEM stay eager.

**Next (in this order):**
1. **Martin, on G21G01** (needs a push and a worktree at the new head):
   `pytest -q -p no:logging tests/unit/test_graphs.py` (captured vs eager: validation,
   analysis routines, UQ reproducibility); the d3r80 bench
   `python -m tests.bench.bench_train_step --features 784 --bond-dim 80 --batch 512
   --micro-batch 128 --beta 0.5 --regime nat` (fits? peak reserved); one analysis
   timed captured vs the old worktree (eager).
2. Read E2c and E2a with Martin (plan.md "Beta", E2: knees, β*, ladder, AT grid). Then
   E2b (`pilot_beta28`, `hparams_from` E2a), on the new head.
3. The rest of the efficiency list (plan.md): cap HPO workers per cell (also the 16/15
   race); `select`/`run --cell` on a subset; a multi-study launcher; the Gibbs O(n²)
   rework; re-estimate `docs/compute.md` (units per GPU from M0).
4. Not urgent: the sampling comparison with the fork's `develop` branch.

Other open items (time series, adaptive attacks, notes on the other laptop) are in
`docs/plan.md`.
