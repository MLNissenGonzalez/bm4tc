# Notes for coding agents

Read [README.md](README.md), then [GUIDE.md](GUIDE.md) (concepts, pipeline, code
map). Design decisions D1–D95 are in [docs/decisions.md](docs/decisions.md), the
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

## State (2026-10-13)

Branch `ousterhout`, pushed to GitHub up to `f9e7cb0c` (D95). Phases 0–7 of the refactor are done; `main` is untouched and the local tag
`pre-ousterhout` marks it. Phase 8 (journal studies on the HPC) is in its pilots;
`docs/plan.md` has the details, `docs/compute.md` the (pre-graph) cost estimates.

**Start here: E2a relaunches on D95 (`f9e7cb0c`; Next 1); D92's crash is fixed
(D94, cluster check passed 2026-10-13).** `docs/plan.md`,
Efficiency, "Captured validation crashed the training graph" has the cause, the
evidence and the next steps. In short: tensorkrowch's
`norm_net` state and the log Z cache kept the warm-up's autograd graph alive, so the
captured backward summed gradients on the warm-up stream, outside the graph's memory
pool; a later capture's `empty_cache` released that memory. The fix captures on the
warm-up stream. **Studies may run with `trainer.cuda_graph` on only from the D94
commit on, after the cluster check; never on `5c2db9cd`..`fc9d452e`.** Martin brings
tk intuition, not CUDA expertise: explain the GPU side plainly. tk's source is in
`../tensorkrowch` (branch `develop`).

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

Laptop: the eGPU (RTX 2080, 8 GB) is back since 2026-10-13: the CUDA tests run here
(full suite with the GPU: ≈ 3.5 min).

Done on 2026-10-09, late (CPU suite green; on CUDA, D92's captured validation crashes):
- **D91:** `clean_weight` deleted; AT is `(1-β)·L_dis(x_adv) + (β/N)·L_gen(x)`, AT
  validation attacks every sample, no `n_rob`. The AT seams were re-pinned at cw = 0
  first, then reproduced exactly.
- **D92, captured evaluation (broken, see "Start here"):** `Graphs` (one pool per graph,
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
- **D94:** the crash's cause (plan.md) and its fix, one side stream per `Graphs` for
  warm-ups and captures. New seams in `tests/unit/test_graphs.py`: every allocation
  of a capture on one stream (tk-free and through the Trainer), and the E2a-shape run
  captured vs eager number for number.

**Next (in this order):**
1. **E2a, relaunched on D95** (Martin, on G21G01). D94's cluster check passed
   (2026-10-13: graph tests and bench fine). The first relaunch found D95's three
   problems (old journal kept by `--replace`, `--per-gpu 6` out of memory, Ctrl+C
   would count running trials); the study's 2026-10-09 outputs are in
   `outputs/pilot_beta12.broken-20261009`. Relaunch from a worktree at the D95
   commit, without `--replace` (it would now archive the trials done since):
   `python -m bm4tc run pilot_beta12 --gpus <idle> --per-gpu 4 --mps`. Trials the
   stopped launch left running are run again; out-of-memory ones too (D93).
2. Read E2c (copied to `pilots/cluster/pilot_beta_spirals/`, untracked; it ran on
   `d127eaad` and is sound) with Martin, then E2a once it has run (plan.md "Beta", E2:
   knees, β*, ladder, AT grid). Then E2b (`pilot_beta28`, `hparams_from` E2a).
3. The rest of the efficiency list (plan.md): cap HPO workers per cell (also the 16/15
   race); `select`/`run --cell` on a subset; a multi-study launcher; the Gibbs O(n²)
   rework; re-estimate `docs/compute.md` (units per GPU from M0).
4. Not urgent: the sampling comparison with the fork's `develop` branch.

Other open items (time series, adaptive attacks, notes on the other laptop) are in
`docs/plan.md`.

## Personal laptop vs lab laptop (2026-10-12)

This compares the personal laptop's local state with the lab laptop's state (this
branch), for plan.md's open item "Agent notes on the other laptop". It records
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
| Known bug, open: spirals AT `rob` (0.929 / 0.829) above the data-only ceiling (0.679 / 0.609 at abs 0.2 / 0.3), on real-dtype runs from before D87 | `analysis/ceiling.py` and the `clean` part compute the ceiling. The bug is not mentioned in plan.md |
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
