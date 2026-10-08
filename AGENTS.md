# Notes for coding agents

Read [README.md](README.md), then [GUIDE.md](GUIDE.md) (concepts, pipeline, code
map). Design decisions D1–D85 are in [docs/decisions.md](docs/decisions.md), the
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

## State (2026-10-08, evening)

Branch `ousterhout` (pushed to GitHub up to the pilot commit `87663bc3`; later commits
are local): Phases 0–7 of the refactor are done; `main` is untouched and the local tag
`pre-ousterhout` marks it. Phase 8 (journal studies on the HPC) has started with the
pilots; `docs/plan.md` has the details, `docs/compute.md` the cost estimates.

Done on 2026-10-07/08 (D78–D87):
- D78–D85: dataset cache lock; micro-batches; AT with cw = 0 and lr-only HPO; 15 trials
  per tuned hparam with median pruning; four pilot studies; pilot A verdict (log Z
  target n·ln d / 2 for cold MNIST); seam tests force AVX2 CPU kernels.
- **D86, β replaces α:** the code optimises L_β = (1−β)·L_dis + β·L_gen/N, N = n + 1
  (features and class), a weighted mean of per-variable NLLs. Logged: `loss_dis` [nats
  per label], `loss_x` = −log p(x)/n [nats per feature] (replaces `loss_gen`), `log_px`
  per feature, `norm/*` per site (headroom absolute); norm penalty ÷N. Cells `b<β>`;
  figures plot β at ln(β/(1−β)). Placeholder ladder {0, 0.01, 0.1, 0.5, 0.9, 0.99, 1}
  for every dataset, AT grid {0, 0.1, 0.5}, `paper.yaml` β values are placeholders:
  all revisited after the β pilots. Translation: β/(1−β) = N·α/(1−α). The theory is
  Martin's untracked note `docs/interpolation.md`. "TPM" = the TPM 2026 paper
  (`_paper/main.tex`).
- **D87, a core bug found by E0:** tensorkrowch's obc boundary vectors were trainable
  and norm_net had its own pair, so during training log Z was not the normaliser of ψ
  (pilot A's ~45-nat valid/test L_gen gap). Every β > 0 run trained before D87 (pilot A,
  likely TPM) optimised and selected on a corrupted objective; test numbers were exact
  for the saved models. Fixed (`_freeze_boundaries`), regression-tested against a
  numerical integral, seams re-pinned. Pilot B (β = 0) is essentially unaffected.

Cluster: G21G01 is set up (env, clone, worktree `runs/87663bc3` = pre-D86 code, where
pilot B runs). It is ≈ 2.4× slower per unit than the laptop (CPU-bound), and `/ceph` is
ceph over NFSv4.2. New studies need a new worktree at the current commit; runs from
before D86 (cells `a…`, column `alpha`) cannot be warm-start sources. Martin's notes
`docs/hpc_*.md`, `docs/interpolation.md`, the pilot CSVs and `pilots/e0_valid_test_gap.py`
in `pilots/` are untracked on purpose (personal, not for git).

**Next (in this order):**
1. With Martin: pilot B's results (`pilot_pgd5` due 2026-10-08 evening, `pilot_pgd10`
   2026-10-09 morning): the PGD step count goes to `configs/trainer/at.yaml` (D43
   rule), and whether AT's clean accuracy is acceptable (levers: capacity, a fixed
   cw = 0.5, TRADES). E0 on the cluster (`pilots/e0_valid_test_gap.py`, run from the
   `runs/87663bc3` worktree): it should show pilot A's gap equals the log Z drift.
2. **Efficiency and compute cost, before any new pilot** (Martin's order):
   - the per-step overhead (plan.md "Efficiency": measured, two prototypes ruled out;
     next a custom autograd Function for the renormalised chain and the log Z
     zip-up, then fewer operations per site, then CUDA graphs; D30's bar);
   - cap HPO workers per cell; let `select`/`run --cell` work on a subset of cells; a
     multi-study launcher; the Gibbs O(n²) rework;
   - then re-estimate the cost of the β pilots and of Phase 8 (`docs/compute.md`).
3. Plan the β pilots with Martin (plan.md "Beta", E2: knees on MNIST 12×12, full MNIST
   and spirals; they set the ladder, the AT grid and re-check D84's target), then run
   them.
4. Not urgent: the sampling comparison with the fork's `develop` branch (plan.md,
   "Sampling"; Martin clones it first).

Other open items (time series, adaptive attacks, notes on the other laptop) are in
`docs/plan.md`.
