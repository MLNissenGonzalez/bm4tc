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

## State (2026-10-08)

Branch `ousterhout` (pushed to GitHub up to the pilot commit `87663bc3`; later commits
are local): Phases 0–7 of the refactor are done; `main` is untouched and the local tag
`pre-ousterhout` marks it. Phase 8 (journal studies on the HPC) has started with the
pilots; `docs/plan.md` has the details, `docs/compute.md` the cost estimates.

Done on 2026-10-07/08 (D78–D85): dataset cache lock + atomic write; micro-batches
(`trainer.micro_batch_size`, per-arch `archs:` block, analysis chunks follow); AT with
cw = 0 and lr-only HPO; 15 trials per tuned hparam with median pruning; AT α grid; four
pilot studies; pilot A verdict (log Z target n·ln d / 2 for cold MNIST) and a denser
small-α ladder; seam tests force AVX2 CPU kernels.

Cluster: G21G01 is set up (env, clone, worktree `runs/87663bc3`). It is ≈ 2.4× slower
per unit than the laptop (CPU-bound), and `/ceph` is ceph over NFSv4.2. Martin's notes
`docs/hpc_*.md` and the pilot CSVs in `pilots/` are untracked on purpose (personal,
not for git).

**Next:**
1. Read pilot B (`pilot_pgd5` due 2026-10-08 evening, `pilot_pgd10` 2026-10-09
   morning): the PGD step count goes to `configs/trainer/at.yaml` (D43 rule). Decide
   with Martin whether AT's clean accuracy is acceptable (levers in the 2026-10-08
   discussion: capacity, a fixed cw = 0.5, TRADES); also whether the AT α grid should
   move to small values (α = 0.1 is almost purely generative on MNIST, D84).
2. **Review the β plan critically, then implement it** ("Planned: β" in
   `docs/plan.md`): one dimension-corrected parameter replaces α everywhere. Start
   with its six review questions (measure the gradient-norm ratio first).
3. Before the big studies: cap HPO workers per cell; let `select`/`run --cell` work on
   a subset of cells; the Gibbs O(n²) rework and the per-step overhead (Phase 9 track);
   a multi-study launcher.

Other open items (time series, adaptive attacks, notes on the other laptop) are in
`docs/plan.md`.
