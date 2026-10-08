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

## State (2026-10-08, night)

Branch `ousterhout` (pushed to GitHub up to `c6436b72`; `c122225c` and this handoff are
local): Phases 0–7 of the refactor are done; `main` is untouched and the local tag
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

Done on 2026-10-08 (no D-number yet, results in plan.md, commit `c122225c`):
- **E0 confirmed D87.** Pilot A's valid/test gap is the evaluation path, not the data:
  one checkpoint through one code path gives train/valid/test within 0.7 nat; at α = 0
  the gap (2.89) equals the training-time minus the checkpoint log Z exactly.
- **CUDA graphs + NVIDIA MPS, prototyped and measured** (plan.md "Efficiency"). A whole
  training step (NAT, or AT with PGD-K; micro-batches; Adam `capturable=True`) captured
  once and replayed is **bit-identical** to the eager step and 9–10× faster per unit on
  G21G01. With MPS, 8 graphed units per GPU give 88 NAT / 9.5 AT PGD-10 steps/s against
  14.2 / 2.0 for today's 6 eager units: **≈ 5–6× NAT, 4–5× AT per GPU**. What is left is
  GPU time, a third of it the renormalisation's complex ÷ real division. tensorkrowch's
  `renormalize` keeps gradients but discards the norm (log Z computes each norm twice);
  a custom autograd Function would sit outside tk nodes (D30), so it belongs in tk.
- Prototypes (untracked, `pilots/`): `graph_bench.py` (eager vs graph, `--mode both|eager|
  graph`, `--regime nat|at`, `--micro`), `graph_throughput.sh` (1…N units per GPU),
  `graph_kernels.py` (GPU kernels per step by kind). Cluster logs in `pilots/cluster/`.
- **MPS gotcha:** start the daemon with the physical `CUDA_VISIBLE_DEVICES=<gpu>` and
  `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` set; clients then use
  `CUDA_VISIBLE_DEVICES=0` (relative to the MPS server). Stop it with
  `echo quit | nvidia-cuda-mps-control`. MPS makes eager units slower: graphs only.

Cluster: G21G01 is set up (env, clone; worktrees `runs/87663bc3` = pre-D86 code, where
pilot B runs, and `runs/c6436b72` for the graph benches). Eager units are ≈ 2.4× slower
than the laptop (CPU-bound); graphed ones are as fast. `/ceph` is ceph over NFSv4.2. New
studies need a new worktree at the current commit; runs from before D86 (cells `a…`,
column `alpha`) cannot be warm-start sources. Martin's notes `docs/hpc_*.md`,
`docs/interpolation.md`, the pilot CSVs and the scripts and logs in `pilots/` are
untracked on purpose (personal, not for git).

Laptop: the eGPU (RTX 2080 in a Thunderbolt enclosure) is away until Tuesday
2026-10-13; until then GPU checks are commands for Martin to run on the cluster. Its
drop on 2026-10-08 was an Xid 79 (fell off the bus, Thunderbolt link), not load; a
driver crash like that needs a reboot, replugging does not help.

**Next (in this order):**
1. With Martin: pilot B's results (`pilot_pgd5` seeds were training, `pilot_pgd10` HPO
   at 8/15 on 2026-10-08 evening): the PGD step count goes to
   `configs/trainer/at.yaml` (D43 rule), and whether AT's clean accuracy is acceptable
   (levers: capacity, a fixed cw = 0.5, TRADES).
2. **Design the CUDA-graph + MPS integration with Martin** (show each option with a
   code sketch and its effect on pinned numbers first, then a D-number): the graph in
   the `Trainer` (opt-in or always; also for analysis/attacks?); the static batch (the
   last batch); the non-finite check, `_cache_amp_diag`, `loss.item()` and
   `renormalize_` outside the captured step (every N steps / per epoch); ε of the
   curriculum as a device scalar; the AT random start (graph RNG changes seeded draws:
   re-pin AT seams); a recapture on lr or shape changes; the launch running one MPS
   daemon per GPU and `--per-gpu` ≈ 8. Then implement, re-pin in its own commit, and
   re-estimate `docs/compute.md`. Open check: full MNIST d3r40 with micro-batch 256
   ran out of memory on the laptop's 8 GB; verify on the cluster.
3. The rest of the efficiency list (plan.md): cap HPO workers per cell; `select`/`run
   --cell` on a subset of cells; a multi-study launcher; the Gibbs O(n²) rework; then
   the GPU-time cuts (custom backward, fewer ops per site, the faster division).
4. Plan the β pilots with Martin (plan.md "Beta", E2: knees on MNIST 12×12, full MNIST
   and spirals; they set the ladder, the AT grid and re-check D84's target), then run
   them.
5. Not urgent: the sampling comparison with the fork's `develop` branch (plan.md,
   "Sampling"; the clone is at `~/0git/tensorkrowch`, branch `develop`).

Other open items (time series, adaptive attacks, notes on the other laptop) are in
`docs/plan.md`.
