# Notes for coding agents

Read [README.md](README.md), then [GUIDE.md](GUIDE.md) (concepts, pipeline, code
map). Design decisions D1–D116 are in [docs/decisions.md](docs/decisions.md), the
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
- Local GPU: an RTX 2080 eGPU (8 GB), at the lab only (none off-site, since 2026-10-16);
  tests must fit in it.
- **Runs happen on the lab HPC**: SSH, jobs started by hand in screen, several GPUs per
  node, **no AI agents there** (D37). Everything must be operable from short commands
  and diagnosable from `status` and the log files.
  `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. The HPC env `bm4tc` is built
  from `environment.yml` (torch 2.1.0+cu121, tensorkrowch 1.1.6; checked 2026-10-15).
- **Driving the cluster from this laptop (D109).** No agent is installed on the cluster,
  but you may run Martin's short commands there over SSH (key-based, `~/.ssh/config`
  is set up: do not read it). The commands are the ones in `docs/hpc_bm4tc.md`
  (untracked):

      H=N315L-G21G01.ressource.unicaen.fr; R=/ceph/chercheurs/nisseng261/bm4tc
      ssh -o BatchMode=yes $H "source $R/bm4tc.sh && cd $R/runs/<sha> && python -m bm4tc status <study>"
      # deploy a pushed commit:  cd $R/code && git fetch && git worktree add ../runs/<sha> <sha>
      # launch, detached:        screen -dmS ag-<study> bash -c "source $R/bm4tc.sh; cd $R/runs/<sha>; python -m bm4tc run <study> --gpus ... --per-gpu ..."
      # pick GPUs:               ~/0git/hpc-greyc/multitop.py gpu   (then nvidia-smi on the node)
      # results:                 rsync -av --exclude '.logs/' --exclude '.replaced/' --exclude 'models/' \
      #                              --exclude 'wandb/' $H:$R/outputs/<study>/ pilots/cluster/<study>/
      # debugging copy (D116):    git ls-files -z | rsync -a --from0 --files-from=- ./ $H:$R/ag_dev/
      #                           then: source $R/bm4tc.sh && cd $R/ag_dev && pytest -q -p no:logging tests/unit/test_graphs.py

  A script outside the checkout needs `PYTHONPATH=$PWD` there too. Wait for a remote
  job with a background `until ssh ... test -e <file>; do sleep ..; done`, not by
  polling (and delete a stale marker before relaunching). Rules: **debugging is free,
  launches are asked (D116).** Without asking: SSH queries, the `$R/ag_dev/` copy
  (tracked files only), short debugging jobs (tests, benches, one analysis) on **one
  free GPU** at a time (check `nvidia-smi` first; other users' processes run there).
  Ask first, naming the node and GPUs (shared with the lab): studies, ablations,
  anything on several GPUs or for hours. Results that are kept (studies, ablations,
  benches quoted in compute.md) come from a worktree of a pushed commit. Your screens are named `ag-*` (never "claude" in screen, study or file names), you may
  create and kill those only, and remind Martin of his own screens left open; never
  print the remote environment (`bm4tc.sh` exports the W&B key) and never read
  `secrets/`; pushing still needs asking (a branch the agent may push to freely may come
  later). `cluster-run` (`~/0git/hpc-greyc`) is **unsuitable for studies**: it copies
  a script and the modules it imports into a job directory, so there is no worktree and
  no clean sha in `run.json` (plan.md, Phase 8).
- `git add` with a pathspec that no longer exists aborts the whole add.

## State (2026-10-16)

Branch `ousterhout`, pushed to GitHub on 2026-10-16 up to `bdb4531c` (D116). Phases 0–7 of the refactor are done; `main` is untouched and
the local tag `pre-ousterhout` marks it. Phase 8 (journal studies on the HPC) is in its
pilots; `docs/plan.md` ("Now") has the details, `docs/compute.md` the cost estimates and
the measurements since (stale numbers are marked there).

**Since 2026-10-16 Martin works off-site:** the cluster over VPN + SSH as before (D109),
**no eGPU**. Locally the suite runs on CPU (CUDA tests skip: 570 passed, 21 skipped);
CUDA checks (`tests/unit/test_graphs.py`, benches) run on G21G01: from `$R/ag_dev/`
while debugging (D116), from a pushed commit's worktree for numbers that are kept.

Done on 2026-10-16 (commits `3848319d`..`bdb4531c`, pushed; deploy a worktree of the
pushed head for numbers that are kept):
- E2a's checkpoints under adaptive attack, read (`docs/adaptive.md` §7, results in
  `pilots/cluster/pilot_beta12_adaptive/`): MNIST 12×12 purification loses 0.02–0.07
  (one restart: an upper bound), detection 0.24–0.30. **D113:** the AT grid is
  {0, 0.25, 0.5}. **D114:** active defences are reported under the adaptive attack.
- D112 on the cluster (compute.md, "Since D112, on G21G01"; logs in
  `pilots/cluster/bench_83da4cc9/`): 10–15× less memory, `--per-gpu` 8 / 4 / 3. The
  before/after table is the evidence for the tk zip-up PR (Martin, another session).
- D106's float32 check on six trained checkpoints (plan.md, "Sampling" step 6;
  `pilots/cluster/f32_83da4cc9/`): one negative weight in 18.1 M rows (−9e-8 of its
  row's maximum), none beyond tolerance, so Gibbs results from D106 on stand. Left:
  the imaginary-part warning (threshold 1e-5) fires at 784 sites (up to 3e-3).
- **D115:** no micro-batches in the full-MNIST studies (whole batches of 512 up to
  d3r80; their cluster memory is unmeasured); a larger batch for every study is open
  (plan.md, Phase 8, "The batch size"). **D116:** debugging on the cluster without
  asking each time (Environment, above).

Nothing of the agent's is running on G21G01 (no `ag-*` screen).

### Start here: capture the adaptive attack (plan.md, "Now", step 1)

**Goal.** The `adaptive` analysis part (D96, D98) runs eager, and costs ≈ 7.6 h per
MNIST 12×12 checkpoint at E2a's trimmed settings (4.3 h for the attack through
likelihood purification, 3.3 h through purifier and detector, 9 min through the
detector alone); full MNIST would be several times that. D114 makes it the attack of
every active defence in Phase 8, so it must be captured first. **Then**, once it is
debugged and runs smoothly on the cluster, the saturation ablation (step 2; ask
Martin first: GPUs for hours).

**Where the time goes** (`bm4tc/analysis/adaptive.py`, `bm4tc/pipeline/analyse.py`):
`adaptive()` → `AdaptivePGD.generate` loops restarts × (steps + 1) in Python; each
step calls `Defense.attack_step` → `_draw`, which (1) purifies through
`_likelihood_purified` → `LikelihoodPurification.purify`, which builds
`self.graphed(model, Graphs(enabled=False))`, i.e. an **eager** 40-step purification
per attack step; (2) runs `log_joint` (and `log_px` for a detector) forward and
backward. The `uq` part already replays the same purifier and PGD from graphs (D92,
D105: ≈ 5.5× at 144 sites, ≈ 7× at 784, `bench_analysis`).

**The pattern to follow** is D105: one graph per *step*, the loop in Python
(`GraphedPGD` in `bm4tc/core/attacks.py`, `GraphedPurification` in
`bm4tc/analysis/purification.py`, made by `Graphs.wrap`; one `Graphs` per part,
`analyse._graphs(cbm, device)`; budgets as device scalars). A sketch to discuss with
Martin before writing it (his rule for interface choices: show options first):
1. The defence's purifier replays: `_defences` hands `Defense` a purify callable built
   on the part's `Graphs` (`purifier.graphed(model, graphs)`), not
   `LikelihoodPurification.purify`.
2. One graph for the attack step of a deterministic defence: from (x, labels, delta,
   purified, best_x, best_rate) to the next delta and the updated best, with the
   loss, `torch.autograd.grad`, the sign step and the projection inside. The purified
   point is computed outside (BPDA: P forward, identity backward), so the step graph
   never contains the purifier's 40 steps (D105: whole-attack graphs crashed at 784
   sites). Host syncs to remove: the masked writes `best_x[better] = ...` (use
   `torch.where`), Python floats in the step (`eps_abs`, `step` as device scalars).
3. Random defences (Gibbs, SGLD; `Defense.random`) keep their sampler eager (tk's
   Python per site; JEM is eager anyway, `_graphs`); only their classifier and
   detector work could replay, inside the EOT loop. Decide with Martin whether that
   is worth it now.
4. `Defense.rates` (the scoring) is a few forwards per batch: capture it if it is
   simple, otherwise leave it.
5. Add `analysis.adaptive.subsample` (D114: a fixed test subsample, as
   `sweep_purify.subsample`; `_subsample` exists) in the same change; it enters the
   part's hash (`SETTINGS`).

**The bar** (D26, D51, D105): captured = eager number for number on fixed seeds, as
`test_captured_analysis_batch_work_is_bit_identical_to_eager` does for PGD and
purification; a new case in `tests/unit/test_graphs.py` (a failed capture poisons
the process in torch 2.1: such tests run in a child process there). Measure with
`tests/bench/bench_analysis.py` extended by an adaptive option (eager vs captured,
time and peak reserved). On CPU every graph is disabled, so the local suite checks
only that the eager path is unchanged.

**No eGPU: debug on G21G01 (D116).** Every CUDA run is remote, so keep the cycle short:
edit and run the CPU suite locally; sync the tracked files to `$R/ag_dev/`
(Environment, above); run `tests/unit/test_graphs.py -k <new test>` there on one free
GPU (check `nvidia-smi` first; on 2026-10-16 another user's process held 1.5 GiB
on GPU 6), short runs directly over SSH, longer ones in an `ag-dev` screen
with a marker file. Small shapes first (MNIST 12×12 d3r20, batch 32), then 784 sites.
Pitfalls known from D90–D112: capture needs no host sync anywhere in the step; one
side stream per `Graphs` for warm-ups (D94) and one pool (D107); the analysis's
per-shape warm-ups cost ≈ 2× eager memory (D110); the forward keeps tk nodes between
calls (D104) and `sample` builds other ones (`_nodes_key`), so check that a graph
captured on the forward's nodes still replays correctly after a Gibbs defence has
called `sample` in the same part (`_defences` puts Gibbs last).

**Confirm on the cluster**, then report to Martin: the full suite's CUDA tests in
`$R/ag_dev/` (`test_graphs`, the seams); one E2a checkpoint (β 0.25,
`pilot_beta12`, trimmed settings, a subsample of 1000) captured against eager on the
same subsample (eager ≈ 40 min for the purifier attack); the time per checkpoint at
144 and 784 sites. From a pushed commit's worktree for the numbers that go into
compute.md. Then the saturation ablation: ask first.

### Handoff: graph capture (2026-10-15, evening)

Done today (session `E2b-and-AT-graph-capture`): E2b read and decided (D108, plan.md
"Beta / full MNIST"); `bench_train_step --snapshot` (memory per graph pool); **D107**,
one memory pool per `Graphs`; **D110**, the Trainer's graphs warm up everything (epoch 1
eager, plus a discarded warm-up validation when epoch 1 does not validate), then
capture. Martin's idea; it replaced Claude's "capture cold after the first capture"
rule (decisions.md D110 has the rejected options).

The mechanism, plainly: a capture's memory comes from the pool, eager code (warm-ups)
from the general cache, and neither can use the other's free memory. Per-graph pools
made reserved the sum of the graphs' peaks (D107 fixes); eager warm-ups after a
capture added their peak to the pool's (D110 fixes). Peak is now about max(eager,
pool), plus ≈ 0.6 GiB per process for the graphs themselves.

Before (G21G01 at `da7185e8`, β 0.5, batch 512, per process, nvidia-smi):

| Shape | Allocated (peak) | Reserved | Per process |
|---|---|---|---|
| MNIST12 d3r40 NAT | 1.82 GiB | 4.24 GiB | 4.57 GiB |
| MNIST12 d3r40 AT | 2.76 GiB | 10.42 GiB | 10.92 GiB |
| full MNIST d3r40 micro 256 NAT | 5.01 GiB | 9.24 GiB | 10.46 GiB |
| full MNIST d3r40 micro 256 AT | 3.63 GiB | 15.33 GiB unallocated | out of memory |

After, on the eGPU (8 GB; full MNIST micro 256 does not fit there even eagerly for
AT, so these are the nearest shapes): MNIST12 NAT batch 512 3.12 GiB per process, AT
batch 256 1.93, AT batch 512 3.36 (D107); full MNIST NAT chunk 256 5.03 / 5.50 GiB
allocated / reserved (eager 5.02 / 5.34), AT micro 128 3.90 / 5.92 (eager 3.89 /
5.88) (D110). Step time: full-MNIST AT micro 128 1.8 s captured vs 5.1 s eager.

Open:
- The analysis keeps per-shape warm-ups (its parts run in stages, so there is no
  early cycle to warm up): ≈ 2× eager (784 sites, batch 64: 2.99 vs 1.52 GiB),
  bounded by `analysis.batch_size`.
- The dominant tensor at 784 sites is B·n·r² complex64 (2.39 GiB at B = 256, r = 40):
  tk's environment-first forward (B·n·r·d, the tensorkrowch track) would shrink it
  ≈ 13×.
- Gibbs on full MNIST in a real analysis (≈ 50 s per run expected), from D104's list.

**Rule from D94, still binding:** studies may run with `trainer.cuda_graph` on only
from the D94 commit on, after the cluster check; never on `5c2db9cd`..`fc9d452e`.
Martin brings tk intuition, not CUDA expertise: explain the GPU side plainly. tk's
source is in `../tensorkrowch` (branch `develop`; our work on `sampling-efficiency`,
and the PR branch `sample-density`).

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

Laptop: the eGPU (RTX 2080, 8 GB) was back from 2026-10-13 to 2026-10-15 (off-site since
2026-10-16: none): the CUDA tests ran here
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
  process (checked on G21G01). tk's `reset` reorders `parameters()`: compare
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
- **D105:** attacks and likelihood purification captured one step at a time (D105 in
  decisions.md: a graph of the whole 40-step PGD at 784 sites crashed or ran out of
  memory).

Done on 2026-10-15 (session `tk sampling rebuild`):
- **D106, `sample`: double layers only where something is summed, fixed sampling
  nodes** (Martin's rule; supersedes D101's open-leg sweep, whose memory grew as
  d_out^k). An environment is double-layer only if it contains a summed site (an
  output node; without `in_condition`, the inputs to the right); each sweep step is a
  few fixed virtual leaves (`SamplingMPS._sample_kernel`) whose tensors are set from
  site to site, so memory does not grow with n except the kept right environments
  (B·r² each, only left of the class in Gibbs purification). Same draws (Gibbs golden
  unchanged); full MNIST one sweep at batch 64: peak 1.2 → 0.43 GiB (r40), 2.0 →
  1.4 GiB (r80), held between calls 1–2 GiB → 15–53 MiB (compute.md, "Gibbs
  purification since D106").
- **Open: float32 rounding of double layers** (plan.md, "Sampling" step 6): a double
  layer is PSD in exact arithmetic only; its error grows like the square of the
  contraction's cancellation (tk develop has it too). Random models: worst −8e-4 to
  −4.6% of a row's maximum, no raise. **Not checked on a trained model** (none on the
  laptop): do it on the cluster before relying on Gibbs results from D106 on (E3).
  Martin: flag it, do not solve it on the fly.
- **tensorkrowch:** branch `sample-density` (the PR for José) rebuilt and
  force-pushed to Martin's fork: 4 commits on develop (tests; per-sample domain; the
  D106 algorithm; uniform fallback). Full `test_mps.py`: only develop's 24 failures,
  torch 2.1 and 2.14. The note to José, `../tensorkrowch/_notes/note_jose_sample_density.md`
  (untracked), explains it; **Martin reads it and decides on the upstream PR.**

**Next (in this order):** see "Start here" above and plan.md, "Now". Still open from
2026-10-15:
1. **tensorkrowch:** the `sample-density` PR (Martin reads the note, then decides); the
   zip-up PR (Martin, another session; the cluster numbers are in compute.md, "Since
   D112, on G21G01"); a norm-keeping `renormalize` (the log-norm as a node; would
   remove bm4tc's `_inline_contraction` override, the accumulator and the special
   forward), a public "contract with reuse", `reset`'s quadratic renaming and its
   reordering of `parameters()`, `@` on batch-only edges, `einsum`'s connected-edge
   rule, edge inheritance; then what goes upstream and how (PR to the fork's develop,
   then to tensorkrowch).
2. The rest of the efficiency list (plan.md): cap HPO workers per cell (also the 16/15
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
