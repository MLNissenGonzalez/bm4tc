# Hand-off: `ousterhout` branch (2026-10-07, after Phase 5)

For the next Claude session (on this laptop or the other one). Read this first, then
`plan.md` (Phase 6 onward). Delete this file once its contents live in `AGENTS.md`
(Phase 7). Write a new handoff only when the context is close to full (or when Martin
asks); don't update it after every phase.

## What this branch is

Martin is simplifying bm4tc (MPS Born machines for trustworthy classification) along
Ousterhout's *A Philosophy of Software Design*, before re-running every experiment for a
**journal version** of a rejected NeurIPS submission (27079).

| File | What |
|---|---|
| `status.md` | codebase survey (§0–§10); **§11 = decisions D1–D68, the source of truth**; §12 tensorkrowch upstream candidates |
| `plan.md` | target shape (§1), phases 0–9 (§2, Phase 5 marked done), untouched core (§3), sizes, risks, open questions |
| `ousterhout.md` | book summary |
| `_paper/` (untracked, keep it so) | TPM 2026 paper, NeurIPS rebuttal drafts; reference only |

`main` is untouched; tag `pre-ousterhout` marks it (local only). **Nothing is pushed;
ask before pushing.** Suite: **460 passed, 3 skipped, ~2 min**
(`conda run --cwd $PWD -n bm4tc pytest -q -p no:logging`).

Environment: `conda env create -f environment.yml` (everything pinned, verified by a
fresh create + full suite on 2026-10-07; optuna 5.0.0, hydra-optuna-sweeper removed).
The HPC env needs recreating from it.

## Done

- **Phases 0–4:** seam test + benchmark; deletions; one metrics module (D48); one
  `Trainer` (NAT = `evasion: null`); model caches owned by the model (D31/D55);
  production model defaults (complex64, accumulate on); config tree
  `configs/{config,defaults}.yaml`, `configs/studies/*.yaml`, run identity in
  `run.json`, study-rooted run dirs (D58). See status.md §11 D1–D61.
- **Phase 5** (D2, D3, D20–D23, D32, D38; new decisions **D62–D68**):
  - Code lives in **`bm4tc/core`** (model, embeddings, attacks, objective, train),
    **`bm4tc/analysis`** (uq, purification, ceiling, viz, privacy), **`bm4tc/pipeline`**
    (config, data, paths, runs, metrics, stages, executor, analyse, tracking).
    `tests/unit/test_import_rule.py` enforces D32 (core -/-> analysis, pipeline;
    analysis -> core only; bm4tc never imports `analysis/`, `baselines/`, `tests/`).
  - **CLI** `python -m bm4tc <verb> <study> [--cell C] [--seed S] [--gpus 0,1 --per-gpu k] [--replace]`:
    - `hpo`: one Optuna study per cell, JournalFile at `{cell}/hpo/journal.log`,
      trials `{cell}/hpo/t{n}/` (no checkpoint), seed = study's first seed, seeded TPE,
      no pruning (D62).
    - `select`: argmin finite `objective/valid` -> `configs/hparams/<study>.yaml`.
    - `train`: seed runs (unchanged from Phase 4, now `stages.train`).
    - `analyse`: parts `clean`, `uq`, `uq_joint`, `gibbs`, each cached in
      `{run}/analysis.json` under its own hash and seeded on its own (D63); keys
      `quantity/test/{eps}/{q5|d0.1|k3}`, eps 0 = clean (D64); `results.csv` +
      `analysis.yaml` per study.
    - `run`: hpo -> select -> train -> analyse as a DAG of subprocess units
      (`bm4tc/pipeline/executor.py`), logs in `outputs/{study}/.logs/`, per-study lock (D66).
    - `status`: per cell HPO / selected / trained / analysed, plus failed or running
      units with log paths. `prune --keep-one|--all|--old [--yes]`: checkpoints only (D67).
  - **D2:** detection tau calibrated on valid (`UQEvaluation.evaluate(..., calib_loader=)`),
    clean flag rate on test reported. **D3:** an AT study's training radii must be in
    `budgets`. Analysis settings in `defaults.yaml` `analysis:` (the submission's:
    PGD-40, q {1,5,10,20}, delta 0.1, joint on, Gibbs off by default with the MNIST
    settings k {1,3,6,10}, 96 bins, step 0.05, 250 points).
  - Seam (`tests/e2e/test_pipeline.py`) runs `python -m bm4tc run tests/seam_{nat,at}`
    and reads `results.csv`. Re-pinned twice on purpose (D2; then rob from the UQ
    attack's examples + per-part seeds); training curves unchanged since Phase 3.
  - Smaller: Trainer takes `train_loader, valid_loader` (not the DataHandler);
    `DataHandler.split_and_rescale(input_range)`; PGD lost `criterion`;
    `RobustnessEvaluation` gone (use `build_attack(EvasionConfig(...))`).
  - Deleted: `analysis/{run,sweep,gibbs,batch}.py`, `analysis/utils/resolve.py`,
    `analysis/visualize/batch.py`, `tools/`, `src/`, `experiments/`.
  - `tests/e2e/test_stages.py`: hpo -> select -> train (NAT + warm AT), analysis cache,
    a parallel `run` with HPO + `status`. Its parallel test writes and then removes
    `configs/hparams/tests/stages_nat.yaml` (subprocesses can't see the monkeypatch).

## Next: Phase 6, JEM into the pipeline (D24, D35)

See `plan.md` Phase 6. Facts that matter:
- `baselines/jem/` (~5 000 lines incl. tests) is still its own world: `@hydra.main`
  entry point `baselines/jem/train.py` with its own `baselines/jem/configs/` tree
  (model, sampler, trainer, validation_sampler, dataset `mnist_full_r12`), own
  `NaturalTrainer`/`AdversarialTrainer` (`trainer.py`), own PGD (`attacks.py`),
  purification (`purification.py`), analysis/sweep/report/tables/compare/hpo_export.
  Its **HPO configs use the Hydra Optuna sweeper, which is uninstalled now** (D62):
  JEM HPO is broken until it goes through `bm4tc hpo`.
  Imports were rewritten to `bm4tc.*`; tests in `tests/baselines/jem/` pass.
- Target (D35): `bm4tc/core/jem/` = JEM model (`JEMMLP`), SGLD sampler, its trainer,
  SGLD purifier. Shared: studies, manifests, metric keys, select, and **the whole
  `analyse` stage** through a narrow model interface. What the analysis code calls on
  a model today (`bm4tc/analysis/uq.py`, `purification.py`, `core/attacks.py`):
  `class_probabilities`, `marginal_log_probability` (JEM: unnormalised, fine for
  thresholds/ranking), `input_range`, `mixed_nll(x, y, alpha=0)` (PGD loss),
  `amplitudes` (joint attack: MPS-specific; needs a JEM equivalent or a switch),
  `cache_log_Z`, `reset`, `eval`, `to`, `zero_grad`, `out_dim`. Gibbs and the
  likelihood purifier are MPS-specific (plan: "No Gibbs for JEM"). Designing this
  interface is the first decision of Phase 6 (design it twice; Martin picks).
- `bm4tc/pipeline/stages._fit` and `analyse.load` construct a `ConditionalBornMachine`
  directly; a `model:` switch (mps | jem) in the study or run config is needed.
- Parameter matching becomes a study statement (JEM sized to the compared MPS arch;
  `baselines/jem/model.py` has `nearest_uniform_width`, `mps_parameter_count`). The
  rebuttal compared JEM against r20 while some MPS numbers were r40.
- Delete `baselines/jem/{configs,sweep,report,compare,tables,hpo_export}` once Phase 7
  figures cover them (plan); `attacks.py`, `purification.py` go in Phase 6.

## Open items (need Martin or the HPC)

- **HPC:** recreate the env from `environment.yml`; then pilots: PGD-5 vs PGD-10 AT on
  mnist12 d3r40 a0 (D43; preset uses 10), norm-control target for cold MNIST runs
  (0 vs n·ln d / 2, D59).
- `mnist_nat` / `mnist_at` have `arch: ???` until `mnist_capacity` picks the rank (D44).
- TS datasets and grids (D6); JEM studies in Phase 6.
- Study training settings (epochs, batch, patience) copied from old sweeps; Martin may
  review. Whether studies should enable Gibbs (`analysis.gibbs.enabled`) is his call.
- `analysis/{visualize,utils,outputs}`, notebooks, README/GUIDEs still describe the old
  layout and commands; Phase 7 rewrites them.

## Working with Martin

- He decides; Claude proposes 2–3 options with a recommended one (use the question
  tool) and pushes back when needed. He answers in option notes.
- Commit at each coherent step with a detailed message; never push without asking.
- Record new decisions as D69+ in `status.md` §11.
- Behaviour changes are pinned before refactoring, one commit per reason (D51); the
  seam's numbers only move on purpose.
- **tensorkrowch first (D30):** no tensor-network maths in plain torch outside the model;
  the numerical core stays untouched until Phase 9.
- Runs happen on the lab HPC (SSH, tmux, no AI agents there, D37),
  `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. Local tests on the RTX 2080 eGPU.
- Handoff only when the context is nearly full or Martin asks.

## Environment gotchas (this laptop)

- No bare `python`: `conda run --cwd $PWD -n bm4tc pytest -q` (or `python -m …`).
  `-p no:logging` keeps failure output readable (tests log a lot).
- Ad-hoc scripts outside the repo need `PYTHONPATH=$PWD`. Benchmark:
  `python -m tests.bench.bench_train_step`.
- Set `BM4TC_DATA_ROOT` to a scratch dir to keep datasets and outputs out of the repo
  (the seam, `test_runs.py`, `test_stages.py` do this).
- `tqdm` around a DataLoader, and every extra DataLoader iteration, consume a torch RNG
  draw: changing them moves RNG-dependent numbers (it moved a seam pin under D2).
- The full suite takes ~2 min; run it in the background and wait on the output file
  rather than with a 120 s foreground timeout.
- `git add` with a pathspec that no longer exists aborts the whole add; check
  `git status` after a commit that moved files.
- `CLAUDE.md`, `DEFERRED.md`, `.claude/` on the other laptop may hold the old plan and
  issue list; reconcile there.
