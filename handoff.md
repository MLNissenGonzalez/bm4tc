# Hand-off: `ousterhout` branch (2026-10-07, after Phase 6)

For the next Claude session (on this laptop or the other one). Read this first, then
`plan.md` (Phase 7 onward). Delete this file once its contents live in `AGENTS.md`
(Phase 7). Write a new handoff only when the context is close to full (or when Martin
asks); don't update it after every phase.

## What this branch is

Martin is simplifying bm4tc (MPS Born machines for trustworthy classification) along
Ousterhout's *A Philosophy of Software Design*, before re-running every experiment for a
**journal version** of a rejected NeurIPS submission (27079).

| File | What |
|---|---|
| `status.md` | codebase survey (§0–§10); **§11 = decisions D1–D74, the source of truth** (plus a note on trainer unification under D72); §12 tensorkrowch upstream candidates |
| `plan.md` | target shape (§1), phases 0–9 (§2; Phases 0–6 done), untouched core (§3), sizes, risks, open questions |
| `ousterhout.md` | book summary |
| `_paper/` (untracked, keep it so) | TPM 2026 paper, NeurIPS rebuttal drafts; reference only |

`main` is untouched; tag `pre-ousterhout` marks it (local only). **Nothing is pushed;
ask before pushing.** Suite: **490 passed, 3 skipped, ~2.5 min**
(`conda run --cwd $PWD -n bm4tc pytest -q -p no:logging`).

Environment: `conda env create -f environment.yml` (everything pinned). The HPC env
needs recreating from it.

## Done

- **Phases 0–5:** seam test + benchmark; deletions; one metrics module (D48); one MPS
  `Trainer`; model-owned caches (D31/D55); config tree `configs/{config,defaults}.yaml`,
  `configs/studies/*.yaml`, `run.json`, study-rooted run dirs (D58); package
  `bm4tc/{core,analysis,pipeline}` with the import rule tested (D32/D68); CLI
  `python -m bm4tc {hpo,select,train,analyse,run,status,prune} <study>` (D62–D67).
- **Phase 6 (D69–D74), commits `8f1d66f4`..`9f6a7b6b`:**
  - **D69 interface** `bm4tc/core/interface.py`: a model gives `log_joint(x)` (B,K) =
    log p(x,c)+C, `log_normalizer()` = C, `input_range`, `out_dim`, `reset()`.
    `class_probabilities`, `log_px`, `nll` (PGD loss), `best_wrong_log_joint` (joint
    attack) are defined once; `analysis/uq.py`, `LikelihoodPurification`,
    `core/attacks.py` use only these. MPS: `log_joint = log_amp_sq`; `log_normalizer`
    caches log Z per parameter values. `attack.generate(model=...)` (was `born=`).
    MPS seam byte-identical.
  - **JEM** in `bm4tc/core/jem/` (`model.py` JEMMLP + matching, `sampler.py`, `train.py`
    JEMTrainer + `JEMConfig`, `purification.py` SGLDPurification). Run config
    `model: mps | jem`, JEM knobs under `jem:`, `trainer:` shared. JEM studies:
    `model: jem`, `grid.embedding: [raw]`, `arch` = the matched MPS (real DoF, D70),
    `warm_from` default `jem_{dataset}_nat`, `model` in the run identity.
  - JEMTrainer mirrors the MPS objective incl. AT (D71), selects on `objective/valid`
    via the now model-generic `objective.evaluate(log_Z=...)` (JEM: standardized SGLD
    estimate; nan at α=0). Shared helpers `attacked_subset`, `curriculum_eps` in
    `core/train.py`. Epoch-record pass-through key is `diagnostics` (was `norm`).
  - Analysis: `analysis.gibbs` → `analysis.sweep_purify` {enabled, sweeps, step,
    subsample, gibbs: {num_bins, batch_size}, sgld: {steps, step_size, noise_std,
    batch_size}}; part `gibbs` (MPS) or `sgld` (JEM, D72) via one UQ block
    (`UQEvaluation.evaluate(sweep_purifier=)`, UQConfig fields `run_sweeps`, `sweeps`,
    `sweep_subsample`; results `sweep_purification_results`). JEM's clean part has no
    `loss_gen`. Likelihood-aware PGD dropped (joint attack is the adaptive one).
  - Studies `jem_mnist12_nat` (cold, α ladder, d3r20/d3r40, rebuttal's JEM HPO space)
    and `jem_mnist12_at` (warm, α {0,1e-2}, HPO lr + clean_weight, SGLD settings
    inherited via the new study key `hparams_from: {study, params}`, D74).
  - Pins: `tests/unit/test_jem.py` (moved model/sampler exact vs old code; trainer =
    old first epoch, later epochs differ on purpose: shared evaluate's disabled tqdm
    costs an RNG draw); JEM seam `tests/e2e/test_jem_pipeline.py` on
    `configs/studies/tests/seam_jem_{nat,at}.yaml`.
  - Deleted from `baselines/jem`: model, sampler, trainer, train (Hydra entry), attacks,
    purification, analysis, device, sweep, hpo_export, **generate** (JEM sample grids),
    **transfer_purify**, `configs/`, their tests. Left: `plots.py`, `tables.py`,
    `compare.py`, `report.py` (CSV only, old metric names) + `GUIDE.md` (marked outdated).

## Next: Phase 7, figures and docs (D27, D28)

See `plan.md` Phase 7:
- `paper.yaml` + `bm4tc/pipeline/figures/` (plan §1 puts it in pipeline): one function
  per figure/table, input = study `results.csv`, output = `figures/…` and `.tex`
  tables; CLI verb `figures paper`. Keys are documented in `bm4tc/pipeline/metrics.py`.
- Replace what the deleted/leftover code did: `baselines/jem/{plots,tables,compare,
  report}.py` (then delete them and `tests/baselines/jem/`), JEM sample grids (old
  `generate.py`; `SGLDSampler.sample_fresh` and MPS `sample_all_classes` exist),
  MPS-vs-JEM transfer-purification figure (old `transfer_purify.py`,
  `analysis/visualize/mnist_transfer_purify.py`).
- Top-level `analysis/{visualize,utils,outputs}` still use the old layout; `analysis/
  CSV_SCHEMA.md` goes (metrics.py documents keys). Notebooks (`notebooks/*.ipynb`, incl.
  `jem_mnist.ipynb`, `transfer_purify_mps_jem.ipynb`) reference removed modules: strip
  outputs, keep for exploration only, or delete.
- Docs: README, one `GUIDE.md`, `AGENTS.md` (+ `CLAUDE.md` → `@AGENTS.md`); fold this
  handoff into AGENTS.md and delete it. DEFERRED.md content → issues or a GUIDE section.
- First decision of Phase 7: which figures/tables the journal needs (status.md §1–§2
  list the submission's; D41 study grids). Ask Martin (options + recommendation).

## Open items (need Martin or the HPC)

- **HPC:** recreate the env; pilots PGD-5 vs PGD-10 AT on mnist12 d3r40 a0 (D43) and the
  cold-MNIST norm target (0 vs n·ln d / 2, D59). They gate only Phase 8 runs.
  `jem_mnist12_nat` then `jem_mnist12_at` can run any time.
- `mnist_nat` / `mnist_at` have `arch: ???` until `mnist_capacity` picks the rank (D44).
- TS datasets and grids (D6); JEM TS studies if wanted.
- Study training settings copied from old sweeps; Martin may review. Gibbs/SGLD
  purification enabled per study (`analysis.sweep_purify.enabled`).
- Trainer unification (note under D72 in status.md): revisit, now that both trainers
  sit in `core/`.
- `tests/unit/test_privacy.py::test_membership_inference_runs_end_to_end` fails when only
  `tests/unit tests/integration` run (order-dependent on the session `cbm` fixture);
  passes in the full suite. Pre-existing.

## Working with Martin

- He decides; Claude proposes 2–3 options with a recommended one (question tool) and
  pushes back when needed. For interface/architecture choices, explain each option in
  chat first (code sketch, effect on pinned numbers, trade-off), then ask.
- Commit at each coherent step with a detailed message; never push without asking.
- Record new decisions as D75+ in `status.md` §11.
- Behaviour changes are pinned before refactoring, one commit per reason (D51); the
  seams' numbers only move on purpose.
- **tensorkrowch first (D30):** no tensor-network maths in plain torch outside the model;
  the numerical core stays untouched until Phase 9.
- Runs happen on the lab HPC (SSH, tmux, no AI agents there, D37),
  `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. Local tests on the RTX 2080 eGPU.
- Handoff only when the context is nearly full or Martin asks.

## Environment gotchas (this laptop)

- No bare `python`: `conda run --cwd $PWD -n bm4tc pytest -q` (or `python -m …`).
  `-p no:logging` keeps failure output readable.
- Ad-hoc scripts outside the repo need `PYTHONPATH=$PWD`. Benchmark:
  `python -m tests.bench.bench_train_step`.
- Set `BM4TC_DATA_ROOT` to a scratch dir to keep datasets and outputs out of the repo.
- `tqdm` around a DataLoader (even `disable=True`), and every extra DataLoader
  iteration, consume a torch RNG draw: changing them moves RNG-dependent numbers.
- Don't run a background seam/benchmark while editing the code it imports; use a
  `git worktree` of HEAD for "before" numbers.
- OmegaConf: a config field named `keys` (or another dict method) is unreachable by
  attribute (`cfg.keys` is the method).
- The full suite takes ~2.5 min; run it in the background.
- `git add` with a pathspec that no longer exists aborts the whole add.
- `CLAUDE.md`, `DEFERRED.md`, `.claude/` on the other laptop may hold the old plan and
  issue list; reconcile there.
