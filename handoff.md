# Hand-off: `ousterhout` branch (2026-10-06, after Phase 2)

For the next Claude session (on this laptop or the other one). Read this first, then
`plan.md` (Phase 3 onward). Delete this file once its contents live in `AGENTS.md`
(Phase 7). Write a new handoff only when the context is close to full; don't update it
after every phase.

## What this branch is

Martin is simplifying bm4tc (MPS Born machines for trustworthy classification) along
Ousterhout's *A Philosophy of Software Design*, before re-running every experiment for a
**journal version** of a rejected NeurIPS submission (27079).

| File | What |
|---|---|
| `status.md` | codebase survey (§0–§10); **§11 = decisions D1–D50, the source of truth**; §12 tensorkrowch upstream candidates |
| `plan.md` | target shape (§1), phases 0–9 (§2), untouched core (§3), sizes, risks, open questions |
| `ousterhout.md` | book summary |
| `_paper/` (untracked, keep it so) | TPM 2026 paper, NeurIPS rebuttal drafts; reference only |

`main` is untouched; tag `pre-ousterhout` marks it (local only). **Nothing is pushed;
ask before pushing.** HEAD = `a087c8cc`. Suite: **462 passed, 4 skipped, ~50 s**.

## Done

- **Phase 0** (`92fb5b3b`): `pytest.ini`; seam test `tests/e2e/test_pipeline.py` (CPU,
  ~27 s, deterministic; tiny spirals, legendre d4r3, NAT a=0 → warm AT a=0.01 →
  `analyze_run`; pins objective, acc, dis_loss, rob, detection, purified acc;
  `detection` is expected to change with D2). Benchmark
  `python -m tests.bench.bench_train_step` (RTX 2080, d3r20 c64, batch 256, a=0:
  **NAT 175 ms/step, AT PGD-10 1163 ms/step**), the D31 gate.
- **Phase 1** (`624b31e3`, `19d34ff3`, `f04a882d`): deleted `hpo.py`, migration tools,
  `run_local.py`, `simp`, FGM, trainer smoke blocks, old `analysis/outputs/seed_sweep/*`,
  the 7 unused `plot_*` in `statistics.py`. MIA out of the pipeline:
  `analysis/privacy.py` (standalone, shrunk, numerically equal to the old one) and
  `analysis/utils/runs.py` (`load_run_config`, `find_model_checkpoint`).
- **Phase 2** (`82b02822`, `24474cfe`, `8e4b820c`, `a087c8cc`):
  - D49: `configs/experiments/*` deleted except `tests/{nat,at}.yaml`, plus `fill_hpo.py`,
    `patch_checkpoint.py`, `experiments/batch.py`, and the path-based config fallback.
    Nothing launches a production sweep from this branch until Phase 4 studies exist.
  - D8/D11/D48: both trainers select on `objective/valid` (always minimised);
    `stop_crit` is gone. Trainers pass `on_epoch_end(epoch, {"train": {...}, "valid":
    {...}, "norm": {...}})` with plain names (`rob` as `{eps_rel: value}`);
    `experiments/metrics.py` (`key`, `flatten`, `flatten_epoch`) builds every logged
    key `quantity/split[/budget]`. `experiments/train.py` returns `best["objective"]`.
    `mix()` is public in `src/utils/train.py` (core). `eval_split`/`eval_at` return
    `objective, loss_dis, loss_gen, loss_adv, acc, rob, n_rob`.
  - D10: groups `trainer/nat`, `trainer/at`; config fields `trainer.nat`, `trainer.at`.
    Class names (`NLLTrainer`, `AdversarialTrainer`) wait for Phase 3 (D50).
  - D25: `base_config` first in `configs/config.yaml`. `ConditionalBornMachine` and
    `DataHandler` merge their config onto the schema once (no defensive `getattr`).
    `tests/unit/test_config_schema.py` composes every config option.
  - `resolve.py` shrunk to the two path resolvers (nat/at only); `wandb_fetcher.py`
    deleted (D39); its two helpers live in `tools/delete_runs.py`.

Plan deviations already agreed or noted: `select_best` not written yet (no caller
until Phase 5's `select`); docs were patched minimally (Phase 7 rewrites them);
notebook runbook cells (2dtoy, mnist) still print the pre-refactor workflow on purpose.

## Next: Phase 3, one Trainer (D15, D18, D31, D40, D50)

See `plan.md` Phase 3. Facts that matter for it:
- `src/train/nll.py` 331 L, `src/train/adversarial.py` ~430 L, `src/utils/train.py`
  582 L (`eval_metrics`, `eval_split`, `eval_at`, `eval_rob`, `mix`, norm control).
- Exit: seam NAT numbers bit-for-bit (explain any RNG-order change), AT numbers
  unchanged. **Caveat:** the seam AT run uses `trainer/at=test`, which has
  `gen_on_clean: false` (the non-split objective that D18 deletes), so deleting it
  *will* change the AT seam numbers. Either switch the seam AT run to split first
  (re-pin in its own commit, reason stated) or accept the change; ask Martin.
- `eval_metrics` averages per-batch means; `eval_split`/`eval_at` average per sample.
  One validation path must pick one (per sample is correct).
- `acc_floor` still exists (tests in `tests/unit/test_adversarial.py`); D40 deletes it.
- NaN/collapse handling exists only in the NAT trainer; it should cover both.
- The warm-start `accumulate` patch is still in `experiments/train.py` (moves into
  `ConditionalBornMachine.load`).
- D31 cache ownership: design it twice, benchmark against the numbers above.
- `AdversarialTrainer._init_best` now requires `eval_rob_freq >= 1` in every mode.

## Known gaps and findings

- An explicit `+key=value` Hydra override bypasses the schema (force-add). Plain
  overrides and YAML keys are checked.
- Training needs an `outputs/` ancestor in the run dir (W&B group from path); D47
  replaces it in Phase 4 (group = study + grid cell, name = seed, `job_type`).
- Still open (Phase 4): `spirals_embedding` α set; `mnist12_at` arches (r40 only or also
  r20); TS dataset list and grids (D6); the norm-control target for cold MNIST runs
  (0 or `n·ln d / 2`); PGD-5 vs PGD-10 pilot on mnist12 d3r40 before locking D43.

## Working with Martin

- He decides; Claude proposes 2–3 options with a recommended one (use the question
  tool) and pushes back when needed. He answers in option notes.
- Commit at each coherent step with a detailed message; never push without asking.
- **tensorkrowch first (D30):** no tensor-network maths in plain torch outside the model;
  the numerical core stays untouched until Phase 9.
- Runs happen on the lab HPC (SSH, tmux, no AI agents there, D37),
  `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. Local tests on the RTX 2080 eGPU.

## Environment gotchas (this laptop)

- No bare `python`: `conda run --cwd $PWD -n bm4tc pytest -q` (or `python -m …`).
- Ad-hoc scripts outside the repo need `PYTHONPATH=$PWD` (an unrelated `src` package
  shadows the repo's otherwise). Run the benchmark as a module (`python -m tests.bench…`).
- A training subprocess run dir must sit under `.../outputs/...`; set
  `BM4TC_DATA_ROOT` to a scratch dir to keep datasets and outputs out of the repo (the
  seam harness does this).
- `tqdm` around a DataLoader consumes one torch RNG draw: removing or adding a progress
  bar changes RNG-dependent numbers (it did when MIA moved).
- `coverage` is not in the env; install into the scratchpad with `pip --target` if needed.
- `CLAUDE.md`, `DEFERRED.md`, `.claude/` on the other laptop may hold the old plan and
  issue list; reconcile there.
