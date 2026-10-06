# Hand-off: `ousterhout` branch (2026-10-06, after Phase 4)

For the next Claude session (on this laptop or the other one). Read this first, then
`plan.md` (Phase 5 onward). Delete this file once its contents live in `AGENTS.md`
(Phase 7). Write a new handoff only when the context is close to full; don't update it
after every phase.

## What this branch is

Martin is simplifying bm4tc (MPS Born machines for trustworthy classification) along
Ousterhout's *A Philosophy of Software Design*, before re-running every experiment for a
**journal version** of a rejected NeurIPS submission (27079).

| File | What |
|---|---|
| `status.md` | codebase survey (§0–§10); **§11 = decisions D1–D61, the source of truth**; §12 tensorkrowch upstream candidates |
| `plan.md` | target shape (§1), phases 0–9 (§2), untouched core (§3), sizes, risks, open questions |
| `ousterhout.md` | book summary |
| `_paper/` (untracked, keep it so) | TPM 2026 paper, NeurIPS rebuttal drafts; reference only |

`main` is untouched; tag `pre-ousterhout` marks it (local only). **Nothing is pushed;
ask before pushing.** Suite: **453 passed, 3 skipped, ~55 s**
(`conda run --cwd $PWD -n bm4tc pytest -q`).

## Done

- **Phases 0–2:** seam test + benchmark; deletions (HPO analysis, migration tools, MIA
  moved to `analysis/privacy.py`); one metric-key module (`experiments/metrics.py`,
  `quantity/split[/budget]`, D48); selection = argmin `objective/valid` (D8); schema
  enforced (D25).
- **Phase 3** (D15, D18, D31, D40, D51–D55):
  - Before touching code, the numbers that had to change were re-pinned on the old
    code, one commit per reason (D51): seam AT on the split objective with
    `clean_weight=0.5`; `eval_metrics` per sample. The seam also pins **per-epoch
    curves** (`objective/{train,valid}`, AT `rob/valid`) at rel 1e-6. Use the same
    pattern for any refactor that should not change numbers.
  - One `Trainer`/`TrainConfig` (`src/train/trainer.py`): AT = `trainer.evasion` set,
    NAT = `evasion: null`. One `evaluate()` (`src/utils/train.py`). Non-split
    objective, `eval_at`, `acc_floor` deleted. `eval_every` + patience in validations
    for both regimes. Curriculum end is a fraction (`curriculum_end`).
  - `trainer.evasion` is `Optional[Dict]` in the schema (D53): OmegaConf 2.3.1 under
    Hydra's merge crashes filling an `Optional[dataclass]` that is None.
    `evasion_config()` checks it against `EvasionConfig`.
  - D31: model caches keyed on parameter version counters (`_params_key`); public
    `log_Z(recompute=False)`, `forward_stats()`, `log_amp_sq` (dispatch) and
    `log_amp_sq_accumulate` (always safe). No private model access outside `model.py`
    in the trainer. `load(path, accumulate=...)` persists the override.
  - One PGD core (`_PGD`), shared `normalizing`/`project`/`random_in_ball` in
    `src/utils/evasion.py` (purification imports them).
  - Benchmark (RTX 2080, d3r20 c64, batch 256, a=0), ms/step NAT/AT:
    Phase 0 175/1163, now 171/1141.
- **Phase 4** (D4, D14, D16, D17, D19, D29, D47, D56–D61):
  - Model defaults = production model: complex64, randn_eye, std 1e-9, legendre,
    **accumulate on**. `tests/integration/test_accumulate_spirals.py` compares on/off.
    Finding: at lr 1e-2 the two paths drift ~0.2%/epoch (Adam amplifies ~1e-7
    boundary-node gradients that are float noise); at lr 1e-3 < 3e-4.
  - `weight_decay` is 0 in `OptimizerConfig` and is **not a knob** (norm control does
    a similar job). HPO space default: lr only.
  - Config tree: `configs/config.yaml` (picks groups only), `configs/defaults.yaml`
    (study-level: seeds 5, grid defaults incl. alpha ladder and AT eps 0.1,
    per-embedding settings, HPO space, budgets), `configs/studies/*.yaml` (9 studies +
    `tests/seam_{nat,at}`), flat `configs/dataset/`, `configs/trainer/{nat,at}.yaml`,
    `configs/tracking/{online,disabled}.yaml`. `configs/born/`, resolvers,
    `descriptor`/`stage`/`model_path`, `configs/experiments/` are gone.
  - `experiments/runs.py` owns names and run identity: `Study`, `Cell`, `Job`;
    `Job.compose()` (Hydra compose API, key-by-key on the schema; nothing can be set
    below NAT's `evasion: null`); run dir
    `outputs/{study}/{embedding}/{arch}/a{alpha}[/eps{eps}]/s{seed}/`; `run.json`
    written last (identity, warm source + hash, git sha, times, result, config, config
    hash); `claim()` = skip same hash / refuse different hash / `--replace` archives to
    `outputs/{study}/.replaced/{date}/` / restart unfinished; warm start found by
    manifest (`find_runs`); W&B group `{study}/{cell}`, name `s{seed}`.
  - `python -m experiments.train <study> [--cell legendre/d3r40/a0.01] [--seed 3] [--replace]`.
  - Seam test runs from the two seam studies; numbers unchanged since D51.

## Next: Phase 5 (D2, D3, D20–D23, D32, D38)

See `plan.md` Phase 5. Facts that matter:
- **Nothing production can train yet:** studies with an HPO space need
  `configs/hparams/<study>.yaml` (per cell name, every space key) and fail loudly
  without it. So `hpo` + `select` come first. Plan: one Optuna study per cell,
  JournalFile storage under the study's output dir, trials save no checkpoint (D22),
  `select` writes `configs/hparams/<study>.yaml` (D21, D34). The HPO space syntax is
  defined in `defaults.yaml` (`{log: [lo, hi]}`, `[lo, hi]`, `{choice: [...]}`) but no
  sampler implements it yet. HPO trial dirs: under the cell, e.g. `.../a0.01/hpo/`
  (not yet coded; `runs.py` should own it).
- The old analysis still parses old-layout paths and **does not find new runs**:
  `analysis/sweep.py`, `gibbs.py`, `batch.py`, `analysis/utils/resolve.py`,
  `tools/delete_runs.py`. Phase 5 deletes/merges them into `analyse` (reads
  `run.json`, per-run `analysis.json` keyed by settings hash) and `prune`.
  `analysis/run.py::analyze_run(run_dir)` works on new runs (the seam uses it);
  `load_run_config` reads `run.json`.
- D2 (UQ calibration on valid) changes the seam's `detection` numbers by design;
  re-pin in its own commit.
- PGD's constructor still accepts and ignores `criterion`; `EvasionConfig.criterion`
  and three analysis callers that pass it can go.
- Package move `src/ + experiments/ + analysis/ -> bm4tc/{core,analysis,pipeline}`
  with an import-rule test (D32); executor with `--gpus/--per-gpu` (D38); `status`
  verb.

## Open items (need Martin or the HPC)

- **HPC pilots:** PGD-5 vs PGD-10 training on mnist12 d3r40 a0 (D43; the AT preset
  uses 10 until then); norm-control target for cold MNIST runs, 0 vs n·ln d / 2
  (0 for now, D59).
- `mnist_nat` / `mnist_at` have `arch: ???` until `mnist_capacity` picks the rank (D44).
- TS datasets and grids (D6); JEM studies after Phase 6.
- Study training settings (epochs, batch, patience) were copied from the last seed
  sweeps; Martin may want to review them.

## Working with Martin

- He decides; Claude proposes 2–3 options with a recommended one (use the question
  tool) and pushes back when needed. He answers in option notes.
- Commit at each coherent step with a detailed message; never push without asking.
- Record new decisions as D62+ in `status.md` §11.
- **tensorkrowch first (D30):** no tensor-network maths in plain torch outside the model;
  the numerical core stays untouched until Phase 9.
- Runs happen on the lab HPC (SSH, tmux, no AI agents there, D37),
  `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. Local tests on the RTX 2080 eGPU.
- Handoff only when the context is nearly full.

## Environment gotchas (this laptop)

- No bare `python`: `conda run --cwd $PWD -n bm4tc pytest -q` (or `python -m …`).
- Ad-hoc scripts outside the repo need `PYTHONPATH=$PWD` (an unrelated `src` package
  shadows the repo's otherwise). Run the benchmark as a module:
  `python -m tests.bench.bench_train_step`.
- Set `BM4TC_DATA_ROOT` to a scratch dir to keep datasets and outputs out of the repo
  (the seam harness and `tests/unit/test_runs.py` do this).
- `tqdm` around a DataLoader consumes one torch RNG draw: adding/removing a progress bar
  changes RNG-dependent numbers.
- `coverage` is not in the env; install into the scratchpad with `pip --target` if needed.
- `CLAUDE.md`, `DEFERRED.md`, `.claude/` on the other laptop may hold the old plan and
  issue list; reconcile there.
