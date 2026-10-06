# Simplification plan

Draft, 2026-10-06, branch `ousterhout`. It implements the decisions D1–D47 in
`status.md` §11. The principles come from `ousterhout.md`. This plan covers the
*how* and the *order*. Where it has to choose something the decisions leave open,
it says so; see §6.

**Guiding constraints**
- **Behaviour is pinned before it is touched.** Phase 0 builds a seam test, and
  every later phase keeps it green, or changes its expected numbers on purpose,
  with the reason in the commit.
- **The numerical core is out of scope** (D30): contractions, `log_partition_function`,
  accumulate, sampling. The only core-adjacent change is cache *ownership* (D31),
  and it is benchmarked.
- **Clean break** (D12). No compatibility shims. The tag `pre-ousterhout` preserves
  everything old.
- **Each phase ends in a working, tested state** and can be merged on its own.
  Phases are ordered so that deletions come before restructuring: less code to move.

---

## 1. Target shape

What the repository should look like at the end of phase 7.

```
bm4tc/
  core/                 # the maths; no files, no configs (D32)
    model.py            # ConditionalBornMachine (tensorkrowch); numerical core unchanged
    embeddings.py       # fourier, legendre, hermite, chebychev1/2
    objective.py        # L_alpha, split AT objective, norm penalty, mix()
    train.py            # Trainer (MPS): fit, validate, select on objective/valid
    attacks.py          # PGD, joint PGD (one shared projection/init core)
    jem/                # JEM model + SGLD trainer + SGLD purifier (D35)
  analysis/             # evaluate any trained model -> metric dicts; imports only core
    robustness.py  detection.py  purification.py  gibbs.py
    ceiling.py     privacy.py   # MIA, standalone (D1)
  pipeline/             # weaving: knows configs, files, runs
    config.py           # schema dataclasses, enforced by Hydra (D25)
    data.py             # datasets, splits, rescaling (takes an input range, not a model)
    runs.py             # names (D14), run.json (D17), study resolution, warm-start lookup (D19)
    metrics.py          # metric keys, objective/{split}, selection rule; documents CSV columns
    stages.py           # hpo, select, train, analyse, prune, status
    executor.py         # job DAG + GPU worker pool (D38)
    figures/            # one function per paper figure/table, driven by paper.yaml
  __main__.py           # CLI: python -m bm4tc <verb> <study>
configs/
  config.yaml           # Hydra entry; inherits the schema
  defaults.yaml         # every constant, once (optimizer, patience, budgets grid, 5 seeds, analysis block)
  datasets/*.yaml       # spirals, mnist12, mnist, ecg200, italypowerdemand, ...
  studies/*.yaml        # one per dataset x regime (x model): grid arch x alpha (x embedding) (D36)
  hparams/<study>.yaml  # written by `select <study>`, never by hand (D34)
paper.yaml              # figure/table -> studies
tests/                  # + test_import_rule.py: core -/-> analysis, pipeline; analysis -/-> pipeline
README.md  GUIDE.md  AGENTS.md  CLAUDE.md (-> @AGENTS.md)
```

**Import rule (D32), enforced by a test:** `core` imports neither `analysis` nor
`pipeline`; `analysis` imports only `core`; `pipeline` and `__main__` may import
everything. The Trainer returns plain names (`objective`, `loss_dis`, …), and
`pipeline/metrics.py` adds the `/{split}` keys and logs them.

**JEM (D35):** its own trainer in `core/jem/`. After training it is analysed by the
same `analysis/` code through a narrow model interface: `log_p_c_given_x`,
`log_p_x` (unnormalised allowed), `save` / `load`. The MPS implements the same
interface.

**The pipeline as the user sees it** (D23):

```
python -m bm4tc hpo      studies/mnist12_nat     # Optuna trials, no checkpoints (D22)
python -m bm4tc select   studies/mnist12_nat     # -> configs/hparams/mnist12_nat.yaml (D21, D34)
python -m bm4tc train    studies/mnist12_nat     # seed runs, one checkpoint each, run.json
python -m bm4tc analyse  studies/mnist12_nat     # per-run analysis.json (cached by settings hash) -> study CSV
python -m bm4tc run      studies/mnist12_nat     # all of the above, skipping finished steps
python -m bm4tc figures  paper                   # every figure and .tex table
python -m bm4tc prune    studies/mnist12_nat --keep-one | --all | --old
python -m bm4tc status   studies/mnist12_nat     # done / running / failed per cell, log paths

# parallel on a multi-GPU node, started by hand in tmux (D37, D38):
python -m bm4tc run studies/mnist12_nat --gpus 0,1,2,3 --per-gpu 2
```

**Interfaces that carry the design** (each owned by exactly one module):

| Decision encoded | Owner | Replaces |
|---|---|---|
| Vocabulary: `nat`/`at`, name prefixes `d3r40 a0.01 eps0.1 s3` (D10, D14) | `runs.py` | ~25 path parsers, `resolvers.py`, `_alpha_suffix` |
| What a run *is* (D17) | `runs.py` (`run.json`) | path regexes, `descriptor`, `stage` |
| Metric keys, `objective/{split}`, selection rule (D8, D11) | `metrics.py` | 6–7 stop_crit mappings, `CSV_SCHEMA.md` |
| Every knob and its default (D25) | `config.py` + `defaults.yaml` | 506 YAMLs restating constants, `getattr(..., default)` |
| Tuned hyperparameters (D16, D21, D34) | `configs/hparams/<study>.yaml` via `select` | `fill_hpo.py`, `patch_checkpoint.py`, `???` placeholders |
| Warm-start source (D19) | `runs.py` lookup | hand-patched `model_path` |
| Attack/purification radii, one grid (D3, D20) | `defaults.yaml` `budgets:` | YAML evasion blocks ×2, `sweep.py` globals |
| Model caches (D31) | `model.py` only | trainer-side `_log_Z_cache` access |

---

## 2. Phases

### Phase 0: Safety net (no behaviour change)

1. Tag `pre-ousterhout` on `main` (local; push when you say).
2. Commit a `pytest.ini` (or a `[tool.pytest]` section) with `pythonpath = .` and the
   `slow` marker, so plain `pytest` works. Un-ignore `pytest.ini`, `AGENTS.md`,
   `CLAUDE.md`.
3. **Seam test** (`tests/e2e/test_pipeline.py`, CPU, < 60 s):
   - tiny spirals (≈ 200 points), legendre d4r3, 2–3 epochs, fixed seed;
   - NAT α=0 → warm AT α=0.01 → analysis of both;
   - pins: `objective/valid`, clean acc, rob at one radius, purified acc, detection
     rate, all with tolerances;
   - the harness function is the only part rewritten as entry points change; the
     expected numbers stay put.
4. **Benchmark** (`tests/bench/`, not in CI): wall time per training step for NAT and
   AT at d3r20 on MNIST12-sized random data. The baseline for D31.
5. Fix nothing yet. Known bugs (D2 calibration, `patch_checkpoint` metric) are
   recorded as expected-to-change values in the seam test.

*Exit:* `pytest` green, seam test green, benchmark numbers in the commit message.

### Phase 1: Delete what nothing needs (≈ 3 000 lines)

- `analysis/hpo.py` (D13)
- `tools/migrate_configs.py`, `migrate_metric_keys.py`, `backfill_config_columns.py`,
  `alpha_lr_interp.py`, `fetcher.ipynb` (D12)
- `experiments/run_local.py`
- `analysis/visualize/alpha_dist_plots.py` (referenced nowhere)
- `simp` embedding + `configs/born/simp`, the FGM branch, the trainers'
  `__main__` smoke blocks
- **MIA (D1):**
  - move `load_run_config` / `find_model_checkpoint` out of
    `analysis/utils/mia_utils.py` into `analysis/utils/runs.py` (later
    `pipeline/runs.py`);
  - strip MIA from `run.py`, `sweep.py`, `CSV_SCHEMA.md` and the integration tests;
  - move `src/analysis/mia.py` to `analysis/privacy.py`, keeping only what the attack
    needs, with its own unit test.
- `analysis/outputs/seed_sweep/{adv,cls,cls_reg,comb,gen}` (old layout, preserved by
  the tag)

*Exit:* suite and seam test green; `git grep` finds no reference to deleted modules.

### Phase 2: One vocabulary, one metrics module, enforced schema

1. **`metrics.py`:**
   - key constructors (`objective(split)`, `rob(eps_rel)`, …);
   - one `select_best(runs)` = argmin `objective/valid` (D8);
   - one `mix(dis, gen, α)`, replacing the three copies.
2. **Trainers log `objective/{split}` plus components** (D11). Remove `stop_crit`
   from the dataclasses, configs and `experiments/train.py` (the Optuna objective
   becomes `best["objective"]`, always minimised).
3. **Enforce the schema** (D25): add `base_config` to the defaults list, then fix every
   stale key it reports (`criterion`, `auto_stack`, `auto_unbind`,
   `tracking.evasion`, …), and remove the defensive `getattr(cfg, …, default)`.
4. **Vocabulary rename** (D10): config groups `trainer/nll` → `trainer/nat`,
   `trainer/adversarial` → `trainer/at`, class and variable names, and the `REGIME`
   values in analysis.

*Exit:* the seam test's selection keys are renamed; numbers unchanged. Composing every
remaining config passes the schema.

### Phase 3: One Trainer (D15, D18, D31)

- **One `Trainer`:** objective `(1-α)·[(1-cw)·L_dis(x_adv) + cw·L_dis(x)] + α·L_gen(x)`;
  NAT is `attack=None` (then `cw` is irrelevant and the bracket is `L_dis(x)`).
- **One validation path:** every `eval_every` epochs, computing `objective/valid` and
  components, plus `rob` when there is an attack. Patience counts validation events,
  in one documented sense. `curriculum` stays (default on); `acc_floor` is deleted (D40).
- **Delete** the non-split objective, `eval_at`, the third (metrics-only) AT path, and
  the duplicate metric-set constants. Record the non-split ablation idea in
  `GUIDE.md` (D18).
- **Collapse/NaN handling** (today NLL-only) applies to both regimes.
- **Cache ownership (D31): design it twice.** (a) version-counter-keyed caches inside
  the model; (b) one public `cbm.on_parameters_changed()` called by the trainer. Pick
  the one that passes the benchmark within noise and leaves no private access outside
  `model.py`. Move the warm-start `accumulate` patch into `ConditionalBornMachine.load`.
- **Small duplications:** PGD and joint PGD share one projection/init core; one
  `normalizing()`.

*Exit:* seam test NAT numbers unchanged (bit-for-bit expected; any RNG-order change is explained in the commit); AT numbers unchanged (the seam
test uses split already); benchmark within noise; `src/train/` ≈ 950 → ≈ 400 lines.

### Phase 4: Configs, names, manifests (D4, D14, D16, D17, D19, D29)

1. **`defaults.yaml`:** every constant that never varied (status §3 census): the
   optimizer, `eval_every`, attack steps / criterion / random start, split seeds,
   `seeds: 5`, the `budgets:` radius grid, the analysis block (D20).
2. **`datasets/*.yaml`:** spirals, mnist12, mnist, ecg200, italypowerdemand, plus the
   TS datasets in scope (D6).
3. **`studies/*.yaml`:** the journal grid only. Agreed 2026-10-06 (D42–D46).

   **Default α ladder** (in `defaults.yaml`, used by every NAT study unless it overrides):
   α ∈ {0, 1e-3, 1e-2, 1e-1, 0.5, 1}. That is log-spaced plus the generative end; 0.2
   is dropped (D45).

   | Study | Grid |
   |---|---|
   | `spirals_nat` | legendre d10r6, default α ladder, cold |
   | `spirals_at` | legendre d10r6, α ∈ {0, 1e-2}, warm, reduced-budget PGD-AT (D43) |
   | `spirals_capacity` | legendre (d,r) ∈ {(4,3),(6,4),(10,6),(30,18)}, α ∈ {0, 1}, NAT, cold, **own HPO per arch**. The (10,6) cells are *re-run on purpose* as a consistency check against `spirals_nat`: same best HPs (within the search's resolution) and the same test metrics (within seed spread) (D42) |
   | `spirals_embedding` | 5 embeddings, d10r6, α ∈ {0, 1}?, NAT (D5) |
   | `mnist12_nat` | d3r{10,20,40}, default α ladder, cold |
   | `mnist12_at` | d3r40 (+ r20?), α ∈ {0, 1e-2}, warm, reduced-budget PGD-AT (D43) |
   | `mnist_capacity` | legendre d3, r ∈ {10, 20, 40, 80, …} up to the 8 GB/cluster memory limit, α ∈ {0, 1}, cold, NAT. Two pass criteria (D44): **(i) α=0:** smallest r whose clean test acc is within 0.5 points of the best r; **(ii) α=1:** smallest r after which doubling r gains < 2 points of clean acc (a plateau rule, not an absolute bar, since α=1 may never get near 0.9 on images). Report both; `mnist_nat` uses the larger. Must finish before `mnist_nat` / `mnist_at` |
   | `mnist_nat` | r from `mnist_capacity`, legendre, default α ladder, cold |
   | `mnist_at` | r from `mnist_capacity`, α ∈ {0, 1e-2}, warm, reduced-budget PGD-AT (D43, D46) |
   | `ts_{dataset}_nat`, `ts_{dataset}_at` | TBD (D6) |
   | `jem_*` | after phase 6; mirrors the MNIST and TS Born studies (same α, same attacks/defences). JEM sized to the parameter count of the MPS arch it is compared against |

   **AT training budget (D43).** Training attack: PGD with 5 steps (down from 10),
   step size 2.5·ε/steps, random start. Validation: PGD 10 steps on the fixed valid
   subset. Evaluation: PGD 40 steps (unchanged). Plus a larger `eval_every` and a
   smaller HPO budget per cell. **Before Phase 4 locks this in:** a one-off pilot on
   mnist12 d3r40 α=0 comparing PGD-5 against PGD-10 training (robust acc at 40-step
   evaluation, wall time). If PGD-5 loses more than ~1 point of robust accuracy, keep 10.

4. **`configs/hparams/<study>.yaml`:** keyed by the study's grid cell `(arch, α[, embedding])`; consumed via a
   resolver at launch. Missing entries fail loudly; only `hpo` runs without them.
5. **`runs.py`:**
   - writes `run.json` (identity axes, study, init source, git sha, resolved
     hparams);
   - builds run dirs with the D14 name rule;
   - gives W&B its group (study + grid cell), run name (seed) and `job_type` from
     the run identity (D47), replacing `_derive_group_key`;
   - finds runs by querying manifests;
   - resolves `init: warm` to the selected α=0 NAT run of the same
     dataset/arch/seed.
6. **dtype default `complex64`** in the schema and the model's fallback. Drop the
   `c64` suffix everywhere. Real dtype stays a schema value; it's no longer the
   default and isn't in any study.
7. **Delete:** the 506 experiment YAMLs, `configs/born/*` (arch becomes `d`/`r` in
   the study grid), `experiments/resolvers.py`, `tools/fill_hpo.py`,
   `tools/patch_checkpoint.py` (superseded by phase 5's `select`), `descriptor`,
   `stage`.

*Exit:* every study composes under the schema; the seam test runs from a study file;
no code outside `runs.py` builds or parses a run path.

### Phase 5: Pipeline verbs and analysis (D2, D3, D20–D23)

1. **Package move** `src/` + `experiments/` + `analysis/` → `bm4tc/{core,analysis,pipeline}` (D32),
   with `__main__.py` exposing the verbs and `test_import_rule.py` guarding the boundaries.
2. **`select`** (D21) and **`train`**: the study grid is expanded into jobs (one per
   cell × seed). Hydra composes each job's config (compose API); **execution is our own
   executor** (D38), not a Hydra launcher.
   - **`pipeline/executor.py`:** a DAG of jobs (warm runs depend on their α=0 run;
     analysis depends on training) run by a worker pool, `--gpus 0,1 --per-gpu k`.
     Each worker pins `CUDA_VISIBLE_DEVICES`. Logs go to one file per job. A job is
     done when its output (checkpoint + `run.json`, or `analysis.json`) exists, which
     makes every verb resumable after a crash or a closed tmux.
   - **`status <study>`** (new verb): shows done/running/failed/pending per cell, with
     the path to the failing job's log. On the cluster there is no agent to ask
     (D37), so this replaces it.
   - **HPO:** one Optuna study per grid cell with a JournalFile storage under the
     study's output dir; `--per-gpu`/`--gpus` workers pull trials in parallel.
3. **`analyse`:**
   - per-run `analysis.json` keyed by a hash of the resolved analysis settings;
     resumes after failure;
   - aggregation to `results.csv` per study, plus a settings dump;
   - merges `run.py` + `sweep.py` + `gibbs.py`. Gibbs and the joint attack are
     switches in the analysis block.
   - **D2:** `UQEvaluation` gets a calibration loader (valid) and an evaluation
     loader (test).
   - **D3:** one `budgets` grid feeds AT training radius, robustness, UQ and
     purification.
4. **`prune`** with `--keep-one`, `--all`, `--old` (D22), replacing
   `tools/delete_runs.py`. Destructive, so it prints the plan and asks for
   confirmation unless `--yes`.
5. **Delete:** `analysis/run.py`, `sweep.py`, `gibbs.py`, all three `batch.py`,
   `analysis/utils/resolve.py`, `wandb_fetcher.py` (if `select` reads manifests only),
   `tools/`.
6. **`datahandler`** takes the embedding's input range instead of the model.

*Exit:* `python -m bm4tc run studies/<seam>` reproduces the seam numbers, except the
detection numbers, which change by design (D2).

### Phase 6: JEM into the pipeline (D24)

- `core/jem/` keeps its own SGLD trainer and SGLD purifier (D35). After training,
  JEM goes through the same `analysis/` code via the narrow model interface
  (`log_p_c_given_x`, `log_p_x`, `save`/`load`); `baselines/jem/attacks.py` and
  `purification.py` are deleted.
- Studies `jem_mnist12_nat`, `jem_mnist12_at` use the same vocabulary, manifests and
  metrics.
- Delete `baselines/jem/{configs,sweep,report,compare,tables,hpo_export}` once
  `figures` covers them.
- **Parameter matching** becomes a study-level statement: e.g. JEM sized to the MPS
  arch it's compared against. This fixes the r20-vs-r40 mismatch from the rebuttal.

### Phase 7: Figures and docs (D27, D28)

- `paper.yaml` + `bm4tc/figures/`: one function per figure/table, input = study
  CSVs, output = `figures/…` and `.tex` tables.
- Notebooks: outputs stripped, exploration only (or deleted).
- Docs: README, one `GUIDE.md`, `AGENTS.md` (+ `CLAUDE.md` → `@AGENTS.md`). Delete
  the other guides and `CSV_SCHEMA.md`. DEFERRED.md content → GitHub issues or a
  GUIDE section.

### Phase 8 (later): Reproduce the journal paper

- **`python -m bm4tc run paper`:** resolves `paper.yaml` into the study DAG
  (NAT α=0 → warm NAT/AT → analysis → figures), runs independent studies in
  parallel through the launcher, and skips finished steps.
- Uses the D38 executor across studies (one DAG for the whole paper). Needs a
  compute estimate per study and the final study list (D6, D7, D41). Runs by hand on
  the lab HPC in tmux (D37); `status paper` reports progress.

### Phase 9 (separate track): tensorkrowch

- Verify each §12 upstream candidate against the current tensorkrowch source.
  Prototype as patches in a fork, run the bm4tc test suite against the fork, and
  only then propose upstream.
- Any change to the bm4tc numerical core happens here, not in phases 1–7.

---

## 3. What stays untouched in phases 0–7

- `ConditionalBornMachine` contractions, `log_partition_function`, `accumulate`,
  sampling, `condition_on_class`, `renormalize_` (D30).
- Embedding maths (only the `simp` deletion).
- `src/analysis/uq.py` detection arithmetic (only the loader split for D2),
  `purification.py`, `margin.py`.
- The maths unit tests: kept, and rewritten only where they read private attributes.

## 4. Expected size

Rough, to be checked after each phase:

| Phase | Δ lines (code + YAML) |
|---|---|
| 1 | −3 000 |
| 2 | −300 |
| 3 | −600 |
| 4 | −27 000 YAML, +500 code |
| 5 | −3 500 (analysis/tools/runners), +1 200 |
| 6 | −1 500 |
| 7 | −1 500 Markdown |

## 5. Risks

- **Tests on 8 GB** (RTX 2080): the seam test and the slow tests must fit; MNIST-scale
  checks use tiny arches.
- **Hydra limits:**
  - per-cell hparams lookup inside a multirun grid → solved with a resolver reading
    `configs/hparams/<study>.yaml`;
  - Optuna sweeps per grid cell → one `hpo` launch per cell, driven by the CLI.
  - Prototype both in phase 4 before deleting the old tree.
- **Schema enforcement** may expose keys that are actually read somewhere through
  `getattr`. Each one is either added to the schema or deleted, decided case by case.
- **Clean break vs. running work:** the spirals AT sweeps (status §2.2) are pending
  on `main`. Run them from `main`, or wait for this branch. Re-running is planned
  anyway (D9).
- **Cache ownership (D31)** is the only change near the numerical core. Benchmark
  gate in phase 3.

## 6. Open questions

1. ~~Package layout~~ → three areas (D32).
2. ~~Scheduler~~ → lab HPC, plain SSH, own executor (D37, D38). Outputs and datasets live on ceph:
   `export BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc` (not local `/data`).
3. ~~W&B~~ → live curves only (D39).
4. **Study grids:** Martin edits the Phase 4 table (D41).
5. ~~`acc_floor` / `curriculum`~~ → keep curriculum, drop acc_floor (D40).
6. ~~CLI~~ → confirmed (D33).
