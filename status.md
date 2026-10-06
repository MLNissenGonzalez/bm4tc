# Codebase status: complexity survey

Survey of `bm4tc` at `aa697706` (2026-09-18), branch `ousterhout`. It is written
against the vocabulary of `ousterhout.md`; red flags are named in **bold** where they
apply. The survey only describes what is there. Decisions go into `plan.md`.

**How "actually run" was established.** A sweep counts as run if its analysis output
is committed under `analysis/outputs/`. Raw `outputs/` lives on mathqi only and is not
on this laptop. HPO studies are inferred from filled `???` placeholders and commit
messages. W&B was not queried.

---

## 0. Headline findings

1. **Most of the config tree is not part of the paper.** 506 experiment YAMLs span 12
   datasets × up to 5 embeddings. Committed analyses cover **4 datasets** (spirals,
   mnist_full_r12, ecg200, italypowerdemand), **1 embedding** (legendre) and
   **9 architectures**. 263 of the 506 files still contain `???` and have never
   been runnable as written.
2. **The stop-criterion → metric mapping is re-implemented in 6 places, and they
   disagree** (§5.1). This is the clearest case of **information leakage** in the repo.
   `patch_checkpoint.py` silently mis-selects on `mixed_loss`, and
   `analysis/utils/resolve.py` knows neither `mixed_loss` nor `at_loss`.
3. **The structured-config schema is registered but never applied.** `Config`
   dataclasses are stored in Hydra's ConfigStore, but `configs/config.yaml` never
   inherits `base_config`. Stale or misspelled keys are therefore accepted silently.
   For example, all 41 configs carrying `trainer.adversarial.auto_stack/auto_unbind`
   set keys that `AdversarialConfig` doesn't have.
4. **The two trainers are near-copies** (`nll.py` 409 L, `adversarial.py` 539 L).
   They share the same `_init_best` / `_update` / `_summarise_training` / norm-control /
   epoch bookkeeping, each with its own copy of the metric sets. AT alone has
   **three validation paths** (§4).
5. **The two pipeline ends have no tests.** The suite is healthy (772 passed, 4
   skipped; 34% statement coverage overall). But `experiments/train.py`,
   `analysis/run.py`, `analysis/sweep.py`, every `tools/` script and every batch
   runner are at **0%**. The code that actually turns configs into paper numbers is
   untested.
6. **The output-directory layout is parsed in ~25 files.** The layout is a convention
   defined in `configs/config.yaml` (`hydra.run.dir`). It is decoded again by regexes
   and `.parts` indexing across analysis, tools, batch runners, the JEM baseline and
   notebooks. Several vocabularies coexist: `nat/at`, `nll/adversarial`,
   `dis/gen/adv`, `cls/cls_reg/comb`.
7. **About a third of the tooling was written for one occasion.** It is
   migrations/backfills (≈ 1 000 L) and three separate `batch.py` runners. This
   matches your note that you mostly ran commands Claude composed.
8. **`CLAUDE.md` (and `DEFERRED.md`, `.claude/`) are git-ignored**
   (`.gitignore:71,84,85`). Code comments point at "Budget vocabulary in CLAUDE.md"
   and issue IDs like C2/D1, so canonical design notes exist only on the other
   laptop. This is also why the earlier plan didn't travel.

---

## 1. Size

| Area | Python lines | Notes |
|---|---:|---|
| `src/` (library) | 6 409 | `model.py` 1 109, `datahandler.py` 730, `analysis/uq.py` 844, `analysis/mia.py` 692 |
| `experiments/` | 731 | `train.py`, `run_local.py`, `batch.py`, `tracking.py`, `config.py`, `resolvers.py` |
| `analysis/` | 7 884 | `hpo.py` 1 463, `sweep.py` 976, `utils/statistics.py` 777, `visualize/` 2 146 |
| `tools/` | 2 628 | 7 scripts + a notebook |
| `baselines/jem/` | 3 804 | its own Hydra config tree, trainer, analysis, plots |
| `tests/` | 5 612 | 31 files |
| **Total** | **27 068** | plus 506 experiment YAMLs (28 262 lines, 26% comments) and 1 933 lines of Markdown guides |

There is more analysis code than library code, and more test code than either
`experiments/` or `tools/`.

---

## 2. What experiments were actually run

### 2.1 Committed seed-sweep analyses

| Dataset | Regime | Arch (legendre, c64) | Sweeps | Last |
|---|---|---|---|---|
| spirals | nat | d4r3, d6r4, d10r6 | `a0` + cold α-ladder {0.001, 0.01, 0.1, 0.5, 1} | 0208 |
| spirals | nat | d10r6 (old, non-c64) | `seed_sweep_a0/a05/a1`, two `alpha_curve` | 0506 |
| spirals | at | d10r6 (old) | `seed_sweep_0206` (re-analysed 1809; old numbers above the ceiling, issue D1) | 0206 |
| mnist_full_r12 | nat | d3r10, d3r20, d3r40, d3r100 | `a0` + warm α-ladder {0.01, 0.1, 0.2, 0.5, 1}; d3r20 also cold {0.001, 0.01, 0.1, 1} | 3007 |
| mnist_full_r12 | at | d3r10, d3r20, d3r40 | a0, a001 | 2907 |
| ecg200 | nat | d3r10, d3r20 | `a0` + cold α-ladder {0.01, 0.1, 0.2, 0.5, 1} | 2707 |
| italypowerdemand | nat | d3r10, d3r20 | same as ecg200 | 2707 |
| JEM baseline | nat/at | mlp_h550x480 on mnist_full_r12 | α ∈ {0, 0.01, 0.1, 0.2, 0.5, 1} + AT | 2707 |

`analysis/outputs/seed_sweep/{adv,cls,cls_reg,comb,gen}/` is an older layout. Its last
change was 2026-07-29, and its vocabulary is not produced by any current config.

### 2.2 Pending (configs ready, not yet run or analysed)

- **spirals AT, legendre**: {d4r3, d6r4, d10r6} × {a0, a0001}. Filled 1809; the next
  thing to run.

### 2.3 Present in configs, absent from analyses

| Config subtree | Files | Status |
|---|---:|---|
| moons, circles (all embeddings, nat + at) | 102 | no analysis outputs; last edited 0208 (mass edit) |
| cricketx/y/z, chlorineconcentration, syntheticcontrol (nat + at) | 165 | no analysis outputs; 105 of them still `???` |
| ecg200/at, italypowerdemand/at | 44 | all `???` |
| mnist_full (unresized) | 15 | no analysis outputs since 0610 |
| spirals non-legendre (fourier/hermite/chebychev) | 36 | no analysis outputs |
| legendre arches d3r60, d3r120, d5r100, d15r40, d30r18 | ~130 | no committed seed-sweep analyses |

The paper notebooks (§6.4) read only the datasets in §2.1.

### 2.3a The paper: published version vs. next version

**Source.** `_paper/main.tex` and `_paper/detection.tex`: the TPM 2026 version
(`\documentclass[accepted]{tpm2026}`). Per Martin, this is **outdated**. The NeurIPS
rebuttal version added the JEM baseline and proper upscaling to MNIST r12; that
`.tex` is not here.

**What the published version reports, and where it came from:**

| Paper item | Data / models | Code-side source |
|---|---|---|
| Fig. 1 `dists_with_adv` | spirals, α ladder + AT, p(c\|x) and p(x) | `figures/dists_with_adv.pdf` |
| Main §4: Fig. `alpha_curve_accuracy`, `alpha_curve_nll`, `defense_comparison_eps0.2`; Table `detection` | MNIST r12, **d3r20c64**, α ∈ {0, 0.01, 0.1, 0.2, 0.5, 1}, AT; 5 seeds | `figures/mnist/…` (`detection.tex` is byte-identical to the repo copy) |
| App. spirals quantitative (`legendre_d10D6_2804_*`, `metric_eps02_col`, `summary_attacks`) | spirals **d10r6, real-valued** (not c64); 20 seeds at α∈{0,1}, 5 in between, 8 for AT; Gibbs only here | old `seed_sweep_a0_3005/a05_3005/a1_0506`, `at/…/seed_sweep_0206`, i.e. the pre-c64 sweeps |
| App. capacity | spirals legendre (d,r) ∈ {(4,3),(6,4),(10,6),(30,18)} at α ∈ {0,1} | `figures/spirals/capacity/*`; only d4r3/d6r4/d10r6 c64 have committed analyses (0208) |
| App. embedding | spirals d10r6, α=1, five embeddings | `figures/spirals/embedding/*`; **no committed analysis** for the non-legendre configs |
| App. joint attack | spirals ε=0.2; MNIST ε=0.2 | `COMPUTE_JOINT_ATTACK` in `sweep.py` |
| — | MIA | **not in the paper at all** |
| — | time series | not in the paper |

**The next version, as stated by Martin (2026-10-06):**
- **Main:** MNIST, moving from r12 to **full-resolution MNIST**. Legendre. The NAT
  α-ladder, **AT at α=0 and α=0.1** (the committed sweeps are α=0 and **0.01**; to
  confirm), and a **strengthened JEM baseline**. Probably **several capacities** for
  full MNIST; otherwise capacity is mostly irrelevant.
- **Spirals:** qualitative in the main text, with capacity and embedding appendices.
  The arch is d6r4c64 or similar (to confirm; the published version used d10r6
  real).
- **Time series:** *maybe* added as a stronger case for MPS, so the TS configs are
  not dead yet.

**Discrepancies between paper text and code** (worth fixing before the next
submission):
1. **The detection threshold is calibrated on the test set, not validation.** The
   paper (Implementation Details) says "10th percentile of *validation-set*
   likelihoods". `analysis/run.py:299` passes `classification["test"]` to
   `UQEvaluation.evaluate`, which calibrates τ on that same loader
   (`uq.py:442`). The commit `0a7941d7` notes that this makes the clean FPR exactly
   q. Either the text or the code has to change; using validation is the clean
   choice.
2. **The MNIST AT radius was changed after publication.** The published AT used
   `eps_rel=0.3` (abs 0.6, "relative strength 0.3" in the paper). The current
   configs use 0.1, per their own headers. The paper's AT numbers are therefore
   from a different radius than the current sweeps.
3. **The spirals models in the published version are real-valued d10r6.** The
   0208 sweeps are complex64 d4r3/d6r4/d10r6. The appendix figures need
   regenerating from one consistent set.

**Consequences for the simplification:**
- **In scope and must stay:**
  - NAT and AT training, warm start, norm control
  - likelihood purification, detection, joint attack, Gibbs (spirals only)
  - the four extra embeddings (embedding appendix)
  - the robust-accuracy ceiling
  - the JEM baseline, which will *grow*
- **Possibly in scope:** time series (configs and loaders stay until decided).
- **Out of scope as far as the paper shows:**
  - MIA (`src/analysis/mia.py` 692 L, MIA settings in `sweep.py`). Note that
    `analysis/utils/mia_utils.py` is misnamed: it holds `load_run_config` and
    `find_model_checkpoint`, which every analysis needs. → **Vague name**
  - `analysis/hpo.py` (1 463 L)
  - moons/circles
  - `mnist_full` 28×28 configs in their current form (to be rebuilt for the
    full-MNIST study)
- **Design requirement:** "same study, different dataset or capacity" must be cheap
  (r12 → full MNIST, several bond dimensions, possibly TS). Today that means copying
  dozens of YAMLs per arch (§3).

### 2.4 Stopping criterion: is it uniform?

Your claim: *"all experiments now use as stopping criterion the same used for
training, evaluated on the validation set."* It is **true for every sweep run since
late July, with one systematic exception**:

| Sweeps | `stop_crit` | Matches training objective? |
|---|---|---|
| all AT configs (190) | `at_loss` (the AT objective mirrored on valid, `eval_at`/`eval_split`) | yes |
| spirals c64, all nat (incl. `a0`) | `mixed_loss` | yes |
| mnist_full_r12 warm/cold ladders, ecg/italy cold ladders | `mixed_loss` | yes |
| **mnist_full_r12 `a0` (all arches), ecg200/italy `seed_sweep_a0`, their `hpo` a0 studies** | **`acc`** | **no.** At α=0 the objective is `dis_loss` |
| older / unrun configs | `acc` 107, `gen_loss` 89, `dis_loss` 28 | mixed |

The `a0` checkpoints are **warm-start sources** for every MNIST warm ladder and for
MNIST AT. So the base of the MNIST chain is still selected on accuracy, while
everything downstream selects on loss. Whether to re-run those `a0` sweeps is a
research decision, not a code one. If the rule is truly universal, `stop_crit`
should stop being a parameter (Ousterhout §7: *a parameter that never varies is not
a parameter*).

---

## 3. Config tree (`configs/`)

**Shape.** `experiments/{dataset}/{nat|at}/{embedding}/{arch}/…`, with group defaults
in `born/` (23), `dataset/` (14), `trainer/{nll,adversarial}/` (4) and
`tracking/` (2).

**Coexisting layouts** within `{embedding}/`:

| Pattern | Files |
|---|---:|
| `ARCH/seed_sweep_*.yaml` (flat, older) | 196 |
| `ARCH/seed_sweep/*.yaml` (directory, newer) | 147 |
| `ARCH/hpo/*.yaml` vs `hpo*.yaml` at embedding level vs `ARCH/hpo*.yaml` | 77 / 34 / 7 |
| `grid_sweep*`, `alpha_curve*`, `best_seed*`, `cls_reg*` | 48 |

`experiments/batch.py`, `tools/fill_hpo.py` and `tools/GUIDE.md` document the flat
`{kind}.yaml` convention. `fill_hpo.py` evidently handles the newer layout too (it
was run on 1809), but the docs don't say so. → **Inconsistency, nonobvious code.**

**Knob census** (all 503 non-test experiment YAMLs, via a throwaway YAML-flattening script):

- **Never varies** (1 distinct value in every file that sets it):
  - `optimizer.name` = adam
  - `trainer.adversarial.stop_crit` = at_loss
  - `eval_rob_freq` = 5
  - `adversarial.norm_control.hard_every` = 0
  - `norm_control.debug` = False
  - `evasion.criterion` = NLL, `evasion.num_steps` = 10, `evasion.random_start` = True,
    `evasion.step_size` = None
  - `dataset.split_seed` = 11, `dataset.gen_dow_kwargs.seed` = 25
  - `born.accumulate` = True wherever set; `gen_on_clean` = True wherever set
  - `auto_stack`/`auto_unbind` (dead, see below)

  Every one of these is restated per file. → **Overexposure**, and **change
  amplification** (one decision, up to 190 copies).
- **Duplicated block:** `tracking.evasion` repeats `trainer.adversarial.evasion`
  verbatim in all 41 AT configs that set it. Training never reads
  `tracking.evasion`; only `tools/patch_checkpoint.py:70` does. → **Dead interface.**
- **Free parameters that really vary:** `lr`, `weight_decay`, `clean_weight`,
  `soft_strength`, `alpha`, `max_epoch`/`patience`/`batch_size` (4–8 values each),
  `model_path`, arch and dataset.
- **Comment density.** 26% of YAML lines are comments. The newest configs (spirals AT)
  carry ~40-line headers that restate the same rationale in each of 6–7 sibling
  files. The *why* is valuable, but it has no single home. → **Repetition**; should
  live in one place and be referenced.
- **Schema not enforced** (finding 3). `experiments/config.py` registers `Config`,
  `NLLConfig`, `AdversarialConfig`, … but no YAML inherits them. In practice the
  dataclasses only document defaults, and runtime objects are plain `DictConfig`s.
  Hence the defensive `getattr(cfg, …, default)` sprinkled through `model.py` and
  `analysis/`.
- **Hydra resolvers** (`experiments/resolvers.py`) compute the output path from config
  content (`training_regime`, `dtype_suffix`, `stage_path`, `alpha_suffix`). They are
  the one owner of the layout, but nothing else asks them; everyone else re-parses
  the path (§5.2).

---

## 4. Training code (`src/train/`, `src/utils/train.py`, `experiments/train.py`)

**Structure.** `experiments/train.py` (92 L) picks a trainer from which config node is
non-null and returns `trainer.best[stop_crit]`, negated for `acc`/`rob`, as the
Optuna objective.

**NLLTrainer vs AdversarialTrainer.**

| Concern | NLL | AT | Shared? |
|---|---|---|---|
| `_LOSS_METRICS` / `_ACC_METRICS` / `_VALID_STOP_CRIT` | own copy | own copy (+`at_loss`) | no |
| `_init_best`, `_update`, `_summarise_training` | ~45 L | ~60 L (+`acc_floor`) | no, near-identical |
| norm-control setup in `train()` | 6 L | same 6 L | no |
| epoch loss bookkeeping (`losses`, `nll_losses`, `reg_losses`, `NormTracker`) | yes | yes | no |
| `mixed_loss = α·gen + (1-α)·dis` | inline | inline | also `_mix()` in `utils/train.py`: **3 copies** |
| NaN/collapse handling | retry + diagnostics (~80 L) | none | no |
| `if __name__ == "__main__"` smoke test | yes | yes | duplicates tests |

→ **Repetition**, **conjoined classes**. A change to selection logic (e.g. the
`acc_floor` idea, or the stop-criterion rule) must be made twice.

**AT validation has three paths** (`adversarial.py:400-440`):
1. `gen_on_clean=True` → `eval_split` every `eval_rob_freq` epochs, nothing in between.
   Patience then counts validation events, not epochs.
2. `gen_on_clean=False` on rob epochs → `eval_at`.
3. `gen_on_clean=False` off rob epochs → `eval_metrics`, which has no `at_loss`, so
   `_update` silently skips those epochs.

The two objectives are algebraically identical at α=0 (documented in the config
headers). In practice the non-split objective is used only by the MNIST AT `a0`
sweeps; every newer AT config sets `gen_on_clean: true`. → **Special–general
mixture**; and *patience means different things depending on a flag*, which is
**nonobvious**.

**Rarely used knobs** in `AdversarialConfig`: `acc_floor` (12 configs, MNIST AT
only), `curriculum` off (4), FGM attack branch (0 configs), `stop_crit` other than
`at_loss` (0).

**Model-cache leakage.** Both trainers reach into private model state:
`cbm._log_Z_cache`, `cbm._amp_diag_cache`, `cbm._invalidate_log_Z_cache()`. NLL
always invalidates; AT only when a regularizer is active. Each call site has a
paragraph-long comment explaining why. `ConditionalBornMachine` keeps **four**
separate caches (`_log_Z`, `_log_Z_cache`, `_amp_diag_cache`, `_log_norm_acc`).
Correctness depends on callers invalidating them at the right moment.
→ **Information leakage**, the *hard to describe* red flag.

**Warm-start override** (`experiments/train.py:47-62`). Loading a checkpoint then
patching `cbm.accumulate` and its saved config takes 15 lines of explanation in the
entry point; it belongs in `ConditionalBornMachine.load`.

`experiments/run_local.py` (182 L, 0% coverage) is a second, Hydra-free entry point
with its own config block. It duplicates `train.py`.

---

## 5. Cross-cutting leakage

### 5.1 `stop_crit` → (metric key, direction)

| Location | Knows | Missing / wrong |
|---|---|---|
| `src/train/nll.py:27-29` | dis, gen, mixed, acc, rob | — |
| `src/train/adversarial.py:72-74` | + at_loss | — |
| `experiments/train.py:85-89` | sign flip for acc/rob | — |
| `tools/fill_hpo.py:109-130` | all six | — |
| `tools/patch_checkpoint.py:77-90` | dis, gen, acc, rob | **mixed_loss, at_loss fall through to max `acc/valid`** (bit you on 1809) |
| `analysis/utils/resolve.py:481-507` | acc, dis/"loss", gen, rob | **mixed_loss, at_loss → unresolvable**; rob = mean over all eps |

The JEM baseline has its own selection logic too (`baselines/jem/trainer.py`, default `stop_crit="acc"`). One decision,
seven owners.

### 5.2 Output-path layout and vocabularies

The path is defined once (`configs/config.yaml` + `resolvers.py`) and decoded in
≈ 25 files: 25 hits in `sweep.py`, 17 in `resolve.py`, 10 in `fill_hpo.py`, plus
`batch.py` ×3, `mia_utils.py`, `statistics.py`, `patch_checkpoint.py`,
`delete_runs.py`, the JEM modules, …

Coexisting vocabularies for the same concept:
- `nat`/`at` (paths)
- `nll`/`adversarial` (config groups)
- `dis`/`gen`/`adv` (`sweep.py` `REGIME`, fallback `"dis"`)
- `cls`/`cls_reg`/`comb`/`gen`/`adv` (old analysis outputs)
- descriptors `nat_cold`, `nll_cold`, `nll_pretrained`, `at_pretrained`, `at_warm`,
  … (10 distinct)

→ **Information leakage**, **inconsistency**.

### 5.3 Metric keys

`rob/` is spelled out in 16 modules, `acc/valid` in 6, and `mixed_loss/valid` in 4.
`tools/migrate_metric_keys.py` and the `_lookup_metric` rename-tolerance in
`patch_checkpoint.py` exist because a key was renamed once. `analysis/CSV_SCHEMA.md`
(426 L) documents the CSV columns by hand.

### 5.4 Smaller duplications

- `ProjectedGradientDescent` and `JointProjectedGradientDescent`: `_project`,
  `_random_init` and `_bounded_delta` are copies that differ only in comments.
- `normalizing()` exists in both `src/utils/evasion.py` and
  `src/analysis/purification.py`.
- `datahandler.split_and_rescale(cbm)` needs the *model* to learn the embedding's
  input range. The data layer depends on the model layer.

---

## 6. Experiment and analysis code

### 6.1 Analysis entry points

| Script | L | Role | Driven by |
|---|---:|---|---|
| `analysis/run.py` | 407 | analyse one run dir (acc, losses, rob, ceiling, MIA, UQ) | API + CLI |
| `analysis/sweep.py` | 976 | analyse a seed sweep → `evaluation_data.csv` + `evaluation_summary.txt` | jupytext `# %%` script; a "CONFIGURATION – EDIT THIS SECTION" cell with ~15 module globals + `argparse.parse_known_args` |
| `analysis/gibbs.py` | 547 | Gibbs purification (expensive, split out) | same style |
| `analysis/hpo.py` | 1 463 | HPO exploration plots/importance | same style, 0% covered |
| `analysis/batch.py` | 165 | run `sweep.py` over unanalysed dirs | CLI |
| `analysis/visualize/*` | 2 146 | 8 plotting scripts + another `batch.py` | mixed; `alpha_dist_plots.py` referenced nowhere |

`sweep.py` is a **notebook disguised as a script**. Its settings (evasion budgets,
UQ percentiles, Gibbs knobs, MIA features) are module globals, so the
"interface" for analysing a sweep is "edit the file or trust the defaults". These
are exactly the settings that must be identical across all paper sweeps, and nothing
records which values produced a given `evaluation_summary.txt`. Its default
`SWEEP_DIR` points at a February path in a layout that no longer exists.

### 6.2 Library side of analysis (`src/analysis/`)

`uq.py` (96% covered after the 1809 refactor) and `purification.py` (81%) are the
deepest, best-tested modules. `mia.py` (692 L) is 19% covered, and
`COMPUTE_MIA = False` is the default in `sweep.py`. Is MIA still in the paper?

### 6.3 JEM baseline

`baselines/jem/` (3 804 L) is a parallel mini-project: its own Hydra tree, trainer,
`sweep.py`, `analysis.py`, `plots.py`, `tables.py`, `report.py`, `compare.py`. It
reimplements the stop-criterion, path and metric-key conventions. It is fairly well
tested where it's pure (plots/tables/report > 90%), but `train.py`, `sweep.py` and
`compare.py` are at 0%.

### 6.4 Paper notebooks (`notebooks/`)

- `2dtoy.ipynb`, `mnist.ipynb`, `ts.ipynb`, `coldvswarm.ipynb`, `jem_mnist.ipynb`,
  `transfer_purify_mps_jem.ipynb`.
- They read `evaluation_data.csv` / `gibbs_data.csv` from **hard-coded,
  date-suffixed sweep paths**.
- `2dtoy.ipynb` still points at the old non-c64 `d10r6/seed_sweep_*_3005/0506` and
  `at/…/seed_sweep_0206`, not at the 0208 c64 sweeps.
- Notebook outputs are committed (0.6–2 MB each).

The mapping "paper figure ← sweep" therefore exists only inside notebook cells. This
is the main obstacle to the "one script runs everything for the paper" goal.

---

## 7. Tools (`tools/`, batch runners)

| Script | L | Kind | Last used (commit) | Note |
|---|---:|---|---|---|
| `fill_hpo.py` | 723 | recurring | 1809 | regex-patches YAML in place; docs describe old layout |
| `patch_checkpoint.py` | 242 | recurring | 1809 | **wrong default metric for mixed/at_loss** |
| `delete_runs.py` | 689 | recurring, destructive (local + W&B) | 0610 | 0% tested |
| `migrate_configs.py` | 232 | **one-off** (old `nll/{dis,gen}` → `{nat,at}` layout) | 0601 | done |
| `migrate_metric_keys.py` | 239 | **one-off** (abs → rel budget keys) | 0730 | done |
| `backfill_config_columns.py` | 378 | **one-off** (W&B → CSV columns) | 0731 | done |
| `alpha_lr_interp.py` | 125 | historical (its own guide says so) | 0601 | — |
| `fetcher.ipynb` | — | ad hoc | 0601 | — |
| `experiments/batch.py` | 224 | runner | 0622 | documents flat layout |
| `analysis/batch.py` | 165 | runner | 0610 | — |
| `analysis/visualize/batch.py` | 211 | runner | 0610 | — |
| `experiments/run_local.py` | 182 | alt. entry point | 0729 | duplicates `train.py` |

**≈ 1 100 L is finished one-off migrations.** Three batch runners each discover
directories their own way. The recurring HPO → seed-sweep hand-off is two separate
tools (`fill_hpo` for hyperparameters, `patch_checkpoint` for `model_path`) that
rewrite YAML text with regexes, with different selection logic. → **Shallow
modules**; the workflow "HPO → pick best → seed sweep → analyse" has no single owner.

---

## 8. Tests

**Run:** `python -m pytest -q` (bm4tc env) → **772 passed, 4 skipped, 73 s** on the
eGPU. Bare `pytest` fails collection: the repo root is not on `sys.path` (no
`pytest.ini`/`pyproject.toml`; `pytest.ini` is in `.gitignore`).

**Coverage** (`coverage run -m pytest`, all tests including slow):

| Module group | Coverage |
|---|---|
| `src/analysis/uq.py` 96%, `margin.py` 92%, `utils/train.py` 88%, `train/adversarial.py` 86%, `utils/embeddings.py` 83%, `purification.py` 81% | good |
| `model.py` 75%, `train/nll.py` 75%, `evasion.py` 63%, `datahandler.py` 54% | partial |
| `src/analysis/mia.py` 19%, `analysis/utils/statistics.py` 18%, `resolve.py` 28% | thin |
| `experiments/train.py`, `analysis/run.py`, `analysis/sweep.py`, `gibbs.py`, `hpo.py`, all `tools/*`, all `batch.py` | **0%** |
| **Total** | **34%** (10 306 statements) |

**Character of the tests.**
- Strong unit tests on the maths: CBM contraction, embeddings, budgets, UQ detection
  (mutation-checked).
- `test_experiment_configs.py` composes configs, but no test runs `train.py` end to
  end, even for 2 epochs. No test runs `sweep.py` on a tiny sweep.
- Bugs of the class you hit on 1809 sit exactly in the untested seams: the `0.0000`
  detection that no test could fail, and `patch_checkpoint` selecting on the wrong
  metric.
- `test_cbm.py` (878 L, 59 tests) and `test_adversarial.py` (733 L, 33 tests) are
  large. Some reach into private attributes (`_log_Z_cache`, `auto_stack`), which
  will make simplifying the model more expensive.

---

## 9. Documentation

- 1 933 lines of committed Markdown: `README`, `GUIDE`, `experiments/GUIDE`,
  `analysis/GUIDE`, `analysis/utils/GUIDE`, `analysis/CSV_SCHEMA`, `tools/GUIDE`,
  `baselines/jem/GUIDE`. Plus the git-ignored `CLAUDE.md` / `DEFERRED.md`, which hold
  the budget vocabulary and the C*/D* issue list.
- **Drift examples:**
  - `analysis/GUIDE.md` says `evasion_override` strengths are *absolute*; `run.py`
    says *relative* (`eps_rel`).
  - `model.py:121` docstring says `auto_stack=True`; the code sets `False`.
  - `experiments/train.py` usage example uses the pre-migration path
    `nll/gen/legendre/...`.
  - `sweep.py` usage mentions `analysis/seed_sweep_analysis.py`.
- **Comment style.** Comments are rich in *why*, which is good by Ousterhout §11. But
  cross-module decisions (budget convention, split objective, patience semantics,
  warm-start rule) are re-explained at every site instead of living in one canonical
  place.

---

## 10. Open questions for you (to settle before `plan.md`)

1. **Paper scope.** *Mostly answered, see §2.3a.* Still open:
   - the spirals arch (published: d10r6 real; new sweeps: c64)
   - AT α=0.1 vs 0.01
   - whether TS goes in
   - the rebuttal-version `.tex` (JEM section, r12 upscaling)
2a. **Detection calibration.** Switch τ to validation likelihoods (matches the paper
   text), or change the text to say test?
2. **Stop criterion.** Should "select on the training objective evaluated on valid"
   become a *rule* (no `stop_crit` knob)? And do the MNIST/ECG/Italy `a0` sweeps
   (selected on `acc`) need re-running to comply?
3. **AT objective.** Is `gen_on_clean=true` now the only AT objective? If so, the
   non-split path and two of the three validation paths can go. MNIST AT a0 was run
   with the old path; at α=0 the losses coincide, but validation cadence and patience
   semantics differ.
4. **MIA and `hpo.py`.** Neither appears in the paper. Drop or archive? (Gibbs and the
   joint attack do appear and stay.)
5. **JEM baseline.** Keep it as its own package, or fold its shared conventions
   (paths, metric keys, selection) into the main code?
6. **CLAUDE.md.** Un-ignore it (or move its content into `GUIDE.md`) so design notes
   travel with the repo?
7. **Raw outputs.** Is mathqi `outputs/` the system of record, with only
   `analysis/outputs/` in git? A paper-reproduction script needs to know where
   checkpoints live.
