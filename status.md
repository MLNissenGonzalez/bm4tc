# Codebase status: complexity survey

Survey of `bm4tc` at `aa697706` (2026-09-18), branch `ousterhout`. It is written
against the vocabulary of `ousterhout.md`; red flags are named in **bold** where they
apply. The survey only describes what is there. Decisions go into `plan.md`.

**How "actually run" was established.** A sweep counts as run if its analysis output
is committed under `analysis/outputs/`. Raw `outputs/` lived on mathqi (Martin's old thesis cluster) and is not
on this laptop. Future runs use the lab HPC (D37). HPO studies are inferred from filled `???` placeholders and commit
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

**The NeurIPS submission and rebuttal (27079, rejected).** Source: `_paper/general_draft_v2.tex`
and the three `response_*_draft_v1.tex` files. What the rebuttal reported, and therefore
what the journal version will at least have to reproduce:

| Rebuttal claim | Models / settings | Committed analysis? |
|---|---|---|
| Headline table: clean / robust / purified / gain | MNIST r12, **MPS d3r40c64**, α ∈ {0, 0.01, 0.1, 1}, AT; JEM α ∈ {0, 0.01}, JEM AT; ε=0.1 pixel units (= `eps_rel` 0.1), δ=0.1; 5 seeds | yes: NAT d3r40 `a0_2506`, `a*_2107`; AT d3r40 `a0_2507`, `a001_2607`; JEM `*_2607`, `at_2707` |
| Capacity at α=1: 0.451 / 0.568 / 0.644 | d3r10, d3r20, d3r40 | yes (`a1_2107` per arch) |
| Accuracy vs. coverage at q ∈ {1, 5, 10, 20}% | d3r40, α ∈ {0, 0.01} | yes (UQ percentiles), **but τ came from the test set** while the text says "calibrated on clean validation data" |
| Purification radius δ ∈ {0.1, 0.15} | d3r40, α ∈ {0, 0.01} | not visible in committed summaries (`delta_rel` default is [0.10]) |
| Gibbs on MNIST: 0.733 → 0.756 after 6 sweeps; timing 5.6 ks (r10) vs 5.8 ks (r20) | MNIST | `gibbs_data.csv` (sweeps setting differs from `sweep.py` default [1, 3, 5]) |
| Joint-likelihood attack: 0.930 purified vs 0.783 | d3r40 α=0.01 | yes (`COMPUTE_JOINT_ATTACK`) |
| JEM SGLD purification 0.539 vs gradient 0.536 | JEM α=0.01 | `baselines/jem/` |
| Class-conditional sample means show digit structure for α>0 | MNIST | `analysis/visualize/mnist_samples.py` |
| "The appendix of the revised paper reports time-series experiments" | ecg200, italypowerdemand, … | NAT only (d3r10/d3r20 ladders); no TS AT |
| Promised, not run: BPDA/EOT adaptive attacks, semi-supervised L_U, full-resolution MNIST | — | — |

Points from the rebuttal that matter for the code:
- **The headline MPS is d3r40, not d3r20.** Capacity matters after all: r10/r20/r40
  are all reported.
- **JEM fairness.** JEM has "about 349k parameters, matched to our r=20 MPS"
  (d3r20 complex ≈ 145·3·20²·2 ≈ 348k real parameters). But the headline compares
  it against **r=40** (≈ 1.4M). A journal reviewer will see this; it is part of
  "strengthen JEM".
- **AT at α=0.01** is what the rebuttal reports ("the adversarially trained model at
  α=0.01"). That settles the earlier α=0.1 question in favour of the committed `a001`
  sweeps.
- **Time series are promised in the appendix**, so TS is in scope, at least for NAT.
- **Adaptive attacks (BPDA/EOT) are the obvious journal-reviewer demand.** Unrolled
  white-box PGD through gradient purification, and BPDA+EOT for Gibbs. Not implemented.

**The journal version, as stated by Martin (2026-10-06):**
- **Main:** MNIST, moving from r12 to **full-resolution MNIST**, possibly at several
  capacities. Legendre. The NAT α-ladder, AT, and a **strengthened JEM baseline**.
- **Spirals:** qualitative in the main text, with capacity and embedding appendices.
  All architectures in **c64**.
- **Time series:** a likely appendix (see the rebuttal).

**Discrepancies between paper text and code** (worth fixing before the next
submission):
1. **The detection threshold is calibrated on the test set, not validation.** The
   paper (Implementation Details) says "10th percentile of *validation-set*
   likelihoods". `analysis/run.py:299` passes `classification["test"]` to
   `UQEvaluation.evaluate`, which calibrates τ on that same loader
   (`uq.py:442`). The commit `0a7941d7` notes that this makes the clean FPR exactly
   q. **Decided: validation (D2, §11).**
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
   - the spirals main-text arch (published: d10r6 real; new sweeps: d4r3/d6r4/d10r6 c64)
   - ~~AT α=0.1 vs 0.01~~ → 0.01, per the rebuttal
   - ~~whether TS goes in~~ → appendix, NAT at least, per the rebuttal; TS AT?
   - which MNIST bond dimensions the journal version needs (rebuttal: 10/20/40; full MNIST: ?)
2a. ~~Detection calibration~~ → validation (D2).
2. **Stop criterion.** Should "select on the training objective evaluated on valid"
   become a *rule* (no `stop_crit` knob)? And do the MNIST/ECG/Italy `a0` sweeps
   (selected on `acc`) need re-running to comply?
3. **AT objective.** Is `gen_on_clean=true` now the only AT objective? If so, the
   non-split path and two of the three validation paths can go. MNIST AT a0 was run
   with the old path; at α=0 the losses coincide, but validation cadence and patience
   semantics differ.
4. ~~MIA~~ → core only (D1). **`hpo.py`**: not in the paper; drop or archive?
5. **JEM baseline.** Keep it as its own package, or fold its shared conventions
   (paths, metric keys, selection) into the main code?
6. **CLAUDE.md.** Un-ignore it (or move its content into `GUIDE.md`) so design notes
   travel with the repo?
7. ~~Raw outputs~~ → lab HPC, `BM4TC_DATA_ROOT` (D37). Old: is mathqi `outputs/` the system of record, with only
   `analysis/outputs/` in git? A paper-reproduction script needs to know where
   checkpoints live.

---

## 11. Decisions taken (2026-10-06)

| # | Decision | Implication for `plan.md` |
|---|---|---|
| D1 | **Cut MIA from the pipeline.** Keep only the core privacy-attack logic, standalone. | Remove MIA from `run.py`, `sweep.py`, the CSV schema and the tests' integration paths. Shrink `src/analysis/mia.py` to the attack itself. Move the misplaced helpers out of `analysis/utils/mia_utils.py` (`load_run_config`, `find_model_checkpoint`, used by `run.py`, `sweep.py`, `gibbs.py`, `hpo.py`) into a properly named run-loading module. |
| D2 | **Calibrate the detection threshold on validation.** | `UQEvaluation.evaluate` takes a calibration loader (valid) separate from the evaluation loader (test). The detection numbers in the rebuttal (accuracy vs. coverage) must be regenerated. |
| D3 | **Make the attack radii consistent.** | One relative budget grid for training AT, evaluation, UQ and purification. It lives in one place, not in each YAML plus `sweep.py` globals. The MNIST AT sweeps trained at `eps_rel` 0.3 (`a0_2507`) vs 0.1 (`*_2907`) need sorting out. |
| D4 | **complex64 is the default dtype.** Real-valued becomes an explicit opt-in or is deleted. | Code default `float32` → `complex64` (`model.py` `_DTYPE_MAP.get(..., torch.float32)`, `MPSInitConfig.dtype=None`). The `c64` suffix in arch names, config file names and output paths (`${dtype_suffix:}`) becomes redundant. The real-dtype branches (`abs_square`, boundary re-cast) shrink. The old real spirals sweeps (`d10r6/*_3005`, `0506`, `0206`) are superseded. |
| D5 | **Spirals:** default legendre d10r6 (c64). The capacity and embedding appendices need all arches and embeddings, **NAT only**. | Spirals keeps a capacity × embedding grid for NAT; AT only at the default arch. |
| D6 | **Time series need AT too.** | TS gets the full NAT + AT treatment. The generic "study" must cover it without per-dataset hand-editing. |
| D7 | **The full-MNIST bond dimension must be tested.** d3r10 is likely enough at α=0 and underperforms at higher α. | Capacity is a sweep axis for MNIST, not a fixed choice. |
| D8 | **The stop criterion is a fixed rule:** select on the training objective evaluated on validation. | `stop_crit` is removed as a parameter. Every trainer logs one `objective/valid` (name to be settled), so selection needs no stop_crit → metric mapping anywhere (§5.1 disappears). |
| D9 | **The journal version needs a lot of re-running.** | Refactor *before* re-running. Backward compatibility with old run dirs and old CSVs is not a constraint; migration shims (rename tolerance, legacy layouts) can go. |
| D10 | **Regime vocabulary is `nat` / `at` everywhere:** paths, config groups (`trainer/nat`, `trainer/at`), class names, CSV columns, docs. | `nll`/`adversarial`, `dis`/`gen`/`adv` and `cls`/`cls_reg`/`comb` disappear. |
| D11 | **Metric keys:** every trainer logs `objective/{split}` (the optimized loss and the only selection key), plus components `loss_dis`, `loss_gen`, `loss_adv`, `acc` and `rob/{eps_rel}`. | Defined once in a metrics module. The stop_crit → metric mapping (§5.1) is deleted. |
| D12 | **Clean break.** New code doesn't read old run dirs, configs or CSVs. Old outputs stay reachable via a git tag (`pre-ousterhout`). | Rename tolerance, legacy layouts and the migration tools go. Confirms D9. |
| D13 | **Delete `analysis/hpo.py`.** | −1 463 L. |
| D14 | **Name rule:** only identity axes (dataset, regime, embedding, arch, alpha, radius, seed) appear in names, written as prefix + literal: `d3r40`, `a0.01`, `eps0.1`, `s3`. lr, wd and clean_weight never appear in names. | e.g. `outputs/mnist12/at/legendre/d3r40/a0.01/eps0.1/s3/`. Prefixes are defined once. |
| D15 | **One Trainer with an optional attack.** NAT is the no-attack case: one validation path, one selection rule, one patience meaning. The non-split AT objective is deleted (D18). | Replaces `NLLTrainer` + `AdversarialTrainer`. |
| D16 | **Config tree = studies + hparams table:** `defaults.yaml` (every constant, once), `datasets/*.yaml`, `studies/*.yaml` (grids), and `hparams.yaml` written only by the HPO selection step. | Replaces the 506 per-sweep YAMLs, `fill_hpo.py` and `patch_checkpoint.py`. |
| D17 | **Run identity via manifest:** each run writes `run.json` (identity axes, study, source checkpoint, git sha, resolved hparams). Nothing parses paths. | Removes path parsing from ~25 files. |
| D18 | **Delete the non-split AT objective.** Only split remains: the attack enters the discriminative term only, and the generative term sees clean data. | **Possible future ablation**: "generative term on adversarial inputs". It trains p(x) to assign high density to adversarial points, so it **works against the detection/purification mechanism** the paper relies on. That is exactly why it would be an informative ablation, and why it can't be the default. Never run at α>0 in any committed analysis. |
| D19 | **Warm starts are a study-level setting, resolved automatically:** `init: cold \| warm`. Warm means the selected α=0 NAT run of the same dataset/arch/seed, found via manifests and recorded in `run.json`. AT is always warm. | No checkpoint paths in configs. |
| D20 | **Analysis settings live in the study config** (radii, δ, percentiles, Gibbs, joint attack), as defaults overridable per study. One radius grid serves training and evaluation (D3). `analyse` is a plain CLI that writes its resolved settings next to its outputs. Notebooks only read CSVs. | `sweep.py` notebook-style globals go. |
| D21 | **One `select` command** picks the best HPO trial per (dataset, regime, arch, α) on `objective/valid` and writes `hparams.yaml`. | Replaces `fill_hpo.py` and `patch_checkpoint.py`; no YAML text editing. |
| D22 | **Checkpoints:** HPO trials save none. Every seed run keeps exactly its selected checkpoint (needed for warm starts and re-analysis). Per-run `analysis.json` is cached by settings hash, so a failed analysis resumes without retraining. Deletion is an explicit `prune`: `--keep-one` (one per study, for visualisation), `--all`, and `--old` (for studies launched more than once, keep only the newest launch and delete older, presumably broken, ones). | Stages hand over files. Nothing is deleted automatically. |
| D23 | **One CLI with few verbs** (tentative; Martin added the prune flags without picking an option): `python -m bm4tc {hpo,select,train,analyse,figures,prune,run} <study>`. `run` chains the stages, resumable. | Replaces `train.py`, `run_local.py`, the three `batch.py`, `run.py`, `sweep.py`, `gibbs.py` and `tools/*` as user-facing entry points. |
| D24 | **JEM is the same pipeline with a different model:** same studies, select/train/analyse, vocabulary, manifests and metric keys. SGLD-specific training and purification stay in their own module. | `baselines/jem` loses its own Hydra tree, sweep, path and metric conventions. |
| D25 | **Enforce the structured-config schema.** Verified 2026-10-06 that Hydra supports it: adding `base_config` to the defaults list makes composition fail on unknown keys. It immediately caught the stale `criterion` key in `trainer/adversarial/pgd_at.yaml`. | Dataclasses become the single interface for every knob; the defensive `getattr(cfg, …, default)` calls go. |
| D26 | **Seam tests first.** Before refactoring, add a CPU end-to-end test (2-epoch spirals study through hpo→select→train→analyse) that pins the outputs. Keep the maths unit tests; rewrite tests that touch private attributes against the public interface. Commit pytest config so plain `pytest` works. | Refactor steps are verified against the seam test. |
| D27 | **Docs:** README (what and how to run), one GUIDE.md (concepts, vocabulary, pipeline, where each decision lives), and a committed agent file. The other guides are deleted; CSV columns are documented in the metrics module. DEFERRED.md content goes to GitHub issues or a GUIDE section. **Agent file naming:** Martin wants the vendor-neutral name. The established cross-tool convention is `AGENTS.md` (plural). Claude Code is guaranteed to read `CLAUDE.md`, so add a one-line `CLAUDE.md` containing `@AGENTS.md` until native support is confirmed. | Un-ignore both in `.gitignore`. |
| D28 | **Figures via a paper manifest:** `paper.yaml` maps each figure/table to its studies; `bm4tc figures paper` regenerates all figures and `.tex` tables from study CSVs. Notebooks are for exploration only, with outputs stripped. | Core of the "one command reproduces the paper" goal. |
| D29 | **5 seeds everywhere**, set in `defaults.yaml`; a study may override upward. | |
| D30 | **tensorkrowch first.** The core stays a `tk.models.MPS` subclass (`ConditionalBornMachine`). Tensor-network operations use tensorkrowch nodes, edges and contractions, not plain-PyTorch reimplementations. Where the project had to go around tk, the workaround is isolated in the model class and recorded as an **upstream candidate** (§12), so it can later become a tk patch. The first refactor phase leaves the numerical core untouched; changes there need their own design pass. | New code outside `src/model.py` must not do tensor-network maths in plain torch. |
| D31 | **Model caches stay; only their ownership changes.** The four caches (`_log_Z`, `_log_Z_cache`, `_amp_diag_cache`, `_log_norm_acc`) remain, so performance is unchanged. The leak is that trainers *read and invalidate* them (`cbm._log_Z_cache`, `cbm._invalidate_log_Z_cache()` after `optimizer.step()`). Target: the model invalidates its own caches, e.g. keyed on the parameters' version counters, or via one public `cbm.on_parameters_changed()` called by the single trainer. To be designed twice and benchmarked, not assumed. | No trainer touches a private model attribute. |
| D32 | **Directory design: three areas.** `core/` = the maths (model, embeddings, objective, Trainer, attacks, `jem/`), with no files or configs. `analysis/` = evaluating any trained model (robustness, detection, purification, gibbs, ceiling, privacy) → metric dicts. `pipeline/` = configs, data, runs, metrics keys, stages, figures; the CLI sits in `__main__.py`. **Import rule, enforced by a test:** `core` imports neither `analysis` nor `pipeline`, and `analysis` imports only `core`. | `core/` is also where tensorkrowch upstream candidates come from (D30). |
| D33 | **The CLI is confirmed** (D23 no longer tentative). | |
| D34 | **Hyperparameters per study:** `configs/hparams/<study>.yaml`, written only by `select <study>` (20–60 lines each; one big file would be ≈550). | |
| D35 | **JEM: own trainer, shared analysis.** JEM keeps its SGLD training. It shares the study format, manifests, metric keys, `select` and the *entire* analysis (same attacks, detection, purification) through a narrow interface: `log p(c\|x)`, `log p(x)` (unnormalised allowed), save/load. Its duplicate `attacks.py`/`purification.py` go; the SGLD purifier stays as a JEM-specific defence. | Makes the rebuttal's "same attacks, same protocols" true by construction, not by convention. Supersedes D24's wording. |
| D36 | **Study = dataset × regime (× model).** Appendix variants (capacity, embedding) are their own studies; `paper.yaml` maps figures to studies. | |
| D37 | **Compute: the lab HPC, not mathqi.** Plain SSH, some nodes with several CUDA GPUs; jobs are started by hand in tmux/screen. **No AI agents on the cluster.** Tests and development run on the local RTX 2080 (8 GB). | The pipeline must be operable by hand from short commands, resumable, and must log to files. Parallelism is ours to provide (no scheduler). Tests must fit 8 GB. |
| D38 | **Parallelism = one local job pool.** Every unit of work (a grid cell of a study: train or analyse one run) is an independent job with dependencies (warm runs wait for their α=0 run). `run` executes the DAG with a worker pool: `--gpus 0,1,…` × `--per-gpu k` (several jobs share one GPU's memory). The same pool parallelises runs within a study and across studies. | Hydra stays for config *composition*; execution uses our own small executor, not Hydra launchers. HPO trials parallelise through Optuna's shared storage. |
| D39 | **W&B: live curves only.** Training logs curves (online on the cluster, disabled in tests); nothing reads back from W&B. | `wandb_fetcher.py` and the W&B branches in tools go. |
| D40 | **AT: keep the curriculum, drop `acc_floor`.** The curriculum moves into defaults (ramp to the full radius by a fixed fraction of `max_epoch`). `acc_floor` was a second selection rule competing with D8. | |
| D41 | **Study grids:** Martin edits the Phase 4 table in `plan.md` directly. | |
| D42 | **`spirals_capacity` re-runs the d10r6 cells with its own HPO**, as a consistency check against `spirals_nat`: best HPs and test metrics should agree within search resolution and seed spread. Every capacity gets its own HPO (no HP transfer, which would confound capacity with tuning). | A built-in reproducibility check. |
| D43 | **AT stays PGD-AT, at a reduced budget:** 5 training PGD steps (from 10), validation PGD 10 steps, evaluation 40 steps, larger `eval_every`, smaller HPO budget. **Fast AT (FGSM + random start) rejected:** it is prone to catastrophic overfitting, is untested on MPS (log-domain, complex gradients), and AT is the strongest baseline in the paper, so a weaker AT would read as a strawman to reviewers. A PGD-5 vs PGD-10 pilot (mnist12 d3r40) gates the step count. | The FGM branch is still deleted in Phase 1. |
| D44 | **`mnist_capacity` has two pass criteria:** α=0, the smallest r within 0.5 acc points of the best; α=1, a plateau (doubling r gains < 2 points), not an absolute bar. `mnist_nat` / `mnist_at` use the larger r. | Thresholds are proposals; adjust after the first look at the curve. |
| D45 | **Default α ladder {0, 1e-3, 1e-2, 1e-1, 0.5, 1}** for all NAT studies (0.2 dropped, 1e-3 added). | One ladder in `defaults.yaml`. |
| D46 | **`mnist_at` added:** r from `mnist_capacity`, α ∈ {0, 1e-2}, warm, D43 budget. | |
| D47 | **W&B grouping by grid cell:** group = study + grid cell, run name = seed, `job_type` = `hpo` \| `train`. W&B averages a group, so each cell shows its seeds as a mean with a spread band. Derived from `run.json` identity, not from the run path. | Replaces `_derive_group_key` (and its `outputs/` ancestor requirement) in Phase 4. |
| D48 | **Metric keys are `quantity/split[/budget]`:** `objective/{train,valid}`, `penalty/train`, `loss_dis`, `loss_gen`, `loss_adv` (split AT: L_dis on x_adv), `acc`, `rob/{split}/{eps_rel}`, `n_rob/valid`, `eps_rel/train`, `norm/*`. `objective/train` excludes the norm penalty so it compares to `objective/valid`. Selection = argmin `objective/valid`; Optuna minimises `best["objective"]`. | Settles the names left open in D8/D11. Analysis keys follow the same rule in Phase 5 (`rob/test/0.1`). |
| D49 | **Delete `configs/experiments/` (except `tests/`) in Phase 2,** not Phase 4. With it go the tools that only operate on it: `tools/fill_hpo.py`, `tools/patch_checkpoint.py`, `experiments/batch.py`, `test_experiment_configs.py`, and the config-rebuild fallback in `analysis/utils/runs.py`. | No migration of ~500 files that Phase 4 deletes. Until Phase 4 studies exist, no production sweep launches from this branch; old HPs stay in the `pre-ousterhout` tag and the §3 census. |
| D50 | **Trainer class renames wait for Phase 3:** Phase 2 renames config groups (`trainer/nat`, `trainer/at`), YAML keys, `REGIME` values and log keys; `NLLTrainer`/`AdversarialTrainer` keep their names until Phase 3 replaces them with `Trainer`/`TrainConfig`. | Avoids a rename that lives for one phase. |
| D51 | **Intended number changes are pinned before a refactor, on the old code, one commit per reason.** For Phase 3: seam AT run on the split objective with `clean_weight=0.5` (all terms exercised), and `eval_metrics` per sample. The seam also pins per-epoch training curves (`objective/{train,valid}`, AT `rob/valid`) at rel 1e-6. | The refactor itself then has to reproduce the numbers exactly; Phase 3 did (no re-pin). |
| D52 | **One `trainer` config field;** presets `trainer=nat/{default,test}`, `trainer=at/{pgd_at,test}` of the same `TrainConfig`. AT = `trainer.evasion` set. | Replaces the `trainer.nat`/`trainer.at` fields and `~trainer/nat`. D10's vocabulary stays as preset names and the `training_regime` value. |
| D53 | **`trainer.evasion` is `Optional[Dict]` in the schema,** checked against `EvasionConfig` by the Trainer (`evasion_config`) and by `test_config_schema`. | OmegaConf 2.3.1 under Hydra's `no_deepcopy_set_nodes` merge asserts when a preset fills an `Optional[dataclass]` field that is `None`. Revisit if OmegaConf fixes it. |
| D54 | **Curriculum end is a fraction:** `curriculum_end` (of `max_epoch`, default 1.0) replaces `curriculum_end_epoch`; default stays off in Phase 3. Phase 4 sets the default (on, and the fraction) with the D43 pilot. | D40's "fixed fraction of `max_epoch`". |
| D55 | **D31 resolved as (a):** caches stamped with the parameters' identity + version counters (`_params_key`); `log_Z(recompute=False)` and `forward_stats()` serve only current values. Public `log_amp_sq` (dispatch on `accumulate`) and `log_amp_sq_accumulate` (always safe). | Benchmark ms/step NAT/AT: Phase 0 175/1163, (b) 179/1157, (a) 171/1141. (a) removes the caller's obligation instead of documenting it. |
| D56 | **Jobs are composed with the Hydra compose API from Phase 4 on;** `@hydra.main`, `hydra.run.dir` and the resolvers are gone. `experiments/runs.py` expands a study into jobs (cell x seed); `python -m experiments.train <study> [--cell] [--seed] [--replace]` runs them in order. | Phase 5 adds verbs and the executor on the same job list. |
| D57 | **Study files** as agreed: `dataset`, `regime`, `init`, `warm_from` (default `{dataset}_nat`), `grid` (embedding x arch x alpha [x eps for AT], arch as `d3r40`), `config:` (fixed run values), `hpo:` (`n_trials`, `space`; `null` = none), merged onto `configs/defaults.yaml`. HPO space default: lr only; weight_decay fixed at 0 as in every recent sweep. Run-level constants are schema defaults; per-embedding model settings (hermite: canonical init) live in `defaults.yaml`. | `dataset/` stays the group name (flat), not `datasets/`. |
| D58 | **Run dirs are study-rooted** (`outputs/{study}/{embedding}/{arch}/a{alpha}[/eps{eps}]/s{seed}/`; D14's example had no study level, which would make `spirals_capacity`'s d10r6 cells collide with `spirals_nat`). Nothing is overwritten silently: same config hash -> skip; different -> refuse, unless `--replace`, which moves the old run to `outputs/{study}/.replaced/{date}/`; no `run.json` -> unfinished, restarted. | `prune --old` (D22) works on `.replaced/` and on `run.json` git sha / launch time. |
| D59 | **Grid choices:** `spirals_embedding` alpha {0, 1}; `mnist12_at` d3r20 and d3r40. The norm-control target for cold MNIST runs stays 0 until an HPC pilot decides between 0 and n·ln d / 2. | |
| D60 | **Model defaults = the production model:** complex64 (D4), randn_eye, std 1e-9, obc, legendre, and `accumulate: true`. A spirals test checks accumulate on vs off (`tests/integration/test_accumulate_spirals.py`). | Training at lr 1e-2 differs by ~0.2% after one epoch between the two paths: Adam turns the boundary nodes' ~1e-7 gradients (float noise in both paths) into steps of ~lr. |
| D61 | **AT preset:** PGD Linf, eps 0.1, 10 steps until the D43 pilot, eval_every 5, curriculum on from 0.01 to the full radius at 0.7·max_epoch (every previous AT study used it). | Settles D54's default. |
| D62 | **HPO:** one Optuna study per grid cell in a JournalFile (`{cell}/hpo/journal.log`), trials in `{cell}/hpo/t{n}/` (curves, no checkpoint). Trials train with the study's **first seed** (warm studies start from that seed's α=0 run). **Seeded TPE** (42 + worker, 6 startup trials), **no pruning**. Trials count until `n_trials` are finished or running (FAIL counts); a crashed launch's RUNNING trials are marked FAIL at the next launch. A journal started under another config, warm start or space is refused unless `--replace` archives it. **optuna ≥ 4**, hydra-optuna-sweeper dropped (JEM's sweeper HPO waits for Phase 6). | `select` writes `configs/hparams/<study>.yaml` (argmin finite `objective/valid`, trial number and value as comments) and refuses while a cell's HPO is unfinished. |
| D63 | **Analysis = parts, each cached and seeded on its own:** `clean` (acc, losses, ceiling), `uq` (PGD at every budget; detection and likelihood purification on the same examples), `uq_joint`, `gibbs`. `{run}/analysis.json` keeps each part under the hash of its settings, the budgets and the run's config hash; a part whose UQ dropped a budget raises. Robust accuracy comes from the detection attack's examples (no second pass). | Seam re-pinned once (rob, detection, purified; new pin: clean flag rate). |
| D64 | **Analysis keys follow D48:** `quantity/test/{eps}/{setting}` with settings `q{pct}`, `d{radius}`, `k{sweeps}`; **budget 0 = clean** (`detect/test/0/q5` is the false-positive rate of the valid-calibrated tau, D2); `_joint` for JOINT_PGD. `results.csv` per study: identity, selected hparams, best `objective/valid`, metrics; `analysis.yaml` beside it. | Documented in `bm4tc/pipeline/metrics.py`. |
| D65 | **D3 enforced:** an AT study's training radii must be in its `budgets`. Analysis settings in `defaults.yaml` are the submission's: PGD-40, q {1,5,10,20}, δ 0.1 with 40 steps, joint attack on, Gibbs off by default (k {1,3,6,10}, 96 bins, step 0.05, 250 test points when on). | |
| D66 | **Execution:** every stage runs as units (subprocesses, one log each in `outputs/{study}/.logs/`) in a pool of `--gpus × --per-gpu` slots; a failure skips what depends on it, the rest goes on. One launch per study at a time (lock with pid). `run` = hpo → select → train → analyse. `status` reads the disk plus `.logs/status.json`. | `select`, `status`, `prune` run in process. |
| D67 | **`prune` deletes checkpoints only** (`--all`; `--keep-one` keeps seed 1's per cell); results, curves, analyses and logs stay. `--old` deletes `.replaced/`. A pruned run counts as finished; `train --replace` retrains it; warm starts and analysis parts that need it say so. | Asks first unless `--yes`. |
| D68 | **Package layout:** `bm4tc/core/{model,embeddings,attacks,objective,train}`, `bm4tc/analysis/{uq,purification,ceiling,viz,privacy}`, `bm4tc/pipeline/{config,data,paths,runs,metrics,stages,executor,analyse,tracking}`, CLI `python -m bm4tc`. The Trainer takes train/valid loaders (DataHandler is pipeline). `PGD` lost its ignored `criterion`, `RobustnessEvaluation` is gone; `DataHandler.split_and_rescale(input_range)`. | `analysis/{visualize,utils,outputs}` stay at top level until Phase 7, `baselines/jem` until Phase 6. Import rule tested. |
| D69 | **Model interface for the analysis (Phase 6, refines D35):** a protocol in `bm4tc/core`: `log_joint(x) -> (B, K)` = log p(x, c) + C (differentiable in x), `log_Z()` = that C (MPS: exact, cached; JEM: 0, so log p(x) is unnormalised), `input_range`, `out_dim`, `reset()`, plus nn.Module's `to`/`eval`/`zero_grad` and save/load. log p(c\|x), log p(x), the PGD loss and the joint-attack loss are defined once on top of it; `uq.py`, `purification.py`, `attacks.py` call only these. MPS: `log_joint = log_amp_sq`, so clean/PGD/detection/purification numbers are bit-identical; the joint attack moves from the direct `amplitudes` path to the accumulate path (float-level change, pinned in its own commit). Gibbs and sampling stay MPS-only. | Chosen over "JEM mimics the MPS API" (fake `amplitudes`, `mixed_nll` with an alpha that must be 0) and an adapter class (the same leak, moved). The joint attack applies to JEM for free. |
| D70 | **JEM parameter matching is the study's `arch`:** a JEM study's grid uses the MPS vocabulary (`arch: [d3r20, d3r40]`), meaning a two-hidden-layer uniform MLP whose real parameter count is nearest to 2x the MPS's complex element count (real degrees of freedom, as the rebuttal's 349k for d3r20). The width is computed from the data dim and class count and recorded in `run.json`. | Fixes the rebuttal's r20-vs-r40 mismatch by construction. |
| D71 | **JEM AT mirrors MPS AT:** one JEM objective `(1-α)[(1-cw)CE(x_adv) + cw·CE(x)] + α·L_gen` (contrastive, SGLD negatives) with the shared PGD; studies `jem_*_at` warm from the α=0 `jem_*_nat` runs, α ∈ {0, 1e-2}, selected on `objective/valid` (D8). α=0 is the old MLP-AT (selected on rob before); α=1e-2 (JEM+AT) is new. | |
| D72 | **JEM-only analysis = SGLD purification:** an analysis part `sgld` (projected SGLD sweeps, keys `purify_sgld/test/{eps}/k{n}`, `recovery_sgld/...`) runs for JEM only, the counterpart of `gibbs` for the MPS. The likelihood-aware PGD is dropped: the shared joint attack is the adaptive attack for both models (D35). | |
| D73 | **JEM in the pipeline (Phase 6.2, implements D70–D72):** run config `model: mps \| jem`; JEM's knobs in one `jem:` node (`jem.model` (match_in_dim/bond_dim from the cell's arch), `jem.sampler`, `jem.valid_sampler`, `energy_l2`, `grad_clip`, `input_noise_std`), the `trainer:` node shared. A study's `model: jem` needs `grid.embedding: [raw]` (no embedding; run dirs `{study}/raw/{arch}/...`), `warm_from` defaults to `jem_{dataset}_nat`; `identity` gains `model`. `JEMTrainer` (`bm4tc/core/jem/train.py`) shares `objective.evaluate` (now model-generic: `log_Z` given = JEM's SGLD estimate, nan at α=0), `attacked_subset`, `curriculum_eps`, the epoch record and selection with the MPS Trainer; the record's pass-through key is `diagnostics` (was `norm`). Analysis: `analysis.gibbs` becomes `analysis.sweep_purify` {enabled, sweeps, step, subsample, gibbs: {num_bins, batch_size}, sgld: {steps, step_size, noise_std, batch_size}}; part `gibbs` (MPS) or `sgld` (JEM) through one UQ block (`UQEvaluation.evaluate(sweep_purifier=)`); JEM's `clean` has no `loss_gen` (no exact log Z). Deleted: `baselines/jem/{model,sampler,trainer,train,attacks,purification,analysis,device,sweep,hpo_export,generate,transfer_purify}.py`, `configs/`. | Kept until Phase 7: `baselines/jem/{plots,tables,compare,report}.py` (CSV-only, old metric names). The new JEM trainer matches the old one's first epoch; from epoch 2 it differs on purpose (the shared evaluate's tqdm costs an RNG draw). JEM seam `tests/e2e/test_jem_pipeline.py`. |
| D74 | **JEM NAT is cold** at every α, like the MPS studies (the rebuttal warm-started α>0 from α=0; revisit only if HPO shows SGLD diverging from scratch). **JEM-AT inherits its training SGLD** from the matching `jem_mnist12_nat` cell (same arch and α) through a new study key `hparams_from: {study, params}`; inherited values apply before the study's own hparams and enter the config hash. | Its HPO tunes lr and clean_weight only, as `mnist12_at`. |
| D75 | **Figures (Phase 7, refines D28):** paper manifests `configs/papers/<paper>.yaml` (items: name -> kind + studies; `eps: [..]` = one output per budget), `python -m bm4tc figures <paper> [--item X]`, package `bm4tc/pipeline/figures/`. Seven generic kinds instead of one function per figure: `curve` (metric vs alpha, bond_dim, or `{x}` in the key: eps, sweeps), `bars`, `coverage`, `table` (booktabs `.tex`; `minus` for gains/deltas, `best` bolds) from `results.csv`; `density`, `samples`, `transfer` from the first seed's checkpoint (kept by `prune --keep-one`), through the D69 interface. A model = study + `where` on identity columns and must select exactly one grid cell (mean +- std over its seeds). A failing item is reported, the others are drawn. `paper.yaml` mirrors TPM + rebuttal: main = full MNIST (`mnist_*`, plus `jem_mnist_*` still to be written), appendix spirals, mnist12. Notebooks are deleted (tag keeps them). | Replaces `baselines/jem/{plots,tables,compare,report}`, `analysis/visualize`, the notebooks. |
| D76 | **Dataset cache files are named by every generation setting** (`spirals_n4000_s25_noise0.5.npz`, `mnist_full_s42_r12.npz`), and the cache dir is read from the data root at call time. Before, `spirals.npz` was reused whatever its size (the repo-local cache held 64 points from a unit test). | Seam numbers unchanged (fresh data roots); caches on the HPC are regenerated once. |

**Noted, not decided: one Trainer for both models.** D69's protocol covers analysis only; JEM keeps its own trainer (D35). The two training loops share epochs, `eval_every`, patience on `objective/valid`, best-state keeping, logging, checkpointing, and the AT objective shape `(1-α)[(1-cw)L_dis(x_adv) + cw·L_dis(x)] + α·L_gen` with the same PGD. A unified Trainer would take a model that supplies `loss_dis` and `loss_gen` (MPS: exact via log Z; JEM: contrastive with SGLD negatives) and its own validation of `L_gen`. *Potential:* one selection rule, one AT path, one set of logged keys by construction; roughly 300 lines of JEM trainer gone; JEM+AT comes for free. *Risks:* (1) the MPS Trainer is full of MPS-specific machinery: norm control and `renormalize_`, log Z caches (D31), collapse/NaN recovery via `cbm.reset()`, `forward_stats` diagnostics, tensor-list best state, `prepare()` tracing; a model hook for each makes the Trainer interface wide (the shallow-module trap); (2) JEM's `L_gen` is not a loss value but a gradient estimator: its validation needs a standardized, seeded SGLD chain plus a replay buffer that is part of the best state and the checkpoint, which has no MPS analogue; (3) any change to the MPS Trainer risks the seam's bit-for-bit NAT and AT pins (D51) and the D31 benchmark; (4) JEM's extra knobs (energy_l2, grad_clip, input noise, sampler config) would enter the shared TrainConfig or need a model-specific section. Revisit after Phase 6, when both trainers exist side by side in `core/` and the overlap can be measured; only the loop skeleton (epochs/patience/best/log) looks safe to share.

---

## 12. tensorkrowch upstream candidates (tensorkrowch 1.1.6)

Places where `src/model.py` / `src/utils/embeddings.py` extend or work around
tensorkrowch. These are candidates for patches to the library. Each needs verifying
against the current tk source before proposing it.

| Area | In bm4tc | tk today | Candidate patch |
|---|---|---|---|
| Log-domain norm | `log_partition_function`: zip-up (ladder) contraction with per-site `log(norm)` accumulation and renormalisation; peak memory O(D²·d) instead of O(L·D⁴) | `MPS.norm(log_scale=…)` | zip-up order plus log accumulation as the `norm` implementation |
| Complex norm | conjugates the bra on *every* call (the reuse path previously computed Σψ² instead of Σ\|ψ\|²) | check whether `MPS.norm` has the same reuse-path issue | bug fix if so |
| Overflow-safe amplitudes | `_inline_contraction` override with per-site bond-norm accumulation (`_log_norm_acc`, `accumulate=True`) | `_inline_contraction(mats_env, renormalize, from_left)` | expose the accumulated log-scale from `renormalize=True` contractions |
| Orthonormal polynomial embeddings | Legendre, Hermite, Chebyshev I/II (`src/utils/embeddings.py`) | `poly`, `basis`, `fourier`, `unit`, `discretize` | add orthonormal polynomial families |
| Born-machine API | class-conditioned amplitudes, `marginal_log_probability`, `condition_on_class`, sequential sampling, Gibbs conditionals | `MPS` / `MPSLayer` (no Born-rule layer) | a Born-machine model class, the long-term goal |
| Shared-parameter auxiliary net | `norm_net = self.copy(share_tensors=True)` plus re-casting boundary nodes' dtype/device after `copy()` | `copy()` creates boundary nodes as float32 on CPU | `copy()` should preserve dtype and device |

