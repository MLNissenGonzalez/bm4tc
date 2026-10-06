# Analysis CSV Schema Reference

Every analysis script writes one or more CSV files to `analysis/outputs/`. This document describes the schema of each, what is *not* stored but can be reconstructed, and the conventions needed to interpret the numeric column names.

---

## Common conventions

### Epsilon / delta column names

Attack budgets and purification radii appear as **relative values** in column names — fractions of the input domain width (e.g. `eval/test/rob/0.1`). See the "Budget vocabulary" section of `CLAUDE.md` for the full convention; in short, `eps` is the attacker's budget, `delta` the defense's radius, and both are authored relative.

Convert to absolute model-domain units with `range_size`, which depends on the embedding:

| Embedding | Input range | `range_size` |
|-----------|-------------|-------------|
| `fourier` | (0, 1) | 1.0 |
| `legendre` | (−1, 1) | 2.0 |
| `hermite` | (−4, 4) | 8.0 |
| `chebychev1` | (−0.99, 0.99) | ~1.98 |
| `chebychev2` | (−1, 1) | 2.0 |

The scripts use `EPS_REL = [0.05, 0.1, 0.15]` and `UQ_CONFIG["delta_rel"] = [0.1]`, so the epsilon columns are `0.05`, `0.1`, `0.15` and the purification suffix is `0.1` — **on every embedding**. For legendre those are absolute 0.1 / 0.2 / 0.3 and 0.2.

Because the data is rescaled onto `cbm.input_range` at load time, the relative value is also the budget in the data's *own* units: on MNIST `eps_rel = 0.1` is 0.1 in `[0,1]` pixel space.

Every CSV carries two provenance columns:

| Column | Meaning |
|---|---|
| `range_size` | `hi − lo` for this sweep's embedding. Absolute budget = `eps_rel × range_size`. |
| `eps_unit` | `"rel"`. Marks the convention, so relative-keyed files are distinguishable from pre-migration absolute-keyed ones. |

```python
rs = df["range_size"].iloc[0]                       # 2.0 for legendre
abs_eps = [round(e * rs, 6) for e in [0.05, 0.1, 0.15]]
# → [0.1, 0.2, 0.3]
```

> **Historical CSVs.** Files written before this convention landed key their columns by
> *absolute* epsilon (`rob/0.2` where the same budget is now `rob/0.1` on legendre) and
> have neither provenance column; re-run the analysis to regenerate them.
> `baselines/jem/` is deliberately **not** relative-keyed — JEM has no embedding, so its
> budgets are absolute by design; `compare.py` refuses to merge the two conventions.

### `eval/uq_adv_acc/{eps_rel}` vs `eval/test/rob/{eps_rel}`

When UQ is enabled, the test-split rob columns are *copied from* the UQ adversarial accuracy cache rather than re-running PGD. For any budget in both the UQ attack budgets and the evasion config budgets, `eval/test/rob/{eps_rel}` == `eval/uq_adv_acc/{eps_rel}`. They are stored as separate columns for clarity.

---

## Type 1: `sweep.py` / `batch.py`

**Produced by:** `sweep.py` (single sweep) or batch-triggered by `batch.py`.

**Location:** `analysis/outputs/{seed_sweep|alpha_curve}/{type}/{embedding}/{arch}/{dataset}_{DDMM}/evaluation_data.csv`

**One row per run** in the seed sweep.

### Identity columns

| Column | Type | Description |
|--------|------|-------------|
| `run_name` | str | Numbered sub-directory name (e.g. `"3"`) |
| `run_path` | str | Absolute path to the run directory |
| `config/{key}` | varies | Hydra config values extracted during analysis. The column name is `config/` followed by the full dotted Hydra key (e.g. `config/tracking.seed`, `config/dataset.name`, `config/trainer.nll.alpha`). Which keys are present depends on `CONFIG_KEYS` in `sweep.py`. |

> **Warm vs cold start.** `config/descriptor` is the discriminator, and it is authoritative —
> it records what the run actually did:
>
> | `config/descriptor` | initialisation |
> |---|---|
> | `nll_pretrained` | **warm** — NAT α>0 fine-tuned from the α=0 checkpoint |
> | `nll_cold` | **cold** — NAT α>0 trained from scratch, own HPO |
> | `nll` | **base** — α=0, from scratch; shared by both ladders |
> | `at_pretrained` | AT; a different axis (AT always warm-starts from a NAT checkpoint) |
>
> `config/model_path` is the corroborating evidence (which checkpoint a warm run resumed from),
> **not** a discriminator on its own: every AT config sets it too. Do not infer initialisation
> from the directory name either — new dirs carry a `cold_`/`warm_` prefix, but historical warm
> dirs carry none, so absence of `cold_` does not mean warm. α=0 is deliberately unprefixed.

> **Alpha column.** α lives under the *active* trainer: `config/trainer.nll.alpha` on NAT runs,
> `config/trainer.adversarial.alpha` on AT runs. Both keys are extracted; the inactive one is
> empty. CSVs written before 2026-07-31 instead carried a dead
> `config/trainer.generative.criterion.kwargs.alpha` column, all-NaN because the path has not
> existed since the trainer refactor. **All 65 were migrated to the live keys** on 2026-07-31 by
> `tools/backfill_alpha_column.py` (source of truth: W&B, matched on group + run name + seed);
> no CSV carries the dead column any more. The archived pre-refactor layout under
> `analysis/outputs/seed_sweep/{cls,gen,adv,comb,cls_reg}/…/d10D6/` has no α column at all — its
> regime is encoded in the path.

### Metric columns

Optional groups depend on which metrics were enabled in the `COMPUTE_*` flags at the top of `sweep.py`.

> **Prefix note (post-Phase-7).** The column names in the tables below are shown with the
> historical `eval/` / `eval/test/` prefix. **Current `sweep.py` writes them WITHOUT that
> prefix** — e.g. `acc`, `rob/0.1`, `uq_adv_acc/0.1`, `uq_purify_acc/0.1/0.1`. Strip the
> `eval/test/` / `eval/` prefix when reading recent CSVs (the old prefixed names only appear
> in pre-refactor outputs).

#### Accuracy & loss

| Column | Description |
|--------|-------------|
| `eval/{split}/acc` | Clean classification accuracy on `split` ∈ {`valid`, `test`} |
| `eval/{split}/clsloss` | NLL classification loss |
| `eval/{split}/genloss` | Generative NLL loss (joint p(x,c)) |
| `eval/{split}/fid` | FID-like score (disabled for data_dim ≥ 100) |

#### Robustness

| Column | Description |
|--------|-------------|
| `eval/{split}/rob/{eps_rel}` | Robust accuracy at relative PGD budget `eps_rel`. One column per budget. `split` ∈ {`valid`, `test`}. For the test split, values are reused from `uq_adv_acc` when UQ is enabled (see above). |

#### Uncertainty quantification (UQ)

| Column | Description |
|--------|-------------|
Column names below omit the dropped `eval/` prefix (see prefix note above). `{q}` is a clean
false-positive rate in percent (the threshold `τ` is the `{q}`-th percentile of clean
`log p(x)`); `{eps}` is the relative attack budget; `{radius}` the relative purification radius.

| Column | Description |
|--------|-------------|
| `uq_clean_accuracy` | Clean accuracy (cross-check via UQ pipeline) |
| `uq_clean_log_px_mean` | Mean log p(x) on clean test data |
| `uq_adv_acc/{eps}` | Adversarial accuracy, **no defense**, at `eps`. Equals `rob/{eps}` when both are computed. |
| `uq_detection/{q}pct/{eps}` | **Detection rate**: fraction of adversarial inputs flagged (`log p(x) < τ`) at threshold `τ` = `{q}`-th percentile of clean `log p(x)`. |
| `uq_det_err_detected/{q}pct/{eps}` | Misclassification rate **among detected (flagged)** adversarial inputs. |
| `uq_det_err_passed/{q}pct/{eps}` | Misclassification rate **among passed (non-flagged)** adversarial inputs. ⇒ **accuracy on accepted inputs = `1 − uq_det_err_passed/{q}pct/{eps}`** (conditional on acceptance; pair with `1 − uq_detection/{q}pct/{eps}` for coverage). |
| `uq_purify_acc/{eps}/{radius}` | Accuracy after **likelihood purification** (projected gradient ascent on `log p(x)` within an `radius` ball) of `eps`-attacked inputs. |
| `uq_purify_recovery/{eps}/{radius}` | Recovery rate: fraction of previously-wrong examples corrected by purification. |
| `uq_clean_purify_acc/{radius}` | Accuracy after purifying **clean** inputs (sanity: purification should not hurt clean accuracy). |
| `gibbs_purify_acc/{eps}/{n_sweeps}` | Accuracy after Gibbs-sampling purification. Only when `COMPUTE_GIBBS_PURIFICATION=True`. |
| `gibbs_purify_recovery/{eps}/{n_sweeps}` | Recovery rate after Gibbs purification. |

> The Gibbs columns are keyed by **`n_sweeps`, not a radius** — Gibbs purification is
> attack-radius agnostic. `step_delta_rel` is a *per-sweep* L∞ move (the window re-centres
> every sweep, so the k-sweep envelope is `k × step_delta_rel × range_size`), and strength is
> set by the number of sweeps alone. For a dedicated, more thoroughly reported Gibbs run
> see `gibbs_data.csv` below.

**`uq_joint_*` family** (`uq_joint_adv_acc/{eps}`, `uq_joint_detection/…`,
`uq_joint_det_err_{detected,passed}/…`, `uq_joint_purify_{acc,recovery}/…`) — the **same
metrics measured under the joint / adaptive attack** (`COMPUTE_JOINT_ATTACK=True`): a PGD
attack that degrades classification **while keeping `log p(x)` high to evade the likelihood
detector**. These are *harder-attack* counterparts of the columns above, **not** a separate
"detect + purify" defense. Do not plot `uq_joint_adv_acc` as a defense line.

### Companion file: `evaluation_summary.txt`

Human-readable table with mean ± std across seeds, Pareto-frontier runs, and acc-vs-eps band. Contains no data not derivable from `evaluation_data.csv`.

---

## `gibbs_data.csv` — written by `analysis/gibbs.py`

Lands in the **same** `analysis/outputs/{rel}/` directory as `evaluation_data.csv`; the two
coexist because the filenames differ. One row per run, same `{eps_rel}` relative convention
as above. Gibbs is orders of magnitude more expensive than every other post-hoc metric, which
is why it has its own script and its own file.

| Column | Description |
|--------|-------------|
| `gibbs_clean_acc` | Clean accuracy on the evaluated subsample, no attack, no defense. |
| `gibbs_adv_acc/{eps}` | Accuracy under PGD at `eps`, **no defense**. |
| `gibbs_clean_purify_acc/{k}` | Accuracy after `k` Gibbs sweeps on **clean** inputs (cost of purifying something that did not need it). |
| `gibbs_purify_acc/{eps}/{k}` | Accuracy after `k` Gibbs sweeps on `eps`-attacked inputs. **The headline defense number.** |
| `gibbs_purify_recovery/{eps}/{k}` | Fraction of previously-misclassified adversarial examples corrected by `k` sweeps. |
| `gibbs_clean_log_px_mean`, `gibbs_adv_log_px_mean/{eps}` | Mean `log p(x)` before purification. |
| `gibbs_clean_log_px_mean/{k}`, `gibbs_purify_log_px_mean/{eps}/{k}` | Mean `log p(x)` after `k` sweeps — should rise toward the clean level. |
| `gibbs_n_samples` | **Test samples actually evaluated.** Cost is linear in this; it is often a subsample, so never read a table as full-test-set without checking. |
| `gibbs_n_eval_seed` | Seed for the subsample. Fixed across runs ⇒ every column is paired. |
| `gibbs_step_delta_rel`, `gibbs_num_bins` | Purifier settings used (provenance). |
| `gibbs_runtime_s` | Wall-clock seconds for the run — use it to size larger sweeps. |

**Reading the `k` columns:** `k` is a sweep count, not a radius. Because the restriction
window re-centres each sweep, `k` sweeps reach up to `k × gibbs_step_delta_rel × range_size`
from the input, so the defense is parameterized without reference to the attacker's budget.
Comparing `gibbs_purify_acc/{eps}/{k}` across `k` at fixed `eps` traces the
purification-strength curve; comparing against `gibbs_clean_purify_acc/{k}` shows what that
strength costs on clean data.

### Companion file: `gibbs_summary.txt`

Human-readable accuracy table (rows: no-defense + one per `k`; columns: `eps=0` and each
`eps`), plus across-run std/stderr, recovery rates, and the resolved `N_EVAL` / `step_delta_rel`
/ `num_bins` in the header. Contains no data not derivable from `gibbs_data.csv`.

### Reconstructing aggregates

```python
import pandas as pd
df = pd.read_csv("analysis/outputs/spirals/nat/legendre/d10r6c64/seed_sweep/a0_0208/evaluation_data.csv")

# Mean ± std robust accuracy vs epsilon
rob_cols = sorted([c for c in df.columns if c.startswith("eval/test/rob/")],
                  key=lambda c: float(c.split("/")[-1]))
df[rob_cols].agg(["mean", "std"])

# Best run by test accuracy
best = df.sort_values("eval/test/acc", ascending=False).iloc[0]

# All purification results
df[[c for c in df.columns if "purify_acc" in c]].agg(["mean", "std"])
```

---

## Cross-CSV quick reference

| CSV location | Script | Grouping key | Row = | Diff columns saved? |
|---|---|---|---|---|
| `{seed_sweep\|alpha_curve}/{type}/{emb}/{arch}/{ds}/evaluation_data.csv` | `sweep.py` | `config/tracking.seed` | one run | N/A |
| `…/{same dir}/gibbs_data.csv` | `gibbs.py` | `run_name` | one run | N/A |

