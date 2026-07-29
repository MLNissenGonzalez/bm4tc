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
> have neither provenance column. `tools/migrate_metric_keys.py` converts them in place.
> `baselines/jem/` is deliberately **not** migrated — JEM has no embedding, so its
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
| `config/{key}` | varies | Hydra config values extracted during analysis. The column name is `config/` followed by the full dotted Hydra key (e.g. `config/tracking.seed`, `config/dataset.name`, `config/trainer.generative.criterion.kwargs.alpha`). Which keys are present depends on `CONFIG_KEYS` in `sweep.py`. |

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

#### Membership inference (MIA)

| Column | Description |
|--------|-------------|
| `eval/mia_accuracy` | LR-classifier MIA attack accuracy |
| `eval/mia_auc_roc` | MIA AUC-ROC |
| `eval/mia_wc_best` | Best worst-case threshold MIA accuracy (clean features) |
| `eval/mia_wc/{feat_name}` | Per-feature worst-case threshold accuracy (clean) |
| `eval/adv_mia_wc_best` | Best worst-case threshold MIA accuracy (adversarial features) |
| `eval/adv_mia_wc/{feat_name}` | Per-feature worst-case threshold accuracy (adversarial) |
| `eval/mia_train_correct_probs` | Serialized list of correct-class probabilities for *train* samples |
| `eval/mia_test_correct_probs` | Serialized list of correct-class probabilities for *test* samples |

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
df = pd.read_csv("analysis/outputs/seed_sweep/gen/legendre/d10D6/moons_4k_1203/evaluation_data.csv")

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

## Type 2: `cls_reg_analysis.py`

**Produced by:** `cls_reg_analysis.py`

**Location:** `analysis/outputs/seed_sweep/cls_reg/{regime}/{embedding}/{arch}/{dataset}_{DDMM}/evaluation_data.csv`

**One row per (run, max_epoch) combination**, plus one synthetic row per seed at `max_epoch = 0` representing the *pretrained baseline* (the model before post-training started).

### Identity / grouping columns

| Column | Type | Description |
|--------|------|-------------|
| `max_epoch` | int | Post-training epoch count. **0 = pretrained baseline** (shared across seeds). |
| `seed` | int | Random seed of the run |
| `run_path` | str | Path to run directory (or pretrained checkpoint path for `max_epoch=0` rows) |
| `range_size` | float | Embedding input range size (see conventions above) |
| `run_name` | str | Numbered sub-directory name |
| `config/tracking.seed` | str | Seed from Hydra config (may duplicate `seed`) |

### Metric columns

Same schema as Type 1 (`eval/{split}/acc`, `eval/{split}/rob/{eps}`, UQ columns, etc.), but only the test split (`EVAL_SPLIT = "test"`) is evaluated. The pretrained baseline (`max_epoch=0`) rows have acc and rob metrics but **not** gen_loss or UQ (the pretrained classifier has no synced generator).

### What is NOT stored: `diff/` columns

Diff columns (`diff/test/acc`, `diff/test/rob/{eps}`, `diff/uq_purify_acc/{eps}/{radius}`, etc.) are computed **on every run of the script** as per-seed differences:

```
diff/{metric}[seed, max_epoch] = eval/{metric}[seed, max_epoch] − eval/{metric}[seed, max_epoch=0]
```

They are NOT written back to `evaluation_data.csv`. They appear in `summary.csv` (aggregated). To reconstruct them:

```python
import pandas as pd, numpy as np

df = pd.read_csv("evaluation_data.csv")
metric_cols = [c for c in df.columns if c.startswith("eval/")]

pre = df[df["max_epoch"] == 0].set_index("seed")
for idx, row in df[df["max_epoch"] != 0].iterrows():
    seed = row["seed"]
    if seed not in pre.index:
        continue
    pre_row = pre.loc[seed]
    if isinstance(pre_row, pd.DataFrame):
        pre_row = pre_row.iloc[0]
    for col in metric_cols:
        diff_col = "diff/" + col[len("eval/"):]
        try:
            df.at[idx, diff_col] = float(row[col]) - float(pre_row[col])
        except (TypeError, ValueError, KeyError):
            df.at[idx, diff_col] = np.nan
```

### Companion files

| File | Description |
|------|-------------|
| `summary.csv` | Mean ± std per `max_epoch` for all `eval/` and `diff/` columns. Sufficient to reproduce all plots without re-running the script. |
| `acc.png`, `rob.png`, `purif.png`, `loss.png` | Mean ± std with overlaid per-seed thin lines (use `df` not just `summary.csv` to reproduce the seed lines) |
| `diff.png` | Δ metrics (posttraining − pretrained) mean ± std |
| `evolution.png` | Combined evolution figure from `visualize/cls_reg_evolution.py` |

### Reconstructing summary and plots from `evaluation_data.csv`

```python
import pandas as pd, numpy as np

df = pd.read_csv("evaluation_data.csv")

# (Re)compute diff columns as above, then:
metric_cols = [c for c in df.columns if c not in ("max_epoch", "seed", "run_path", "range_size", "run_name")]
summary_rows = []
for ep in sorted(df["max_epoch"].dropna().unique()):
    g = df[df["max_epoch"] == ep]
    row = {"max_epoch": int(ep), "n_seeds": len(g)}
    for col in metric_cols:
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        row[f"{col}/mean"] = vals.mean() if len(vals) > 0 else np.nan
        row[f"{col}/std"]  = vals.std()  if len(vals) > 1 else np.nan
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
# summary_df now matches summary.csv exactly (modulo floating-point rounding)
```

---

## Type 3: `dev_comb_analysis.py`

**Produced by:** `dev_comb_analysis.py`

**Location:** `analysis/outputs/seed_sweep/comb/{embedding}/{arch}/{dataset}_{DDMM}/evaluation_data.csv`

**One row per run**. Each run contributes two model evaluations — the `models/cls` checkpoint (classifier only) and the `models/gen` checkpoint (after generative post-training).

### Identity / grouping columns

| Column | Type | Description |
|--------|------|-------------|
| `cls_epoch` | int | Number of classification pre-training epochs for this run |
| `seed` | int | Random seed |
| `run_path` | str | Absolute path to the run directory |
| `range_size` | float | Embedding input range size |

### Metric columns

Each model's metrics are stored under a `cls/` or `gen/` prefix, followed by the standard `eval/...` key:

| Column pattern | Description |
|----------------|-------------|
| `cls/eval/{split}/{metric}` | Metric for the `models/cls` checkpoint |
| `gen/eval/{split}/{metric}` | Metric for the `models/gen` checkpoint |

The inner `eval/...` suffix follows exactly the same schema as Type 1 (acc, clsloss, genloss, rob/{eps}, UQ columns). Only the test split (`EVAL_SPLIT = "test"`) is evaluated.

**Example columns:**
```
cls/eval/test/acc
cls/eval/test/clsloss
cls/eval/test/genloss
cls/eval/test/rob/0.2          # eps = 0.1 × range_size (legendre)
cls/eval/uq_clean_accuracy
cls/eval/uq_adv_acc/0.2
cls/eval/uq_purify_acc/0.2/0.2
gen/eval/test/acc
gen/eval/test/clsloss
...
```

### What is NOT stored: `diff/` columns

Diff columns are computed on every script run as `gen/{metric} − cls/{metric}` for all metrics present in both prefixes:

```python
import pandas as pd

df = pd.read_csv("evaluation_data.csv")

cls_keys = {c[len("cls/"):] for c in df.columns if c.startswith("cls/")}
gen_keys  = {c[len("gen/"):] for c in df.columns if c.startswith("gen/")}
for k in cls_keys & gen_keys:
    df[f"diff/{k}"] = (
        pd.to_numeric(df[f"gen/{k}"], errors="coerce")
        - pd.to_numeric(df[f"cls/{k}"], errors="coerce")
    )
```

The diff columns appear in `summary.csv` (aggregated as mean ± std per `cls_epoch`) but not in `evaluation_data.csv`.

### Companion files

| File | Description |
|------|-------------|
| `summary.csv` | Mean ± std per `cls_epoch` for all `cls/`, `gen/`, and `diff/` columns. Sufficient to reproduce all plots. |
| `acc.png` | Clean accuracy of `gen` model vs `cls_epoch` |
| `rob.png` | Robust accuracy (w/ and w/o purification) of `gen` model vs `cls_epoch` |
| `purif.png` | Purification accuracy and recovery rate of `gen` model |
| `loss.png` | cls_loss and gen_loss of `gen` model |
| `diff.png` | Δ (gen − cls) for acc, losses, rob, purification |

### Reconstructing summary and plots

```python
import pandas as pd, numpy as np

df = pd.read_csv("evaluation_data.csv")

# 1. Recompute diff columns (see above)
cls_keys = {c[len("cls/"):] for c in df.columns if c.startswith("cls/")}
gen_keys  = {c[len("gen/"):] for c in df.columns if c.startswith("gen/")}
for k in cls_keys & gen_keys:
    df[f"diff/{k}"] = pd.to_numeric(df[f"gen/{k}"], errors="coerce") \
                    - pd.to_numeric(df[f"cls/{k}"], errors="coerce")

# 2. Aggregate per cls_epoch
metric_cols = [c for c in df.columns if c not in ("cls_epoch", "seed", "run_path", "range_size")]
summary_rows = []
for ep in sorted(df["cls_epoch"].dropna().unique()):
    g = df[df["cls_epoch"] == ep]
    row = {"cls_epoch": int(ep), "n_seeds": len(g)}
    for col in metric_cols:
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        row[f"{col}/mean"] = vals.mean() if len(vals) > 0 else np.nan
        row[f"{col}/std"]  = vals.std()  if len(vals) > 1 else np.nan
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
```

---

## Cross-CSV quick reference

| CSV location | Script | Grouping key | Row = | Diff columns saved? |
|---|---|---|---|---|
| `{seed_sweep\|alpha_curve}/{type}/{emb}/{arch}/{ds}/evaluation_data.csv` | `sweep.py` | `config/tracking.seed` | one run | N/A |
| `…/{same dir}/gibbs_data.csv` | `gibbs.py` | `run_name` | one run | N/A |
| `seed_sweep/cls_reg/{regime}/{emb}/{arch}/{ds}/evaluation_data.csv` | `cls_reg_analysis.py` | `max_epoch` + `seed` | one (run, epoch) | No — recomputed from `max_epoch=0` rows |
| `seed_sweep/comb/{emb}/{arch}/{ds}/evaluation_data.csv` | `dev_comb_analysis.py` | `cls_epoch` + `seed` | one run | No — recomputed as `gen/` − `cls/` |

For all three types: `summary.csv` (where it exists) **does** contain aggregated diff columns and can be used directly to reproduce plots without recomputing anything.
