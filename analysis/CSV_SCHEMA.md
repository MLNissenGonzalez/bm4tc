# Analysis CSV Schema Reference

Every analysis script writes one or more CSV files to `analysis/outputs/`. This document describes the schema of each, what is *not* stored but can be reconstructed, and the conventions needed to interpret the numeric column names.

---

## Common conventions

### Epsilon / radius column names

Attack strengths and purification radii appear as **absolute values** in column names (e.g. `eval/test/rob/0.2`). These are computed from a fraction × `range_size`, where `range_size` depends on the embedding:

| Embedding | Input range | `range_size` |
|-----------|-------------|-------------|
| `fourier` | (0, 1) | 1.0 |
| `legendre` | (−1, 1) | 2.0 |
| `hermite` | (−4, 4) | 8.0 |
| `chebychev1` | (−0.99, 0.99) | ~1.98 |
| `chebychev2` | (−1, 1) | 2.0 |

The scripts use `STRENGTH_FRACTIONS = [0.1, 0.2, 0.5]` and `PURIF_RADIUS_FRAC = 0.1`, so for legendre (`range_size = 2.0`) the epsilon columns are `0.2`, `0.4`, `1.0` and the purification radius column suffix is `0.2`. The `range_size` value is stored in every `evaluation_data.csv` so you can always reconstruct which fractions were intended:

```python
fracs = [round(eps / df["range_size"].iloc[0], 6) for eps in [0.2, 0.4, 1.0]]
# → [0.1, 0.2, 0.5]
```

### `eval/uq_adv_acc/{eps}` vs `eval/test/rob/{eps}`

When UQ is enabled, the test-split rob columns are *copied from* the UQ adversarial accuracy cache rather than re-running PGD. For any epsilon in both the UQ attack strengths and the evasion config strengths, `eval/test/rob/{eps}` == `eval/uq_adv_acc/{eps}`. They are stored as separate columns for clarity.

---

## Type 1: `seed_sweep_analysis.py` / `queue_seed_sweep.py`

**Produced by:** `seed_sweep_analysis.py` (single sweep) or batch-triggered by `queue_seed_sweep.py`.

**Location:** `analysis/outputs/{seed_sweep|alpha_curve}/{type}/{embedding}/{arch}/{dataset}_{DDMM}/evaluation_data.csv`

**One row per run** in the seed sweep.

### Identity columns

| Column | Type | Description |
|--------|------|-------------|
| `run_name` | str | Numbered sub-directory name (e.g. `"3"`) |
| `run_path` | str | Absolute path to the run directory |
| `config/{key}` | varies | Hydra config values extracted during analysis. The column name is `config/` followed by the full dotted Hydra key (e.g. `config/tracking.seed`, `config/dataset.name`, `config/trainer.generative.criterion.kwargs.alpha`). Which keys are present depends on `CONFIG_KEYS` in `seed_sweep_analysis.py`. |

### Metric columns

All metric columns are prefixed `eval/`. Optional groups depend on which metrics were enabled in the `COMPUTE_*` flags at the top of `seed_sweep_analysis.py`.

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
| `eval/{split}/rob/{eps}` | Robust accuracy at PGD epsilon `eps` (absolute). One column per strength. `split` ∈ {`valid`, `test`}. For the test split, values are reused from `uq_adv_acc` when UQ is enabled (see above). |

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
| `eval/uq_clean_accuracy` | Clean accuracy (cross-check via UQ pipeline) |
| `eval/uq_clean_log_px_mean` | Mean log p(x) on clean test data |
| `eval/uq_adv_acc/{eps}` | Adversarial accuracy before purification at `eps` |
| `eval/uq_detection/{pct}pct/{eps}` | Detection rate at threshold calibrated to `{pct}`th percentile of clean log p(x), at attack eps `eps` |
| `eval/uq_purify_acc/{eps}/{radius}` | Accuracy after likelihood purification (attack `eps`, purification ball `radius`) |
| `eval/uq_purify_recovery/{eps}/{radius}` | Recovery rate: fraction of previously-wrong examples corrected by purification |
| `eval/gibbs_purify_acc/{eps}/{n_sweeps}` | Accuracy after Gibbs-sampling purification (attack `eps`, `n_sweeps` full feature sweeps). Only present when `COMPUTE_GIBBS_PURIFICATION=True`. |
| `eval/gibbs_purify_recovery/{eps}/{n_sweeps}` | Recovery rate after Gibbs purification: fraction of pre-purification misclassified samples that become correct. |

### Companion file: `evaluation_summary.txt`

Human-readable table with mean ± std across seeds, Pareto-frontier runs, and acc-vs-eps band. Contains no data not derivable from `evaluation_data.csv`.

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
| `{seed_sweep\|alpha_curve}/{type}/{emb}/{arch}/{ds}/evaluation_data.csv` | `seed_sweep_analysis.py` | `config/tracking.seed` | one run | N/A |
| `seed_sweep/cls_reg/{regime}/{emb}/{arch}/{ds}/evaluation_data.csv` | `cls_reg_analysis.py` | `max_epoch` + `seed` | one (run, epoch) | No — recomputed from `max_epoch=0` rows |
| `seed_sweep/comb/{emb}/{arch}/{ds}/evaluation_data.csv` | `dev_comb_analysis.py` | `cls_epoch` + `seed` | one run | No — recomputed as `gen/` − `cls/` |

For all three types: `summary.csv` (where it exists) **does** contain aggregated diff columns and can be used directly to reproduce plots without recomputing anything.
