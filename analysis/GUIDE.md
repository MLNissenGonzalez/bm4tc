# Analysis Module Guide

Post-experiment analysis for Born Machine seed sweeps. The two scripts you will use day-to-day are:

- **`queue_analysis.py`** — batch-run analysis for all unanalyzed sweeps (start here)
- **`sweep_analysis.py`** — analyze one sweep interactively or as a script

For the math behind what is being computed (attacks, purification, MIA, UQ), see [`analysis/utils/GUIDE.md`](utils/GUIDE.md).

---

## Quick Start

```bash
# 1. Analyze every completed-but-unanalyzed seed sweep
python analysis/queue_analysis.py

# 2. Check what would run without doing it
python analysis/queue_analysis.py --dry-run

# 3. Analyze a specific sweep directly
python analysis/sweep_analysis.py outputs/seed_sweep/gen/hermite/d4D3/moons_4k_2102
```

---

## `queue_analysis.py` — Batch Runner

Discovers every directory under `outputs/seed_sweep/` that contains `multirun.yaml`, checks whether `analysis/outputs/<sweep>/evaluation_data.csv` already exists, and runs `sweep_analysis.py` for each unanalyzed sweep.

**Exits non-zero if any analysis fails** — safe to put in a pipeline.

### Usage

```bash
# All unanalyzed sweeps
python analysis/queue_analysis.py

# Dry run — print commands only
python analysis/queue_analysis.py --dry-run

# Show status of every sweep (OK = already analyzed)
python analysis/queue_analysis.py --list

# Filter by training type (gen | cls | adv)
python analysis/queue_analysis.py --filter-type gen

# Filter by embedding
python analysis/queue_analysis.py --filter-embedding hermite
python analysis/queue_analysis.py --filter-embedding fourier

# Filter by architecture
python analysis/queue_analysis.py --filter-arch d4D3

# Filter by dataset (substring match on the base name, strips date)
python analysis/queue_analysis.py --filter-dataset moons_4k
python analysis/queue_analysis.py --filter-dataset circles   # matches circles_4k, circles_hard, ...

# Combine filters (AND)
python analysis/queue_analysis.py --filter-embedding hermite --filter-type gen

# Force re-run even if evaluation_data.csv already exists
python analysis/queue_analysis.py --force
python analysis/queue_analysis.py --force --filter-embedding hermite
```

### How it finds sweeps

1. Recursively scans `outputs/seed_sweep/` for `multirun.yaml` files.
2. The expected directory structure is:
   ```
   outputs/seed_sweep/{type}/{embedding}/{arch}/{dataset}_{DDMM}/
   ```
3. Already-analyzed = `analysis/outputs/seed_sweep/{type}/{embedding}/{arch}/{dataset}_{DDMM}/evaluation_data.csv` exists.

The filter flags match against path components — `--filter-type gen` checks the first component after `seed_sweep/`, `--filter-embedding hermite` checks the second, etc.

---

## `sweep_analysis.py` — Single Sweep Analysis

Loads every model checkpoint in a sweep directory, recomputes metrics post-hoc, and saves results. All configuration is in the **CONFIGURATION section at the top of the file** (lines 46–171).

### Running it

```bash
# From project root — positional argument overrides the hardcoded SWEEP_DIR
python analysis/sweep_analysis.py outputs/seed_sweep/gen/fourier/d4D3/moons_4k_2102

# Interactive in VS Code: open the file, run cells with Ctrl+Enter
# The SWEEP_DIR at the top is used when no CLI argument is given
```

### Configuring the analysis

Open `sweep_analysis.py` and edit the configuration block. The most important options:

#### Attack strengths — range-relative convention

The attack epsilon is expressed as a **fraction of the embedding's input range**, then multiplied by `_RANGE_SIZE` (auto-detected from the sweep path):

```python
_STRENGTH_FRACTIONS = [0.05, 0.10, 0.2, 0.5, 0.8]
# _RANGE_SIZE is auto-detected:
#   fourier   → 1.0   (range  0 to 1)
#   legendre  → 2.0   (range -1 to 1)
#   hermite   → 8.0   (range -4 to 4)
#   chebychev1→ 1.98  (range -0.99 to 0.99)
#   chebychev2→ 2.0   (range -1 to 1)
EVASION_CONFIG = {
    "method": "PGD",
    "norm": 2,           # L2 norm
    "num_steps": 20,
    "strengths": [s * _RANGE_SIZE for s in _STRENGTH_FRACTIONS],
}
```

So for Hermite, `0.10 * 8.0 = 0.8` is the epsilon corresponding to "10% of input range" — the same relative perturbation as `0.10 * 1.0 = 0.1` for Fourier.

#### Metric toggles

```python
COMPUTE_ACC      = True   # Clean accuracy
COMPUTE_ROB      = True   # Robustness under attack
COMPUTE_MIA      = True   # Membership inference attack
COMPUTE_CLS_LOSS = False  # NLL classification loss
COMPUTE_GEN_LOSS = False  # NLL generative loss (needs sync_tensors)
COMPUTE_FID      = False  # FID-like score (disabled >100 dims)
COMPUTE_UQ       = True   # Likelihood-based detection + purification
```

Turn off `COMPUTE_MIA` and `COMPUTE_UQ` for fast robustness-only runs.

#### UQ and MIA settings

```python
# Purification radius = 10% of input range
UQ_CONFIG = {
    "radii": [0.10 * _RANGE_SIZE],
    "percentiles": [1, 5, 10, 20],  # threshold calibration percentiles
}

# Adversarial MIA: epsilon = 10% of range; set None to skip
MIA_ADV_STRENGTH = 0.10 * _RANGE_SIZE
```

#### Evaluation splits

```python
EVAL_SPLITS = ["valid", "test"]
# Always evaluate both; model selection uses valid only
```

### Output files

All outputs go to `analysis/outputs/<sweep_path>/`:

| File | Description |
|------|-------------|
| `evaluation_data.csv` | **One row per run**, all metrics. This is the primary output. |
| `sweep_analysis_summary.txt` | Human-readable summary: statistics table, Pareto runs, acc-vs-eps band, correlations |
| `best_run_samples.png` | Generated samples from the best model |
| `best_run_distributions.png` | Decision boundary + p(x,c) heatmaps for best model |

`evaluation_data.csv` column groups:

| Prefix | Example | What it is |
|--------|---------|------------|
| `run_name`, `run_path` | `3`, `outputs/.../3` | Run identity |
| `config/` | `config/seed` | Extracted Hydra config values |
| `eval/<split>/acc` | `eval/test/acc` | Clean accuracy |
| `eval/<split>/rob/<eps>` | `eval/test/rob/0.8` | Robust accuracy at epsilon |
| `eval/<split>/clsloss` | `eval/valid/clsloss` | NLL classification loss |
| `eval/mia_accuracy` | — | LR-based MIA attack accuracy |
| `eval/mia_auc_roc` | — | MIA AUC-ROC |
| `eval/mia_wc_best` | — | Best worst-case threshold MIA accuracy (clean) |
| `eval/adv_mia_wc_best` | — | Best worst-case threshold MIA accuracy (adversarial) |
| `eval/uq_clean_accuracy` | — | UQ clean accuracy (cross-check) |
| `eval/uq_adv_acc/<eps>` | — | Adversarial accuracy before any defense |
| `eval/uq_detection/<pct>pct/<eps>` | — | Detection rate at threshold/epsilon pair |
| `eval/uq_purify_acc/<eps>/<r>` | — | Accuracy after likelihood purification |
| `eval/uq_purify_recovery/<eps>/<r>` | — | Recovery rate (misclassified → correct) |

**Re-derive anything from the CSV** — all summary stats, Pareto frontiers, and acc-vs-strength curves can be recomputed without re-running analysis:

```python
import pandas as pd
df = pd.read_csv("analysis/outputs/seed_sweep/gen/fourier/d4D3/moons_4k_2102/evaluation_data.csv")

# Mean ± std robust accuracy vs epsilon
rob_cols = sorted([c for c in df.columns if c.startswith("eval/test/rob/")],
                  key=lambda c: float(c.split("/")[-1]))
df[rob_cols].agg(["mean", "std"])

# Best run by test accuracy
df.sort_values("eval/test/acc", ascending=False).iloc[0]

# All purification results
df[[c for c in df.columns if "purify_acc" in c]].agg(["mean", "std"])
```

---

## Sanity check against W&B

`sweep_analysis.py` includes a sanity-check section that compares post-hoc metrics against the W&B summary values logged during training. If a mismatch appears, it usually means the data split changed (different `split_seed`) or the evaluation config differs from training.

Configure which metrics to compare in `SANITY_CHECK_METRICS`:
```python
SANITY_CHECK_METRICS = {
    "eval/test/acc":    "summary/adv/test/acc",
    "eval/valid/clsloss": "summary/adv/valid/clsloss",
}
```

---

## Other analysis scripts

| Script | When to use |
|--------|-------------|
| `hpo_analysis.py` | Explore HPO results: parameter-metric correlations, surface plots. Reads from W&B or local. |
| `mia_analysis.py` | Deep MIA analysis for a single run with histograms and feature importance plots. |
| `uq_analysis.py` | Deep UQ analysis for a single run with detection/purification heatmaps. |
| `visualize_distributions.py` | 2D decision boundary + p(x,c) heatmaps for a single run. |

For most purposes `sweep_analysis.py` / `queue_analysis.py` cover everything above — the standalone scripts are for deeper dives into a single run.

---

## Data flow

```
outputs/seed_sweep/{type}/{emb}/{arch}/{dataset}_{date}/
  ├── 0/.hydra/config.yaml      ← Hydra config
  ├── 0/models/model.pt         ← checkpoint
  ├── 1/.hydra/config.yaml
  ├── 1/models/model.pt
  └── ...

          ↓  evaluate_sweep()  (analysis/utils/evaluate.py)

  For each run:
    1. load config (.hydra/config.yaml)
    2. load BornMachine (models/model.pt)
    3. rebuild DataHandler → split_and_rescale(bm)
    4. compute: acc, rob, MIA, UQ
    5. return flat dict of metrics

          ↓

analysis/outputs/seed_sweep/{type}/{emb}/{arch}/{dataset}_{date}/
  ├── evaluation_data.csv
  └── sweep_analysis_summary.txt
```

**Key invariant**: `DataHandler.split_and_rescale(bm)` uses `bm.input_range`, which is reconstructed from `cfg.embedding` when the model is loaded. Data is always rescaled to the correct embedding domain.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `No valid run directories found` | Sweep path doesn't contain `.hydra/config.yaml` in numbered subdirs | Check path; test sweeps have `.hydra/` directly in root (single run, not multirun) |
| `Metric 'rob' failed` | Attack error, usually NaN gradients | Check if model trained correctly; try smaller epsilon |
| `Metric 'genloss' failed` | Generator not synced | Set `compute_gen_loss=False` or add `bm.sync_tensors(after="classification")` |
| All `uq_purify_acc` ≈ `uq_adv_acc` | Purification radius too small relative to attack epsilon | Increase `radii` in `UQ_CONFIG` |
| `evaluation_data.csv` missing a sweep | `queue_analysis.py` failed silently | Run `--list` to check, then run specific sweep manually to see traceback |
| Hermite robust accuracy suspiciously high | Old analysis run before the range-size bug fix | Re-run with `--force`; now uses correct `_RANGE_SIZE = 8.0` |
