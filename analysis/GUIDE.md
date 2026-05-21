# Analysis Guide

Post-experiment analysis for Born Machine seed sweeps.

For the math behind attacks, purification, MIA, and UQ, see [`analysis/utils/GUIDE.md`](utils/GUIDE.md).  
For the full CSV column schema, see [`analysis/CSV_SCHEMA.md`](CSV_SCHEMA.md).

---

## Scripts

| Script | When to use |
|--------|-------------|
| `seed_sweep_analysis.py` | Evaluate one sweep post-hoc; primary analysis tool |
| `hpo_analysis.py` | Explore HPO results: parameter-metric correlations, surface plots |
| `mia_analysis.py` | Deep MIA analysis for a single run (histograms, feature importance) |
| `uq_analysis.py` | Deep UQ analysis for a single run (detection/purification heatmaps) |
| `plot_ts_datasets.py` | Visualise UCR time-series dataset splits |

---

## `seed_sweep_analysis.py` — Single Sweep Analysis

Loads every model checkpoint in a sweep directory, recomputes metrics post-hoc, and saves results. All configuration is in the **CONFIGURATION section at the top of the file** (lines ~50–170).

### Running it

```bash
# Positional argument overrides the hardcoded SWEEP_DIR at the top of the file
python analysis/seed_sweep_analysis.py outputs/seed_sweep/gen/fourier/d4r3/moons_2102

# Skip distribution plots (faster — queue_seed_sweep.py always does this)
python analysis/seed_sweep_analysis.py outputs/seed_sweep/gen/fourier/d4r3/moons_2102 --no-viz
```

### Configuring the analysis

Open `seed_sweep_analysis.py` and edit the configuration block. The most important options:

#### Attack strengths — range-relative convention

The attack epsilon is expressed as a **fraction of the embedding's input range**, then multiplied by `_RANGE_SIZE` (auto-detected from the sweep path):

```python
_STRENGTH_FRACTIONS = [0.05, 0.10, 0.2, 0.5, 0.8]
# _RANGE_SIZE is auto-detected:
#   fourier    → 1.0   (range  0 to 1)
#   legendre   → 2.0   (range -1 to 1)
#   hermite    → 8.0   (range -4 to 4)
#   chebychev1 → 1.98  (range -0.99 to 0.99)
#   chebychev2 → 2.0   (range -1 to 1)
EVASION_CONFIG = {
    "method": "PGD",
    "norm": 2,
    "num_steps": 20,
    "strengths": [s * _RANGE_SIZE for s in _STRENGTH_FRACTIONS],
}
```

#### Metric toggles

```python
COMPUTE_ACC           = True   # Clean accuracy
COMPUTE_ROB           = True   # Robustness under attack
COMPUTE_MIA           = True   # Membership inference attack
COMPUTE_CLS_LOSS      = False  # NLL classification loss
COMPUTE_GEN_LOSS      = False  # NLL generative loss
COMPUTE_UQ            = True   # Likelihood-based detection + purification
COMPUTE_DISTRIBUTIONS = True   # Best-run distribution plots (or pass --no-viz)
```

Turn off `COMPUTE_MIA` and `COMPUTE_UQ` for fast robustness-only runs.

#### UQ and MIA settings

```python
UQ_CONFIG = {
    "radii": [0.10 * _RANGE_SIZE],
    "percentiles": [1, 5, 10, 20],
}
MIA_ADV_STRENGTH = 0.10 * _RANGE_SIZE   # set None to skip adversarial MIA
```

#### Evaluation splits

```python
EVAL_SPLITS = ["valid", "test"]
```

### Output files

All outputs go to `analysis/outputs/<sweep_path>/`:

| File | Description |
|------|-------------|
| `evaluation_data.csv` | One row per run, all metrics. Primary output. |
| `evaluation_summary.txt` | Human-readable summary: statistics table, Pareto runs, correlations |
| `best_run_samples.png` | Generated samples from best model (skipped with `--no-viz`) |
| `best_class_dist.png` | p(c\|x) conditional heatmap for best model (skipped with `--no-viz`) |
| `best_joint.png` | Marginal p(x) heatmap for best model (skipped with `--no-viz`) |

`evaluation_data.csv` column groups:

| Prefix | Example | What it is |
|--------|---------|------------|
| `run_name`, `run_path` | `3`, `outputs/.../3` | Run identity |
| `config/` | `config/tracking.seed` | Extracted Hydra config values |
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

### Recomputing results from the CSV

```python
import pandas as pd

df = pd.read_csv("analysis/outputs/seed_sweep/gen/fourier/d4r3/moons_2102/evaluation_data.csv")

# Mean ± std robust accuracy vs epsilon
rob_cols = sorted([c for c in df.columns if c.startswith("eval/test/rob/")],
                  key=lambda c: float(c.split("/")[-1]))
df[rob_cols].agg(["mean", "std"])

# Best run by test accuracy
df.sort_values("eval/test/acc", ascending=False).iloc[0]

# Alpha curve: metric vs alpha (mean ± std across seeds)
df = pd.read_csv("analysis/outputs/alpha_curve/gen/legendre/d10r6/circles_1404/evaluation_data.csv")
alpha_col = "config/trainer.generative.criterion.kwargs.alpha"
df.groupby(alpha_col)["eval/test/acc"].agg(["mean", "std"])
```

---

## Sanity check against W&B

`seed_sweep_analysis.py` includes a section comparing post-hoc metrics against W&B summary values logged during training. Configure which metrics to compare in `SANITY_CHECK_METRICS`:

```python
SANITY_CHECK_METRICS = {
    "eval/test/acc":      "summary/adv/test/acc",
    "eval/valid/clsloss": "summary/adv/valid/clsloss",
}
```

---

## Data flow

```
outputs/{seed_sweep|alpha_curve}/{type}/{emb}/{arch}/{dataset}_{date}/
  ├── 0/.hydra/config.yaml      ← Hydra config
  ├── 0/models/model.pt         ← checkpoint
  └── ...

        ↓  evaluate_sweep()  (analysis/utils/evaluate.py)

  For each run:
    1. load config (.hydra/config.yaml)  →  extract CONFIG_KEYS into config/* columns
    2. ConditionalBornMachine.load(models/model.pt)
    3. rebuild DataHandler → split_and_rescale(cbm)
    4. compute: acc, rob, MIA, UQ
    5. return flat dict of metrics

        ↓

analysis/outputs/{seed_sweep|alpha_curve}/{type}/{emb}/{arch}/{dataset}_{date}/
  ├── evaluation_data.csv
  ├── evaluation_summary.txt
  ├── best_class_dist.png   (if COMPUTE_DISTRIBUTIONS)
  └── best_joint.png        (if COMPUTE_DISTRIBUTIONS)
```

**Key invariant**: `DataHandler.split_and_rescale(cbm)` uses `cbm.input_range`, reconstructed from `cfg.embedding` at load time — always correct regardless of which embedding was used.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `No valid run directories found` | `.hydra/config.yaml` missing in numbered subdirs | Check path; test sweeps have `.hydra/` in root (single run) |
| `Metric 'rob' failed` | NaN gradients in attack | Check model trained correctly; try smaller epsilon |
| `Metric 'genloss' failed` | gen_loss unavailable for this run | Set `COMPUTE_GEN_LOSS = False` |
| All `uq_purify_acc` ≈ `uq_adv_acc` | Purification radius too small | Increase `radii` in `UQ_CONFIG` |
| Hermite robust accuracy suspiciously high | Old analysis before range-size bug fix | Re-run with `--force`; `_RANGE_SIZE = 8.0` is now correct |
