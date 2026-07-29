# Analysis Guide

Post-experiment analysis for Born Machine seed sweeps.

For the math behind attacks, purification, MIA, and UQ, see [`analysis/utils/GUIDE.md`](utils/GUIDE.md).  
For the full CSV column schema, see [`analysis/CSV_SCHEMA.md`](CSV_SCHEMA.md).

---

## Scripts

| Script | When to use |
|--------|-------------|
| `run.py` | Single-model deep analysis (acc, rob, MIA, UQ) — callable API |
| `sweep.py` | Post-hoc evaluation of a full seed sweep; primary analysis tool |
| `gibbs.py` | Gibbs-purification defense on a seed sweep — split out because it costs orders of magnitude more than everything in `sweep.py` |
| `hpo.py` | Explore HPO results: parameter-metric correlations, surface plots |
| `batch.py` | Batch-queue runner: processes all unanalyzed sweeps via `sweep.py` |

---

## `run.py` — Single-Model Analysis

Loads one saved model from a Hydra run directory and recomputes metrics on the test set.

### CLI

```bash
python analysis/run.py <run_dir> [--no-acc] [--no-dis-loss] [--no-gen-loss]
                                  [--no-rob] [--no-mia] [--no-uq]
                                  [--device DEVICE]
```

### Callable API

```python
from analysis.run import AnalysisConfig, analyze_run

cfg = AnalysisConfig(
    compute_mia=False,
    evasion_override={"method": "PGD", "norm": "inf", "num_steps": 40,
                      "strengths": [0.05, 0.10, 0.15]},
)
results = analyze_run("outputs/seed_sweep/gen/fourier/d4r3/moons_2102/3", cfg)
# results is a flat dict: {"acc": 0.97, "rob/0.05": 0.88, ...}
```

### `AnalysisConfig` options

| Field | Default | Description |
|-------|---------|-------------|
| `compute_acc` | `True` | Clean classification accuracy |
| `compute_dis_loss` | `True` | Discriminative NLL loss |
| `compute_gen_loss` | `True` | Generative (joint) NLL loss |
| `compute_rob` | `True` | Adversarial robustness |
| `compute_mia` | `True` | Membership inference attack |
| `compute_uq` | `True` | Likelihood-based detection + purification |
| `evasion_override` | `None` | Dict of evasion config overrides; strengths are **absolute** (pre-multiplied by range size) |
| `device` | `"cpu"` | Torch device |

---

## `sweep.py` — Seed Sweep Analysis

Loads every model checkpoint in a sweep directory, recomputes metrics post-hoc, and saves results. All configuration is in the **CONFIGURATION section at the top of the file** (lines ~50–170).

### Running it

```bash
# Positional argument overrides the hardcoded SWEEP_DIR at the top of the file
python analysis/sweep.py outputs/seed_sweep/gen/fourier/d4r3/moons_2102

# Skip distribution plots (faster — batch.py always does this)
python analysis/sweep.py outputs/seed_sweep/gen/fourier/d4r3/moons_2102 --no-viz
```

### Configuring the analysis

Open `sweep.py` and edit the configuration block. The most important options:

#### Attack strengths — range-relative convention

The attack epsilon is expressed as a **fraction of the embedding's input range**, then multiplied by `_RANGE_SIZE` (auto-detected from the sweep path):

```python
_STRENGTH_FRACTIONS = [0.05, 0.10, 0.15]
# _RANGE_SIZE is auto-detected:
#   fourier    → 1.0   (range  0 to 1)
#   legendre   → 2.0   (range -1 to 1)
#   hermite    → 8.0   (range -4 to 4)
#   chebychev1 → 1.98  (range -0.99 to 0.99)
#   chebychev2 → 2.0   (range -1 to 1)
EVASION_CONFIG = {
    "method": "PGD",
    "norm": "inf",
    "num_steps": 40,
    "strengths": [s * _RANGE_SIZE for s in _STRENGTH_FRACTIONS],
}
```

#### Metric toggles

```python
COMPUTE_ACC           = True   # Clean accuracy
COMPUTE_ROB           = True   # Robustness under attack
COMPUTE_MIA           = False  # Membership inference attack
COMPUTE_DIS_LOSS      = True   # NLL discriminative loss
COMPUTE_GEN_LOSS      = True   # NLL generative loss
COMPUTE_UQ            = True   # Likelihood-based detection + purification
COMPUTE_DISTRIBUTIONS = False  # Best-run distribution plots (or pass --no-viz)
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
| `eval/<split>/rob/<eps>` | `eval/test/rob/0.15` | Robust accuracy at epsilon |
| `eval/<split>/dis_loss` | `eval/valid/dis_loss` | NLL discriminative loss |
| `eval/<split>/gen_loss` | `eval/valid/gen_loss` | NLL generative loss |
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

## `gibbs.py` — Gibbs Purification Analysis

Separate from `sweep.py` because Gibbs purification dominates the cost of everything else
combined: every candidate value of every feature is a full chain contraction, so cost is
`N × data_dim × num_bins × max(n_sweeps)` per attack strength, per run.

### Running it

```bash
# Defaults: PGD at 0.05/0.10/0.15 of the range, Gibbs for k = 1, 3, 6 sweeps
python analysis/gibbs.py outputs/moons/nat/legendre/d10r6/seed_sweep_a1_2707

# Cheap smoke run
python analysis/gibbs.py <sweep_dir> --n-eval 100 --sweeps 1,3 --limit-runs 1

# Full test split (expensive)
python analysis/gibbs.py <sweep_dir> --n-eval all
```

### `--n-eval` is the cost knob

Cost is **linear** in the number of test samples evaluated, so `--n-eval` is the first thing
to reach for. The same fixed, seeded subsample is reused for the clean row and every epsilon,
and across every run in the sweep — so all comparisons are paired even though the set is a
subsample. Binomial sampling error on a reported accuracy is about `0.5/√N_EVAL`:

| `--n-eval` | error |
|---|---|
| 100 | ~±5% |
| 1000 (default) | ~±1.6% |
| 4000 | ~±0.8% |

The resolved count is printed up front with a projected contraction count, written to every
CSV row as `gibbs_n_samples`, and echoed in the summary header — a subsampled table can never
be silently misread as a full-test-set one.

Other overrides: `--sweeps`, `--step-radius`, `--num-bins`, `--batch-size`, `--strengths`
(fractions), `--device`, `--limit-runs`, `--n-eval-seed`.

### Attack-radius agnostic — read the `k`, not a radius

`step_radius` (default 0.1 of the feature domain) is a **per-sweep** L∞ move, not a global
budget: the restriction window re-centres at the start of every sweep, so after `k` sweeps a
coordinate can have travelled up to `k × step_radius × range_size`. Purification strength is
therefore controlled by `--sweeps` alone, and the defense never has to be told the attacker's
epsilon. Keep `step_radius` small and vary `k`.

### Output files

Written to the same `analysis/outputs/<sweep_path>/` directory as `sweep.py`'s output:

| File | Description |
|------|-------------|
| `gibbs_data.csv` | One row per run. Schema in `CSV_SCHEMA.md`. |
| `gibbs_summary.txt` | Accuracy table (no-defense + one row per `k` × one column per eps), across-run std/stderr, recovery rates, runtime |

---

## Sanity check against W&B

`sweep.py` includes a section comparing post-hoc metrics against W&B summary values logged during training. Configure which metrics to compare in `SANITY_CHECK_METRICS`:

```python
SANITY_CHECK_METRICS = {
    "eval/test/acc":      "summary/adv/test/acc",
    "eval/valid/dis_loss": "summary/adv/valid/dis_loss",
}
```

---

## Data flow

```
outputs/{seed_sweep|alpha_curve}/{type}/{emb}/{arch}/{dataset}_{date}/
  ├── 0/.hydra/config.yaml      ← Hydra config
  ├── 0/models/model.pt         ← checkpoint
  └── ...

        ↓  analyze_run()  (analysis/run.py)

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
| `Metric 'gen_loss' failed` | gen_loss unavailable for this run | Set `COMPUTE_GEN_LOSS = False` |
| All `uq_purify_acc` ≈ `uq_adv_acc` | Purification radius too small | Increase `radii` in `UQ_CONFIG` |
| Hermite robust accuracy suspiciously high | Old analysis before range-size bug fix | Re-run with `--force`; `_RANGE_SIZE = 8.0` is now correct |
