# Experiments Guide

This directory contains the entry point scripts for running experiments.

## Entry Points

| Script | Purpose |
|--------|---------|
| `classification.py` | Classification-only training |
| `ganstyle.py` | Classification pretraining + GAN-style training |
| `adversarial.py` | Classification pretraining + Adversarial training |
| `generative.py` | Classification pretraining + Generative NLL training |

## Running Experiments

### Basic Usage

```bash
# Classification experiment
python -m experiments.classification +experiments=classification/fourier_d30D18/D18

# GAN-style experiment
python -m experiments.ganstyle +experiments=ganstyle/fourier_d30D18/default

# Adversarial training experiment
python -m experiments.adversarial +experiments=adversarial/fourier_d30D18/hpo/moons

# Generative training experiment
python -m experiments.generative +experiments=generative/fourier_d30D18/hpo/hpo

# Quick test run
python -m experiments.classification +experiments=tests/classification
```

### Command-Line Overrides (debugging only)

```bash
# Fewer epochs
python -m experiments.classification +experiments=tests/classification \
    trainer.classification.max_epoch=10

# Disable W&B
python -m experiments.classification +experiments=tests/classification \
    tracking.mode=disabled

# Different learning rate
python -m experiments.classification +experiments=tests/classification \
    trainer.classification.optimizer.kwargs.lr=1e-3
```

### Multirun (Grid Sweep)

```bash
python -m experiments.classification --multirun +experiments=classification/fourier_d30D18/D18_sweep
```

### Hyperparameter Optimization (Optuna)

**Option 1**: Specify sweeper in experiment config (recommended):
```yaml
# configs/experiments/classification/fourier_d30D18/hpo/lrwd_hpo.yaml
defaults:
  - override /hydra/sweeper: optuna
```
```bash
python -m experiments.classification --multirun +experiments=classification/fourier_d30D18/hpo/lrwd_hpo
```

**Option 2**: Specify sweeper on command line:
```bash
python -m experiments.classification --multirun \
    hydra/sweeper=optuna \
    +experiments=classification/fourier_d30D18/D18
```

**Override HPO settings from command line**:
```bash
python -m experiments.classification --multirun \
    +experiments=classification/fourier_d30D18/hpo/lrwd_hpo \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.n_jobs=4
```

## Output Directory Structure

Each run creates an output directory:

```
outputs/
└── {experiment}_{regime}_{archinfo}_{dataset}_{DDMM_HHMM}/
    ├── .hydra/
    │   ├── config.yaml       # Resolved config (all values)
    │   ├── hydra.yaml        # Hydra settings
    │   └── overrides.yaml    # Command line overrides used
    ├── models/               # Saved model checkpoints (if save=True)
    │   └── {dataset}_{model_info}.pt
    └── wandb/                # W&B local files
        └── ...
```

For multirun/sweep:
```
outputs/
└── {experiment}_{regime}_{archinfo}_{dataset}_{DDMM}/
    ├── 0/                    # First trial (job_num=0)
    │   ├── .hydra/
    │   └── ...
    ├── 1/                    # Second trial (job_num=1)
    │   └── ...
    └── multirun.yaml         # Sweep configuration
```

**Naming components:**
- `{experiment}`: purpose-only label (e.g. `hpo`, `best`, `seed_sweep`)
- `{regime}`: concatenation of active trainer codes (`cls`, `gen`, `adv`, `gan`)
- `{archinfo}`: `d{in_dim}D{bond_dim}{embedding}` (e.g. `d30D18fourier`)
- `{dataset}`: dataset name (e.g. `moons_4k`)
- `{date}`: `DDMM` for multiruns, `DDMM_HHMM` for single runs

## Useful Commands

**Debug config** (print resolved config without running):
```bash
python -m experiments.classification --cfg job +experiments=tests/classification
```

**List available options**:
```bash
python -m experiments.classification --help
```

**Dry run** (resolve config only):
```bash
python -m experiments.classification --info all +experiments=tests/classification
```

## HPO Objective Values

Each entry point returns an objective value for Optuna optimization:

| Script | Objective | Direction |
|--------|-----------|-----------|
| `classification.py` | `trainer.best[stop_crit]` | minimize (negated for acc/rob) |
| `adversarial.py` | `adv_trainer.best[stop_crit]` | minimize (negated for acc/rob) |
| `ganstyle.py` | `gan_trainer.best[stop_crit]` | minimize (negated for acc/rob) |
| `generative.py` | `gen_trainer.best[stop_crit]` | minimize (negated for acc/rob) |

**Notes**:
- **All entry points**: Use the metric specified by `stop_crit` in the respective trainer config. Valid values: `"clsloss"`, `"genloss"`, `"acc"`, `"fid"`, `"rob"`.
- **Classification training**: Use `stop_crit: "clsloss"` or `"acc"`.
- **Generative training**: Use `stop_crit: "genloss"` as the NLL loss is more reliable than GAN-style metrics.
- **Do NOT use `stop_crit: "viz"`** - visualization is not a numeric metric.
- **Robustness**: If `stop_crit: "rob"`, the objective is the average accuracy across all perturbation strengths.
- **Keep `direction: minimize`** in Optuna config - the entry points handle negation for acc/rob internally.
