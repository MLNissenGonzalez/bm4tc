# Experiments Guide

This directory contains the entry point scripts for running experiments.

## Entry Points

| Script | Purpose |
|--------|---------|
| `classification.py` | Classification-only training |
| `ganstyle.py` | Classification pretraining + GAN-style training |
| `adversarial.py` | Classification pretraining + Adversarial training |
| `generative.py` | Classification pretraining + Generative NLL training |
| `softmax_sanity.py` | Softmax interpretation sanity check (raw amplitudes as logits) |
| `queue_experiments.py` | Batch-run/list HPO and seed_sweep configs (skip already-run) |

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

# Softmax sanity check (MPS with softmax loss instead of Born rule)
python -m experiments.softmax_sanity +experiments=tests/softmax/legendre_mnist
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

## Batch-Running Experiments (`queue_experiments.py`)

`queue_experiments.py` discovers all `hpo/` and `seed_sweeps/` configs under
`configs/experiments/` and runs them sequentially, skipping any that already
have a matching output directory.

### Usage

```bash
# List all discovered configs with [ran]/[   ] status
python -m experiments.queue_experiments --list

# Dry-run: print commands without executing
python -m experiments.queue_experiments --dry-run

# Run everything that hasn't been run yet
python -m experiments.queue_experiments

# Re-run even if outputs already exist
python -m experiments.queue_experiments --force

# Filter by training type (cls | adv | gen)
python -m experiments.queue_experiments --filter-type gen --dry-run

# Filter by embedding
python -m experiments.queue_experiments --filter-embedding legendre --dry-run

# Filter by architecture (exact match, e.g. d3D10)
python -m experiments.queue_experiments --filter-arch d3D10 --dry-run

# Filter by kind (hpo | seed_sweep)
python -m experiments.queue_experiments --filter-kind hpo --dry-run

# Filter by dataset (substring match)
python -m experiments.queue_experiments --filter-dataset cricket --dry-run

# Combine filters
python -m experiments.queue_experiments \
    --filter-type gen --filter-embedding hermite --filter-kind seed_sweep
```

### How it works

1. **Discovery**: walks `configs/experiments/{classification,adversarial,generative}/{embedding}/{arch}/{hpo,seed_sweeps}/` and collects all `.yaml` files.
2. **Already-run check**: looks for a matching `outputs/{experiment}/*/{embedding}/d{in_dim}D{bond_dim}/{dataset}_*` directory. If found, the config is skipped (unless `--force`).
3. **Execution**: calls `python -m experiments.{type} --multirun +experiments={type}/{embedding}/{arch}/{kind}/{name}` for each remaining config, stopping on the first non-zero exit code.

### Filters

| Flag | Values | Description |
|------|--------|-------------|
| `--filter-type` | `cls`, `adv`, `gen` (or full names) | Training regime |
| `--filter-embedding` | `fourier`, `legendre`, `hermite` | Embedding type |
| `--filter-arch` | e.g. `d3D10`, `d30D18` | Architecture (exact match) |
| `--filter-kind` | `hpo`, `seed_sweep` | Config phase |
| `--filter-dataset` | any substring | Dataset name substring match |

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
