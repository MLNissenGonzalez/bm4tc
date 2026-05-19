# Experiments Guide

Entry point scripts for running experiments. All scripts are run as Python modules from the project root.

## Entry Points

| Script | Purpose |
|--------|---------|
| `discriminative.py` | Discriminative NLL training |
| `generative.py` | Classification pretraining + generative NLL training |
| `adversarial.py` | Classification pretraining + adversarial training (PGD-AT or TRADES) |
| `softmax_sanity.py` | Softmax interpretation sanity check (raw amplitudes as softmax logits) |
| `queue_experiments.py` | Batch-run/list HPO and seed_sweep configs (skip already-run) |

## Running Experiments

### Basic Usage

```bash
# Discriminative experiment
python -m experiments.discriminative +experiments=discriminative/fourier/d4r3/hpo/moons

# Adversarial training experiment
python -m experiments.adversarial +experiments=adversarial/fourier/d4r3/hpo/moons

# Generative training experiment
python -m experiments.generative +experiments=generative/fourier/d4r3/hpo/moons

# Quick test run (tiny config, no W&B)
python -m experiments.discriminative +experiments=tests/discriminative tracking.mode=disabled

# Softmax sanity check
python -m experiments.softmax_sanity +experiments=tests/softmax/legendre_mnist
```

### Command-Line Overrides (debugging only)

```bash
# Fewer epochs
python -m experiments.discriminative +experiments=tests/discriminative \
    trainer.discriminative.max_epoch=10

# Disable W&B
python -m experiments.discriminative +experiments=tests/discriminative \
    tracking.mode=disabled

# Different learning rate
python -m experiments.discriminative +experiments=tests/discriminative \
    trainer.discriminative.optimizer.kwargs.lr=1e-3
```

### Multirun (Grid Sweep)

```bash
python -m experiments.discriminative --multirun +experiments=discriminative/fourier/d4r3/sweep
```

### Hyperparameter Optimization (Optuna)

**Option 1**: Specify sweeper in experiment config (recommended):
```yaml
# configs/experiments/discriminative/fourier/d4r3/hpo/lrwd_hpo.yaml
defaults:
  - override /hydra/sweeper: optuna
```
```bash
python -m experiments.discriminative --multirun +experiments=discriminative/fourier/d4r3/hpo/lrwd_hpo
```

**Option 2**: Specify sweeper on command line:
```bash
python -m experiments.discriminative --multirun \
    hydra/sweeper=optuna \
    +experiments=discriminative/fourier/d4r3/hpo/lrwd_hpo
```

## Output Directory Structure

Each single run creates:
```
outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{DDMM_HHMM}/
    .hydra/
    │   ├── config.yaml       # Resolved config (all values)
    │   ├── hydra.yaml        # Hydra settings
    │   └── overrides.yaml    # Overrides used
    ├── models/               # Saved model checkpoint (if save=True)
    │   └── model.pt
    └── log.json              # Epoch-level metrics (always written)
```

For multirun/sweep:
```
outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{DDMM}/
    ├── 0/                    # First trial
    │   ├── .hydra/
    │   ├── log.json
    │   └── models/model.pt
    ├── 1/                    # Second trial
    └── multirun.yaml
```

**Naming components:**
- `{kind}`: `hpo` | `seed_sweep` | `alpha_curve` | `test`
- `{regime}`: `dis` | `gen` | `adv`
- `{arch}`: `d{in_dim}r{bond_dim}` — e.g. `d4r3`, `d30r18`
- `{dataset}`: dataset name — e.g. `moons`, `circles`
- `{date}`: `DDMM` for multiruns, `DDMM_HHMM` for single runs

## Useful Commands

**Debug config** (print resolved config without running):
```bash
python -m experiments.discriminative --cfg job +experiments=tests/discriminative
```

**List available options**:
```bash
python -m experiments.discriminative --help
```

## Batch-Running Experiments (`queue_experiments.py`)

Discovers all `hpo/` and `seed_sweep/` configs under `configs/experiments/` and runs them
sequentially, skipping any that already have a matching output directory.

### Usage

```bash
# List all discovered configs with [ran]/[   ] status
python -m experiments.queue_experiments --list

# Dry-run: print commands without executing
python -m experiments.queue_experiments --dry-run

# Run everything that hasn't been run yet
python -m experiments.queue_experiments

# Filter by training type (dis | adv | gen)
python -m experiments.queue_experiments --filter-type gen --dry-run

# Filter by embedding
python -m experiments.queue_experiments --filter-embedding legendre --dry-run

# Filter by architecture (exact match)
python -m experiments.queue_experiments --filter-arch d10r6 --dry-run

# Filter by kind (hpo | seed_sweep)
python -m experiments.queue_experiments --filter-kind hpo --dry-run

# Filter by dataset (substring match)
python -m experiments.queue_experiments --filter-dataset circles --dry-run
```

## HPO Objective Values

Each entry point returns an objective value for Optuna:

| Script | Objective | Direction |
|--------|-----------|-----------|
| `discriminative.py` | `trainer.best[stop_crit]` | minimize (negated for acc/rob) |
| `adversarial.py` | `adv_trainer.best[stop_crit]` | minimize (negated for acc/rob) |
| `generative.py` | `gen_trainer.best[stop_crit]` | minimize (negated for acc/rob) |

Valid `stop_crit` values: `"dis_loss"`, `"gen_loss"`, `"acc"`, `"rob"`.  
Keep `direction: minimize` in Optuna config — entry points handle negation for acc/rob internally.
