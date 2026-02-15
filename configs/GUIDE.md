# Configuration Guide

This directory contains all Hydra configuration files for experiments.

## Primary Workflow: Experiment Configs

**The recommended way to design and run experiments is via `configs/experiments/` files**, not command-line overrides. This ensures:
- Reproducibility
- Clear documentation of experiment settings
- Easy sharing and version control

**Workflow:**
1. Create an experiment config in `configs/experiments/`
2. The config overrides defaults from other config groups
3. Run with `python -m experiments.<script> +experiments=<path>`

**Example:**
```yaml
# configs/experiments/classification/my_experiment.yaml
# @package _global_
experiment: my_experiment_name

defaults:
  - override /born: d30D18
  - override /dataset: 2Dtoy/moons_4k
  - override /trainer/classification: adam500_loss
  - override /tracking: online
  - override /trainer/ganstyle: null      # Disable
  - override /trainer/adversarial: null   # Disable
```

```bash
python -m experiments.classification +experiments=classification/my_experiment
```

## Directory Structure

```
configs/
├── GUIDE.md                    # This file
├── config.yaml                 # Main config with defaults
├── born/                       # BornMachine architecture configs
│   ├── d4D3.yaml              # in_dim=4, bond_dim=3
│   ├── d10D3.yaml             # in_dim=10, bond_dim=3
│   ├── d10D4.yaml             # in_dim=10, bond_dim=4
│   ├── d10D6.yaml             # in_dim=10, bond_dim=6
│   ├── d30D4.yaml             # in_dim=30, bond_dim=4
│   ├── d30D10.yaml            # in_dim=30, bond_dim=10
│   ├── d30D18.yaml            # in_dim=30, bond_dim=18
│   └── test.yaml              # Minimal test config
├── dataset/                    # Dataset configs
│   ├── test.yaml              # Minimal test config
│   └── 2Dtoy/                 # 2D toy datasets (moons, circles, spirals)
│       ├── circles_2k.yaml    # 2k samples, circles
│       ├── circles_4k.yaml    # 4k samples, circles
│       ├── moons_2k.yaml      # 2k samples, moons
│       ├── moons_4k.yaml      # 4k samples, moons
│       ├── spirals_2k.yaml    # 2k samples, spirals
│       └── spirals_4k.yaml    # 4k samples, spirals
│   # Future: mnist/, timeseries/
├── trainer/
│   ├── classification/        # Classifier training configs
│   │   ├── adam500_loss.yaml # Adam, 500 epochs, stop on loss
│   │   ├── adam500_rob.yaml  # Adam, 500 epochs, stop on robustness
│   │   ├── test.yaml         # Minimal test
│   │   └── archive/          # Old/deprecated configs
│   ├── ganstyle/             # GAN training configs
│   │   ├── default.yaml      # Default GAN config
│   │   ├── test.yaml         # Test config
│   │   ├── power.yaml        # Power training config
│   │   └── critic/           # Critic architecture configs (Hydra config group)
│   │       ├── default.yaml  # Default critic (4 layers, width 8)
│   │       ├── test.yaml     # Test critic (smaller)
│   │       ├── d2w8.yaml     # 2 layers, width 8
│   │       ├── d3.yaml       # 3 hidden layers
│   │       ├── d4.yaml       # 4 hidden layers
│   │       ├── d5w8.yaml     # 5 layers, width 8
│   │       ├── d5w12.yaml    # 5 layers, width 12
│   │       ├── d5w20.yaml    # 5 layers, width 20
│   │       ├── d6w8.yaml     # 6 layers, width 8
│   │       ├── wgan1.yaml    # WGAN-GP, lambda=1
│   │       ├── wgan10.yaml   # WGAN-GP, lambda=10
│   │       └── wgan25.yaml   # WGAN-GP, lambda=25
│   ├── generative/           # Generative NLL training configs
│   │   ├── default.yaml      # Default generative config
│   │   └── test.yaml         # Test config
│   └── adversarial/          # Adversarial training configs
│       ├── test.yaml         # Test config
│       ├── pgd_at.yaml       # PGD-AT (Madry et al.)
│       └── trades.yaml       # TRADES (Zhang et al.)
├── tracking/                  # W&B and evaluation configs
│   ├── online.yaml           # Online W&B logging
│   └── test.yaml             # Test config
├── experiments/              # Full experiment configs (organized by task/architecture/phase)
│   ├── classification/       # Classification experiments
│   │   ├── fourier_d30D18/  # Fourier embedding, in_dim=30, bond_dim=18
│   │   │   ├── D18.yaml
│   │   │   ├── D18_sweep.yaml
│   │   │   ├── best/        # Best configs per dataset
│   │   │   │   ├── circles.yaml
│   │   │   │   ├── moons.yaml
│   │   │   │   └── spirals.yaml
│   │   │   ├── hpo/         # HPO experiments
│   │   │   │   ├── lr_hpo.yaml
│   │   │   │   ├── lrwd_hpo.yaml
│   │   │   │   └── sanity_check.yaml
│   │   │   └── seed_sweeps/ # Seed sweep experiments
│   │   │       ├── circles.yaml
│   │   │       ├── moons.yaml
│   │   │       └── spirals.yaml
│   │   └── fourier_d4D3/    # Fourier embedding, in_dim=4, bond_dim=3
│   │       └── hpo/
│   │           ├── circles.yaml
│   │           ├── moons.yaml
│   │           └── spirals.yaml
│   ├── adversarial/          # Adversarial training experiments
│   │   ├── fourier_d30D18/  # Fourier embedding, in_dim=30, bond_dim=18
│   │   │   ├── hpo/         # HPO experiments
│   │   │   │   ├── circles.yaml
│   │   │   │   ├── moons.yaml
│   │   │   │   └── spirals.yaml
│   │   │   ├── best/        # Best HPO configs per dataset
│   │   │   │   ├── circles.yaml
│   │   │   │   ├── moons.yaml
│   │   │   │   └── spirals.yaml
│   │   │   └── seed_sweeps/ # Seed sweep experiments
│   │   │       ├── circles.yaml
│   │   │       ├── moons.yaml
│   │   │       └── spirals.yaml
│   │   └── fourier_d10D4/   # Fourier embedding, in_dim=10, bond_dim=4
│   │       └── hpo/
│   │           └── hpo_test.yaml
│   ├── generative/           # Generative NLL experiments
│   │   └── fourier_d30D18/
│   │       ├── hpo/
│   │       │   └── hpo.yaml
│   │       └── seed_sweeps/
│   │           ├── circles.yaml
│   │           ├── moons.yaml
│   │           └── spirals.yaml
│   ├── ganstyle/             # GAN experiments
│   │   └── fourier_d30D18/
│   │       ├── critic_sweep.yaml
│   │       ├── default.yaml
│   │       ├── hpo.yaml
│   │       ├── no_retrain.yaml
│   │       └── wgan_sweep.yaml
│   └── tests/                # Test experiments (unchanged)
│       ├── classification.yaml
│       ├── classification_hpo.yaml
│       ├── classifications.yaml
│       ├── ganstyle.yaml
│       ├── ganstyle_hpo.yaml
│       ├── ganstyles.yaml
│       ├── adversarial.yaml
│       ├── generative.yaml
│       └── generative_hpo.yaml
└── hydra/                    # Hydra-specific configs
    └── job_logging/
        ├── debug.yaml        # Verbose logging
        └── stream_only.yaml  # Console-only logging
```

## Main Config (`config.yaml`)

```yaml
experiment: default

defaults:
  - _self_                              # Allow overrides
  - dataset: 2Dtoy/circles_2k           # Default dataset (note subfolder!)
  - born: d4D3                          # Default model
  - trainer/classification: test        # Default classifier training
  - trainer/ganstyle: test              # Default GAN training
  - trainer/generative: null            # Generative training disabled by default
  - trainer/adversarial: test           # Default adversarial training
  - tracking: test                      # Default tracking
  - override hydra/job_logging: stream_only
  - override hydra/hydra_logging: none

hydra:
  run:
    dir: outputs/${experiment}_${dataset.name}_${now:%d%b%y_%I%p%M}
  sweep:
    dir: outputs/${experiment}_${dataset.name}_${now:%d%b%y}
    subdir: ${hydra.job.num}
  output_subdir: .hydra
```

## Config Group Reference

### born/ — BornMachine Configs

Naming convention: `d{in_dim}D{bond_dim}`

```yaml
# born/d10D4.yaml
init_kwargs:
  in_dim: 10           # Physical/embedding dimension
  bond_dim: 4          # Bond dimension (expressivity)
  boundary: "obc"      # Open boundary conditions
  init_method: "randn_eye"
  std: 1e-9
embedding: "fourier"   # Embedding type
```

**Parameters:**
| Parameter | Description | Impact |
|-----------|-------------|--------|
| `in_dim` | Embedding dimension | Higher = more expressive but slower |
| `bond_dim` | Bond dimension | Higher = more expressive but slower |
| `boundary` | `"obc"` or `"pbc"` | Open vs periodic boundaries |
| `init_method` | Tensor initialization | `"randn"`, `"randn_eye"`, etc. |
| `embedding` | `"fourier"` or `"legendre"` | Input mapping type |

### dataset/ — Dataset Configs

Dataset configs are organized by dataset type in subdirectories:

```
dataset/
├── test.yaml          # Minimal test config
├── 2Dtoy/             # 2D toy datasets (current)
│   ├── moons_*.yaml
│   ├── circles_*.yaml
│   └── spirals_*.yaml
├── mnist/             # MNIST (planned)
└── timeseries/        # Univariate time series (planned)
```

**2D Toy Datasets** (`dataset/2Dtoy/`):
```yaml
# dataset/2Dtoy/moons_4k.yaml
name: "moons_4k"
gen_dow_kwargs:
  name: "moons_4k"
  size: 4000           # Total samples
  seed: 25             # Generation seed
  noise: 0.05          # Noise level
split: [0.5, 0.25, 0.25]  # Train/valid/test
split_seed: 11
overwrite: false       # Regenerate dataset even if cached (for seed sweeps)
```
- `dataset: 2Dtoy/moons_4k` (do not forget the subfolder)
- Set `overwrite: true` when sweeping over dataset seeds to ensure fresh data each run

### trainer/classification/ — Classifier Training

```yaml
# trainer/classification/adam500_loss.yaml
max_epoch: 500
batch_size: 64
criterion:
  name: "negative log-likelihood"
  kwargs: {eps: 1e-8}
optimizer:
  name: "adam"
  kwargs: {lr: 1e-4, weight_decay: 0.0}
patience: 50           # Early stopping patience
stop_crit: "clsloss"   # Monitored metric ("clsloss", "genloss", "acc", "fid", "rob")
watch_freq: 1000       # Gradient logging frequency
metrics: {"clsloss": 1, "acc": 1, "viz": 30, "fid": 30, "rob": 30}
save: true
auto_stack: true       # tensorkrowch setting
auto_unbind: false     # tensorkrowch setting
```

**Available classification configs:**
| Config | Epochs | Stop Criterion | Description |
|--------|--------|----------------|-------------|
| `adam500_loss` | 500 | `clsloss` | Stop on validation loss |
| `adam500_rob` | 500 | `rob` | Stop on robustness |
| `test` | 10 | `clsloss` | Quick test config |

### trainer/ganstyle/ — GAN Training

The critic configuration is a Hydra config group under `critic/`, enabling easy sweeping over architectures.

```yaml
# trainer/ganstyle/default.yaml
defaults:
  - critic: default      # Loads critic/default.yaml via Hydra composition

max_epoch: 100
sampling:
  num_bins: 200          # Sampling resolution
  num_spc: 128           # Samples per class
  batch_spc: 16          # Sampling batch size
  method: secant         # Sampling method

r_real: 1.0              # Real/synthetic ratio
optimizer: {...}         # Generator optimizer
watch_freq: 100
metrics: {"clsloss": 1, "acc": 1, "rob": 10, "viz": 10, "fid": 10}
retrain_crit: "acc"      # When to retrain
tolerance: 0.05          # Accuracy drop tolerance
retrain: {...}           # ClassificationConfig for retraining
save: false
```

### trainer/ganstyle/critic/ — Critic Architecture Configs

Critic configs are a Hydra config group, composed via the `defaults` list. Fields are at root level (no `critic:` wrapper).

```yaml
# trainer/ganstyle/critic/d2.yaml
backbone:
  architecture: mlp
  model_kwargs:
    hidden_multipliers: [8.0, 8.0]  # 2 hidden layers
    nonlinearity: LeakyReLU
    negative_slope: 0.01
head:
  class_aware: true
  architecture: linear
  model_kwargs: {}
discrimination:
  max_epoch_pre: 100   # Critic pre-training epochs
  max_epoch_gan: 20    # Critic steps per generator step
  batch_size: 32
  optimizer: {...}
  patience: 25
criterion:
  name: "BCE"          # Or "wgan" for WGAN-GP
  kwargs: null
```

**Available critic configs:**
| Config | Hidden Layers | Description |
|--------|---------------|-------------|
| `d2w8` | `[8.0, 8.0]` | Shallow, 2 layers, width 8 |
| `d3` | `[8.0, 8.0, 8.0]` | Medium, 3 layers |
| `d4` | `[8.0, 8.0, 8.0, 8.0]` | Deep, 4 layers |
| `d5w8` | `[8.0, 8.0, 8.0, 8.0, 8.0]` | Very deep, 5 layers, width 8 |

**Sweeping over critic architectures:**
```bash
# Command line multirun
python -m experiments.ganstyle --multirun 'trainer/ganstyle/critic=d2w8,d3,d4,d5w8'
```

```yaml
# Or in experiment config
hydra:
  sweeper:
    params:
      trainer/ganstyle/critic: d2w8, d3, d4, d5w8
```

### trainer/generative/ — Generative Training

```yaml
# trainer/generative/default.yaml
max_epoch: 100
batch_size: 64
optimizer:
  name: "adam"
  kwargs: {lr: 1e-4, weight_decay: 0.0}
criterion:
  name: "nll"             # Uses GenerativeNLL from criterions.py
  kwargs: {eps: 1e-8}
patience: 50
stop_crit: "genloss"      # Monitor generative NLL loss
watch_freq: 100
metrics: {"genloss": 1, "acc": 1, "fid": 10, "viz": 10}
save: false
auto_stack: true
auto_unbind: false
```

### tracking/ — Experiment Tracking

```yaml
# tracking/online.yaml
project: gan_train              # W&B project
entity: your-wandb-entity       # W&B entity
mode: online                    # online, offline, disabled
seed: 42                        # Training seed (model init, shuffling, PGD)
sampling:                       # Evaluation sampling config
  num_bins: 200
  num_spc: 2048
  batch_spc: 64
  method: secant
evasion:                        # Robustness evaluation config
  method: "FGM"               # Attack method: "FGM" or "PGD"
  norm: "inf"
  criterion: {...}
  strengths: [0.1, 0.3]
  # PGD-specific parameters (ignored for FGM):
  num_steps: 10               # Number of PGD iterations
  step_size: null             # Step size (null = auto: 2.5 * strength / num_steps)
  random_start: true          # Start from random point in epsilon ball
```

### experiments/ — Full Experiments

Experiment configs override defaults. The main experiment directories are:
- `classification/` — Classification-only experiments
- `adversarial/` — Adversarial training experiments
- `generative/` — Generative NLL training experiments
- `ganstyle/` — GAN-style training experiments
- `tests/` — Quick test experiments

```yaml
# experiments/tests/classification.yaml
# @package _global_
experiment: classification_test

defaults:
  - override /trainer/ganstyle: null      # Disable GAN training
  - override /trainer/adversarial: null   # Disable adversarial training
```

The `# @package _global_` directive makes overrides apply at the root level.

**Best configs** (`experiments/classification/fourier_d30D18/best/`):
Store validated best configurations for each dataset:
```yaml
# experiments/classification/fourier_d30D18/best/moons.yaml
# @package _global_
experiment: best_moons_4k

defaults:
  - override /born: d30D18
  - override /dataset: 2Dtoy/moons_4k
  - override /trainer/classification: adam500_loss
  # ... validated hyperparameters
```

**Seed sweeps** (`experiments/classification/fourier_d30D18/seed_sweeps/`):
Run the same config with different random seeds for statistical analysis.

## Running Experiments

See `experiments/GUIDE.md` for detailed usage and bash commands.

### Multirun (Grid Sweep)

Define sweep parameters in an experiment config:

```yaml
# configs/experiments/classification/D18_sweep.yaml
# @package _global_
experiment: D18_sweep

defaults:
  - override /born: d30D18
  - override /dataset: 2Dtoy/moons_4k
  # ... other settings

hydra:
  sweeper:
    params:
      born: d10D4,d10D6,d30D18
      trainer.classification.optimizer.kwargs.lr: 1e-3,1e-4
```

### Hyperparameter Optimization (HPO) with Optuna

Use the Optuna sweeper for Bayesian hyperparameter optimization (TPE).

**Enable Optuna in experiment config** (add to defaults):
```yaml
defaults:
  - override /hydra/sweeper: optuna
```

**HPO Experiment Config Example**:
```yaml
# configs/experiments/classification/fourier_d30D18/hpo/lrwd_hpo.yaml
# @package _global_
experiment: classification_hpo

defaults:
  - override /born: d10D4
  - override /dataset: 2Dtoy/moons_4k
  - override /trainer/classification: adam500_loss
  - override /tracking: online
  - override /trainer/ganstyle: null
  - override /trainer/adversarial: null
  - override /hydra/sweeper: optuna  # Enable Optuna

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10

    direction: minimize  # or maximize
    n_trials: 50
    n_jobs: 1

    # Search space
    params:
      trainer.classification.optimizer.kwargs.lr: tag(log, interval(1e-5, 1e-2))
      trainer.classification.optimizer.kwargs.weight_decay: interval(0.0, 0.1)
      born: choice(d10D3, d10D4, d10D6)
      born.init_kwargs.bond_dim: range(2, 10)
```

**Search Space Syntax** (Optuna-specific):
| Syntax | Description | Example |
|--------|-------------|---------|
| `interval(low, high)` | Uniform float | `interval(0.0, 1.0)` |
| `tag(log, interval(low, high))` | Log-uniform float | `tag(log, interval(1e-5, 1e-2))` |
| `range(low, high)` | Integer range | `range(2, 10)` |
| `range(low, high, step)` | Integer with step | `range(0, 100, 10)` |
| `choice(a, b, c)` | Categorical | `choice(adam, sgd, rmsprop)` |

**HPO Notes & Limitations**:
- **Valid `stop_crit` values**: `"clsloss"`, `"genloss"`, `"acc"`, `"fid"`, `"rob"`. Do NOT use `"viz"` (not numeric).
- **Robustness (`stop_crit: "rob"`)**: Uses average across all evaluated perturbation strengths.
- **Direction must be `minimize`**: All entry points negate acc/rob metrics internally for minimization. Do not change `direction: maximize` in Optuna config.
- **Objective selection**: All entry points return `trainer.best[stop_crit]`. Ensure your chosen `stop_crit` metric is included in your metrics config.

**Persistent Storage** (resume studies):
```yaml
hydra:
  sweeper:
    storage: "sqlite:///outputs/hpo_study.db"
    study_name: my_study  # Reuse same name to resume
```

**Note on Sweeper Configuration**:
Do NOT create a custom `configs/hydra/sweeper/optuna.yaml` file. This triggers deprecated Hydra 1.1 automatic schema matching. Instead, configure the Optuna sweeper entirely within your experiment configs using the `hydra.sweeper` section as shown above. The `- override /hydra/sweeper: optuna` default loads the plugin's built-in configuration.

See `experiments/GUIDE.md` for running HPO experiments.

## Config Inheritance

```
                    config.yaml (defaults)
                         │
                         ▼
              ┌──────────┴──────────┐
              │                     │
         born/d4D3.yaml    dataset/circles_2k.yaml ...
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                experiments/*.yaml (overrides)
                         │
                         ▼
                  Command line (final overrides)
```

**IMPORTANT — Defaults List Ordering:**
In Hydra, the **last entry for a config group** in the `defaults` list wins. In `config.yaml`, defaults from the main file are appended *after* the experiment's defaults. This means that the non-override entries in `config.yaml` (e.g., `- dataset: 2Dtoy/circles_2k`) will silently override any `- override /dataset: ...` in your experiment config, because the `config.yaml` entry comes last.

For example, this was the root cause of a bug where adversarial training always reverted to the `circles_2k` dataset: the experiment config had `- override /dataset: 2Dtoy/moons_4k`, but `config.yaml` listed `- dataset: 2Dtoy/circles_2k` after it, which took precedence. The fix is to use `override` in `config.yaml` itself (which is already done for Hydra logging entries) or to ensure your experiment config entries are not silently shadowed.

## Schema ↔ Config Mapping

Every config file corresponds to a dataclass in `src/utils/schemas.py`:

| Schema | Config Directory |
|--------|-----------------|
| `Config` | `config.yaml` (root) |
| `DatasetConfig` | `dataset/` |
| `BornMachineConfig` | `born/` |
| `ClassificationConfig` | `trainer/classification/` |
| `GANStyleConfig` | `trainer/ganstyle/` |
| `CriticConfig` | `trainer/ganstyle/critic/` |
| `GenerativeConfig` | `trainer/generative/` |
| `AdversarialConfig` | `trainer/adversarial/` |
| `TrackingConfig` | `tracking/` |

**If you change a schema, update corresponding configs!**

## Tips

- **W&B offline**: Set `tracking.mode=offline` for no network access
- **Reproducibility**: `tracking.seed` controls training randomness (model init, shuffling, PGD); `dataset.split_seed` and `gen_dow_kwargs.seed` control data pipeline randomness independently
- See `experiments/GUIDE.md` for output directory structure and bash commands
