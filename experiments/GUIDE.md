# Experiments Guide

Entry point scripts for running experiments. All scripts are run as Python modules from the project root.

## Entry Points

| Script | Purpose |
|--------|---------|
| `train.py` | Unified Hydra entry point — NLL (dis/gen) and adversarial training |
| `run_local.py` | Self-contained local runner — no Hydra, no W&B; edit CONFIG BLOCK |
| `batch.py` | Batch-run/list HPO and seed_sweep configs, skip already-run |

---

## `train.py` — Hydra entry point

Mode is inferred from config: if `trainer.adversarial` is set → adversarial; if `trainer.nll` is set → NLL (dis or gen depending on `alpha`).

### Basic Usage

```bash
# NLL discriminative run
python -m experiments.train +experiments=nll/dis/fourier/d4r3/hpo/moons

# NLL generative run
python -m experiments.train +experiments=nll/gen/legendre/d10r6/hpo/circles

# Adversarial training run
python -m experiments.train +experiments=adversarial/fourier/d4r3/hpo/moons

# Quick test run (tiny config, no W&B)
python -m experiments.train +experiments=tests/nll         tracking.mode=disabled
python -m experiments.train +experiments=tests/adversarial tracking.mode=disabled
```

### Command-Line Overrides

```bash
# Fewer epochs
python -m experiments.train +experiments=tests/nll \
    trainer.nll.max_epoch=10 tracking.mode=disabled

# Disable W&B
python -m experiments.train +experiments=tests/nll \
    tracking.mode=disabled

# Different learning rate
python -m experiments.train +experiments=tests/nll \
    trainer.nll.optimizer.kwargs.lr=1e-3 tracking.mode=disabled
```

### Multirun (Grid / Seed Sweep)

```bash
python -m experiments.train --multirun +experiments=nll/gen/legendre/d10r6/seed_sweep/circles
```

### Debug config (print resolved config without running)

```bash
python -m experiments.train --cfg job +experiments=tests/nll
```

### Hyperparameter Optimization (Optuna)

**Option 1** — Specify sweeper in experiment config (recommended):
```yaml
# configs/experiments/nll/dis/fourier/d4r3/hpo/moons.yaml
defaults:
  - override /hydra/sweeper: optuna
```
```bash
python -m experiments.train --multirun +experiments=nll/dis/fourier/d4r3/hpo/moons
```

**Option 2** — Specify sweeper on the command line:
```bash
python -m experiments.train --multirun \
    hydra/sweeper=optuna \
    +experiments=nll/dis/fourier/d4r3/hpo/moons
```

**HPO objective** — `train.py` returns a scalar: `trainer.best[stop_crit]`, negated for `acc`/`rob` so Optuna can always minimize. Keep `direction: minimize` in your Optuna config.

Valid `stop_crit` values: `"dis_loss"`, `"gen_loss"`, `"acc"`, `"rob"`.

---

## `run_local.py` — self-contained runner

No Hydra, no W&B. Edit the CONFIG BLOCK at the top of the file, then run:

```bash
python -m experiments.run_local
```

Key CONFIG BLOCK fields:

| Variable | Values | Notes |
|----------|--------|-------|
| `REGIME` | `"nll"` \| `"adversarial"` | Selects trainer |
| `DATASET_CFG` | `DatasetConfig(name=...)` | See inline comments for dataset names |
| `CBM_CFG` | `CBMConfig(init_kwargs=MPSInitConfig(...))` | Architecture + embedding |
| `NLL_CFG` | `NLLConfig(alpha=...)` | `alpha=0` → dis, `alpha=1` → gen |
| `ADV_CFG` | `AdversarialConfig(...)` | PGD-AT or FGM |
| `MODEL_PATH` | `Path(...)` \| `None` | Pre-trained checkpoint to fine-tune from |

---

## `batch.py` — batch runner

Discovers all `hpo/`, `seed_sweep/`, and `grid_sweep/` configs under `configs/experiments/` and runs them sequentially, skipping any that already have a matching output directory.

Config layout expected:
```
configs/experiments/nll/{dis,gen}/{embedding}/{arch}/{kind}/{dataset}.yaml
configs/experiments/adversarial/{embedding}/{arch}/{kind}/{dataset}.yaml
```

### Usage

```bash
# List all discovered configs with [ran]/[   ] status
python -m experiments.batch --list

# Dry-run: print commands without executing
python -m experiments.batch --dry-run

# Run everything that hasn't been run yet
python -m experiments.batch

# Filter by type (dis | gen | adv or full name)
python -m experiments.batch --type gen --dry-run

# Filter by embedding
python -m experiments.batch --embedding legendre --dry-run

# Filter by architecture (exact match)
python -m experiments.batch --arch d10r6 --dry-run

# Filter by kind (seed_sweep | hpo | grid_sweep)
python -m experiments.batch --kind hpo --dry-run

# Filter by dataset (substring match)
python -m experiments.batch --dataset circles --dry-run

# Re-run even if output already exists
python -m experiments.batch --force --dry-run
```

---

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
- `{regime}`: `nll` (dis/gen) | `adv`
- `{arch}`: `d{in_dim}r{bond_dim}` — e.g. `d4r3`, `d30r18`
- `{dataset}`: dataset name — e.g. `moons`, `circles`
- `{date}`: `DDMM` for multiruns, `DDMM_HHMM` for single runs
