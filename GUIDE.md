# Born Machines Codebase Guide

## Motivation

This codebase investigates whether **Born Machines** based on **Matrix Product States (MPS)** trained as **generative classifiers** are more robust than discriminatively-trained models, and whether they can achieve state-of-the-art robust classification accuracy.

The research hypothesis is that generative models (which learn p(x,c) rather than just p(c|x)) may provide inherent robustness against adversarial attacks due to their understanding of the data distribution.

## Core Concepts

### Born Machines & Born Rule

Born Machines model probability distributions using the Born Rule from quantum mechanics:
- **Probability = |amplitude|²** — the probability of an outcome is the squared magnitude of an amplitude
- Inputs are mapped to a higher-dimensional **Hilbert Space** via embeddings (currently Fourier, also supports Legendre/polynomial)
- Each input feature is embedded independently; the complete input is a "product state"
- The Born Machine is a **tensor network** with structured factorization

### Matrix Product State (MPS) Structure

```
    ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
    │ T₀├───┤ T₁├───┤ T₂├───┤...├───┤ Tₙ│
    └─┬─┘   └─┬─┘   └─┬─┘   └───┘   └─┬─┘
      │       │       │               │
     x₀      x₁      x₂              xₙ
```

- Linear chain of tensors connected by "bond" indices
- Each tensor has a "physical" index connecting to one input feature
- **Bond dimension** controls model expressivity
- Contracting the MPS with an embedded input yields a scalar (amplitude)

### Classification with BornClassifier

The `BornClassifier` adds a special output tensor:

```
    ┌───┐   ┌───┐   ┌───┬───┐   ┌───┐
    │ T₀├───┤ T₁├───┤ T₂│out├───┤ Tₙ│
    └─┬─┘   └─┬─┘   └─┬─┴─┬─┘   └─┬─┘
      │       │       │   │       │
     x₀      x₁      x₂  cls     xₙ
```

- The output tensor has an extra "class" leg that isn't contracted
- Contracting with input leaves a **vector** of dimension `num_classes`
- **Square each component and normalize** → p(c|x)

**CRITICAL**: The forward pass outputs **amplitudes**, not logits! You must square them to get probabilities. This is handled internally by `BornClassifier.probabilities()` and `BornClassifier.parallel()`.

### Sampling with BornGenerator

The `BornGenerator` is equally important — it enables **sampling from the learned distribution** p(x|c), which is essential for:
- GAN-style training (generating synthetic samples)
- Evaluating generative quality (FID, visualization)
- The core hypothesis of this research (generative models → robustness)

**Sequential Sampling Process:**
```
For feature i = 0, 1, ..., D-1 (conditioned on class c):
  1. Compute marginal p(xᵢ | x₀,...,xᵢ₋₁, c) via partial MPS contraction
  2. Sample xᵢ using differentiable inverse-transform sampling
  3. Embed sampled xᵢ and add to conditioning set
  4. Repeat for next feature
```

The `BornGenerator` uses **sequential contraction** (not parallel like the classifier) because sampling requires computing marginal distributions over unobserved variables. This is computationally more expensive but enables exact sampling from the Born distribution.

**Key implementation**: `src/models/generator/generator.py` and `src/models/generator/differential_sampling.py`

### Training Architecture

**Training Pipeline:**
1. **Classification Training** — Train MPS as discriminative classifier first
2. **GAN-style Training** (optional) — Improve generative capabilities using adversarial training with a critic/discriminator
3. **Adversarial Training** (planned) — Train for robustness against adversarial examples

**Generative Training (GAN-style):**
- Uses standard GAN loss or Wasserstein distance with gradient penalty
- Requires **differentiable sampling** (see `src/models/generator/differential_sampling.py`)
- Sampling uses product rule: p(x) = ∏ᵢ p(xᵢ | x₁,...,xᵢ₋₁)
- Uses secant-based inverse transform sampling for gradient flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        experiments/                              │
│  classification.py, ganstyle.py (entry points with hydra)       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                          src/                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   models/   │  │   trainer/  │  │  tracking/  │              │
│  │             │  │             │  │             │              │
│  │ BornMachine │  │ Classific.  │  │ Evaluator   │              │
│  │ BornClassif.│  │ GANStyle    │  │ Metrics     │              │
│  │ BornGener.  │  │ Adversarial │  │ W&B utils   │              │
│  │ Critic      │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│        │                │                │                       │
│  ┌─────▼────────────────▼────────────────▼──────┐               │
│  │                    utils/                     │               │
│  │  schemas.py, get.py, evasion/, _utils.py     │               │
│  └──────────────────────────────────────────────┘               │
│        │                                                         │
│  ┌─────▼────────────────────────────────────────┐               │
│  │                    data/                      │               │
│  │  DataHandler, gen_n_load.py                  │               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        configs/                                  │
│  Hydra configuration files (dataset, born, trainer, tracking)   │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
adversarial_training_mps/
├── GUIDE.md                    # This file
├── README.md                   # Original readme (may be outdated)
├── environment.yml             # Conda environment specification
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config with defaults
│   ├── born/                  # BornMachine configs (bond_dim, in_dim, embedding)
│   ├── dataset/               # Dataset configs (moons, circles, spirals)
│   ├── trainer/               # Training configs
│   │   ├── classification/    # Classifier training configs
│   │   ├── ganstyle/          # GAN-style training configs
│   │   └── adversarial/       # Adversarial training configs
│   ├── tracking/              # W&B tracking configs
│   ├── experiments/           # Full experiment configs (override defaults)
│   └── hydra/                 # Hydra-specific configs (logging)
├── experiments/               # Entry point scripts
│   ├── classification.py      # Classification-only training
│   └── ganstyle.py           # Classification + GAN-style training
├── src/                       # Main source code
│   ├── models/               # Model definitions (see src/models/GUIDE.md)
│   ├── trainer/              # Training loops (see src/trainer/GUIDE.md)
│   ├── tracking/             # Evaluation & logging
│   ├── data/                 # Data loading & preprocessing
│   └── utils/                # Utilities, schemas, embeddings
├── .datasets/                # Generated/downloaded datasets (git-ignored)
├── outputs/                  # Experiment outputs (git-ignored)
├── wandb/                    # W&B local files
├── notebooks/                # Jupyter notebooks
└── archive/                  # Old/deprecated code
```

## Key Entry Points

### Designing Experiments

**The primary way to configure experiments is via `configs/experiments/` files**, not command-line overrides. This ensures reproducibility and clear documentation of experiment settings.

**Workflow:**
1. Create/modify an experiment config in `configs/experiments/`
2. Run with `+experiments=<path>` to apply it

**Example experiment config** (`configs/experiments/pretraining/D18.yaml`):
```yaml
# @package _global_
experiment: pretraining_D18

defaults:
  - override /born: d30D18
  - override /dataset: moons_4k
  - override /trainer/classification: adam_b64e300
  - override /tracking: online
  - override /trainer/ganstyle: null      # Disable GAN training
  - override /trainer/adversarial: null   # Disable adversarial training
```

### Running Experiments

```bash
# Run with experiment config (RECOMMENDED)
python -m experiments.classification +experiments=pretraining/D18

# Run GAN-style experiment
python -m experiments.ganstyle +experiments=ganstyle/default

# Multirun sweep (define sweep params in experiment config)
python -m experiments.classification --multirun +experiments=pretraining/D18_sweep
```

**Command-line overrides** are useful for quick tests but not for production experiments:
```bash
# Quick test with small epoch count
python -m experiments.classification +experiments=tests/classification trainer.classification.max_epoch=10

# Disable W&B for local debugging
python -m experiments.classification +experiments=tests/classification tracking.mode=disabled
```

## Configuration System

The codebase uses **Hydra** for configuration management. Key concepts:

1. **Main config**: `configs/config.yaml` defines defaults
2. **Config groups**: Each subdirectory (born/, dataset/, etc.) is a group
3. **Experiment configs**: `configs/experiments/` override defaults for specific experiments (primary workflow)
4. **Schemas**: Dataclasses in `src/utils/schemas.py` define structure

**IMPORTANT**: If you modify `schemas.py`, you MUST also:
1. Update corresponding YAML files in `configs/`
2. Update any code that uses the changed fields

Example schema → config correspondence:
```python
# schemas.py
@dataclass
class BornMachineConfig:
    init_kwargs: MPSInitConfig
    embedding: str
```
```yaml
# configs/born/d10D4.yaml
init_kwargs:
  in_dim: 10
  bond_dim: 4
  boundary: "obc"
embedding: "fourier"
```

## Critical Dependencies

| Library | Purpose |
|---------|---------|
| `tensorkrowch` | Tensor network operations, MPS implementation |
| `hydra-core` | Configuration management |
| `wandb` | Experiment tracking and visualization |
| `torch` | Neural network primitives, autograd |
| `sklearn` | Data generation, preprocessing |

## Navigation Guide

### Where to find specific functionality:

| Task | Location |
|------|----------|
| Modify MPS architecture | `src/models/classifier.py`, `src/models/born.py` |
| Change embedding | `src/utils/get.py` (`_EMBEDDING_MAP`) |
| Add new loss function | `src/utils/get.py` (`_CLASSIFICATION_LOSSES`) |
| Modify training loop | `src/trainer/classification.py`, `src/trainer/ganstyle.py` |
| Add evaluation metric | `src/tracking/evaluator.py` |
| Change data preprocessing | `src/data/handler.py` |
| Add new dataset | `src/data/gen_n_load.py` |
| Configure experiments | `configs/` directory |
| Add adversarial attack | `src/utils/evasion/minimal.py` |

### Critical code sections to understand:

1. **`BornMachine.__init__`** (`src/models/born.py:19-67`) — How classifier and generator share tensors
2. **`BornClassifier.parallel`** (`src/models/classifier.py:54-87`) — Forward pass with Born rule
3. **`BornGenerator._single_class`** (`src/models/generator/generator.py:122-199`) — Sequential sampling
4. **`os_secant`** (`src/models/generator/differential_sampling.py:46-79`) — Differentiable sampling
5. **`Trainer.train`** (`src/trainer/classification.py:183-210`) — Main training loop

## Known Issues & Gotchas

1. **Import Paths**: Some files use relative imports like `from models import ...` instead of `from src.models import ...`. This can cause issues.

2. **`ganstyle.py` trainer** has incorrect imports at line 9: `from data.handler import DataHandler` should be `from src.data.handler import DataHandler`

3. **FID metric** assumes data dimension < 100 (disabled for larger datasets)

4. **Docstrings may be outdated** — Trust the actual code implementation over docstrings

5. **The `BornMachine.sync_tensors` method** is critical after training — without it, classifier and generator can get out of sync

6. **`adversarial.py` trainer** is a stub — not yet implemented

## Future Directions (from README)

1. **More adversarial attack methods**: Currently only FGM, need PGD and others
2. **MNIST support**: Currently only 2D toy data
3. **MPS as discriminator backbone**: Using MPS features as input to critic
4. **More datasets**: Time series, higher-dimensional data
5. **Adversarial training**: Full implementation of robust training

## Quick Reference: Common Commands

```bash
# Setup environment
conda env create -f environment.yml
conda activate <env_name>

# Run classification experiment (using experiment config)
python -m experiments.classification +experiments=pretraining/D18

# Run GAN-style experiment
python -m experiments.ganstyle +experiments=ganstyle/default

# Quick test run
python -m experiments.classification +experiments=tests/classification

# Run with W&B disabled (for debugging)
python -m experiments.classification +experiments=tests/classification tracking.mode=disabled

# Check W&B dashboard
# Visit: https://wandb.ai/<entity>/<project>
```

---

For detailed module documentation, see:
- `src/models/GUIDE.md` — Model architecture details
- `src/trainer/GUIDE.md` — Training pipeline details
- `src/utils/GUIDE.md` — Utilities and configuration details
