# Utils Module Guide

This module contains utilities, configuration schemas, helper functions, and adversarial attack implementations.

## Module Structure

```
src/utils/
├── GUIDE.md              # This file
├── __init__.py           # Exports: get, schemas, set_seed
├── schemas.py            # Hydra config dataclasses (CRITICAL)
├── get.py                # Factory functions for optimizers (re-exports embeddings, criterions)
├── embeddings.py         # Embedding functions (fourier, legendre, poly)
├── criterions.py         # Loss functions (ClassificationNLL, GenerativeNLL)
├── _utils.py             # Misc utilities (seed setting, sample QC)
└── evasion/
    ├── __init__.py
    └── minimal.py        # Adversarial attack implementations (FGM, PGD)
└── purification/
    ├── __init__.py
    └── minimal.py        # Likelihood-based purification (LikelihoodPurification)
```

## schemas.py — Configuration Schemas

**CRITICAL FILE**: This file defines the structure of all configuration objects using dataclasses.

### The Schema → Config Correspondence

Every dataclass in `schemas.py` corresponds to YAML config files:

```
schemas.py                     configs/
─────────────────────────────────────────────────────
Config                    →    config.yaml
DatasetConfig             →    dataset/*.yaml
BornMachineConfig         →    born/*.yaml
ClassificationConfig      →    trainer/classification/*.yaml
GANStyleConfig            →    trainer/ganstyle/*.yaml
CriticConfig              →    trainer/ganstyle/critic/*.yaml
GenerativeConfig          →    trainer/generative/*.yaml
AdversarialConfig         →    trainer/adversarial/*.yaml
TrackingConfig            →    tracking/*.yaml
```

### Key Config Classes

```python
@dataclass
class Config:                    # Top-level config
    dataset: DatasetConfig       # Data generation/loading
    born: BornMachineConfig      # MPS architecture
    trainer: TrainerConfig       # Training settings
    tracking: TrackingConfig     # W&B, evaluation
    experiment: str = "default"  # Experiment name

@dataclass
class TrainerConfig:             # Wrapper for trainer configs
    classification: ClassificationConfig | None
    ganstyle: GANStyleConfig | None
    generative: GenerativeConfig | None
    adversarial: AdversarialConfig | None
```

### When You Change schemas.py

**You MUST also update:**

1. **Corresponding YAML files** in `configs/`
2. **Code that uses the changed fields** (grep for usages)
3. **Default values** must match between schema and YAML

**Example change flow:**
```python
# 1. Add field to schema
@dataclass
class ClassificationConfig:
    ...
    new_option: bool = False  # NEW

# 2. Add to YAML configs
# configs/trainer/classification/adam500_loss.yaml
new_option: false

# 3. Update code that uses it
if train_cfg.new_option:
    do_something()
```

### Hydra Integration

Schemas are registered with Hydra's ConfigStore:

```python
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()

cs.store(group="dataset", name="schema", node=DatasetConfig)
cs.store(group="model/born", name="schema", node=BornMachineConfig)
cs.store(name="base_config", node=Config)
```

## embeddings.py — Embedding Functions

Provides embedding functions for mapping input data to higher-dimensional spaces.

```python
from src.utils.get import embedding, range_from_embedding

# Get embedding function
embed_fn = embedding("fourier")
embedded = embed_fn(data, in_dim=10)  # (batch, D) → (batch, D, in_dim)

# Get valid input range for embedding
input_range = range_from_embedding("fourier")  # (0.0, 1.0)
```

**Available Embeddings:**
| Name | Range | Description |
|------|-------|-------------|
| `"fourier"` | (0, 1) | Fourier basis functions (tensorkrowch) |
| `"poly"` / `"polynomial"` | varies | Polynomial embedding (tensorkrowch) |
| `"legendre"` | (-1, 1) | Legendre polynomials (custom implementation) |
| `"hermite"` | (0, 1) | Normalized physicist's Hermite functions |

**Legendre Embedding** (`embeddings.py:11-49`):
Uses the standard recursive formula for Legendre polynomials:
```
P_0(x) = 1
P_1(x) = x
P_n(x) = ((2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n
```

## get.py — Optimizer Factory

Provides optimizer instantiation and re-exports embedding/criterion functions.

### Optimizers

```python
from src.utils.get import optimizer

opt = optimizer(model.parameters(), cfg.optimizer)
# cfg.optimizer = OptimizerConfig(name="adam", kwargs={"lr": 1e-4})
```

**Available Optimizers:**
`sgd`, `adam`, `adamw`, `rmsprop`, `adagrad`, `adamax`, `nadam`

### Helper Functions

```python
from src.utils.get import indim_and_ncls

# Extract in_dim and num_classes from an MPS model
in_dim, n_cls = indim_and_ncls(mps)
```

## criterions.py — Loss Functions

Provides loss functions suitable for BornMachines (which output amplitudes/probabilities, not logits).

### Classification Loss

```python
from src.utils.get import criterion

loss_fn = criterion("classification", cfg.criterion)
# cfg.criterion = CriterionConfig(name="nll", kwargs={"eps": 1e-12})

loss = loss_fn(probabilities, labels)
```

**ClassificationNLL** (`criterions.py:16-47`):
```python
class ClassificationNLL(nn.Module):
    """Custom NLL loss for MPS classifiers."""
    def forward(self, p, t):
        p = p.clamp(min=self.eps)
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()
```

Note: This expects **probabilities** (already squared and normalized), not raw amplitudes!

### Generative Loss

```python
from src.utils.get import criterion

loss_fn = criterion("generative", cfg.criterion)
loss = loss_fn(bornmachine, data, labels)
```

**GenerativeNLL** (`criterions.py:52-94`):
Computes NLL for the joint distribution p(x,c):
```
-log(p(x,c)) = -log(|psi(x,c)|^2) + log(Z)
```

**Available Loss Functions:**
| Mode | Name | Aliases | Description |
|------|------|---------|-------------|
| `"classification"` | `"nll"` | `"nlll"`, `"negativeloglikelihood"` | Classification NLL |
| `"generative"` | `"nll"` | `"nlll"`, `"negativeloglikelihood"` | Generative NLL |

## _utils.py — Miscellaneous Utilities

### Seed Setting

```python
from src.utils import set_seed

set_seed(42)  # Seeds training randomness (model init, shuffling, PGD)
```

Called **after** data loading and **before** model creation in every entry point.
Controls training randomness only — data pipeline seeds (`gen_dow_kwargs.seed`,
`dataset.split_seed`) are handled independently by their respective functions.

Sets seeds for:
- Python `random`
- NumPy
- PyTorch (CPU and GPU)
- cuDNN deterministic mode
- `PYTHONHASHSEED` environment variable

### Sample Quality Control

```python
from src.utils._utils import sample_quality_control

sample_quality_control(synths, upper=1.0, lower=0.0)
# Logs warnings for out-of-bound values
```

Useful for debugging sampling issues.

## evasion/minimal.py — Adversarial Attacks

Implements adversarial attack methods for robustness evaluation.

### Fast Gradient Method (FGM)

```python
from src.utils.evasion.minimal import FastGradientMethod

fgm = FastGradientMethod(
    norm="inf",           # L-infinity norm (also supports p-norms)
    criterion=CriterionConfig(name="nll", kwargs=None)
)

adversarial = fgm.generate(
    born=bornmachine,
    naturals=data,        # Clean inputs
    labels=labels,
    strength=0.1,         # Perturbation magnitude
    device=device
)
```

**FGM Algorithm:**
```
1. Compute loss on natural examples
2. Compute gradient of loss w.r.t. input
3. Normalize gradient (sign for L∞, unit norm for Lp)
4. Add: x_adv = x + ε * normalized_gradient
```

### Projected Gradient Descent (PGD)

```python
from src.utils.evasion.minimal import ProjectedGradientDescent

pgd = ProjectedGradientDescent(
    norm="inf",           # L-infinity norm (also supports p-norms)
    criterion=CriterionConfig(name="nll", kwargs=None),
    num_steps=10,         # Number of iterations
    step_size=None,       # Auto: 2.5 * strength / num_steps
    random_start=True     # Start from random point in epsilon ball
)

adversarial = pgd.generate(
    born=bornmachine,
    naturals=data,        # Clean inputs
    labels=labels,
    strength=0.1,         # Perturbation magnitude (epsilon)
    device=device
)
```

**PGD Algorithm:**
```
1. Initialize perturbation δ (random if random_start, else zero)
2. For each step:
   a. Compute loss on x + δ
   b. Compute gradient of loss w.r.t. δ
   c. Update: δ = δ + step_size * normalized_gradient
   d. Project: δ = clip(δ, -ε, ε)  [for L∞]
3. Return x_adv = x + δ
```

### Robustness Evaluation

```python
from src.utils.evasion.minimal import RobustnessEvaluation

# Using FGM (single-step)
rob_eval = RobustnessEvaluation(
    method="FGM",
    norm="inf",
    criterion=CriterionConfig(name="nll", kwargs=None),
    strengths=[0.1, 0.3]
)

# Using PGD (iterative, stronger attack)
rob_eval = RobustnessEvaluation(
    method="PGD",
    norm="inf",
    criterion=CriterionConfig(name="nll", kwargs=None),
    strengths=[0.1, 0.3],
    num_steps=10,         # PGD iterations
    step_size=None,       # Auto-computed
    random_start=True
)

# Returns list of accuracies, one per strength
accuracies = rob_eval.evaluate(bornmachine, dataloader, device)
# e.g., [0.85, 0.72] for strengths [0.1, 0.3]
```

### Adding New Attack Methods

1. Create attack class with `generate()` method:
```python
class NewAttack:
    def __init__(self, norm, criterion):
        ...

    def generate(self, born, naturals, labels, strength, device):
        # Return adversarial examples
        return ad_examples
```

2. Register in `_METHOD_MAP`:
```python
_METHOD_MAP = {
    "FGM": FastGradientMethod,
    "PGD": ProjectedGradientDescent,
    "NewAttack": NewAttack,  # Add here
}
```

3. Update `EvasionConfig` in `schemas.py` if needed

## purification/minimal.py — Likelihood Purification

Implements likelihood-based purification of adversarial examples using the Born Machine's marginal probability p(x).

### LikelihoodPurification

```python
from src.utils.purification.minimal import LikelihoodPurification

purifier = LikelihoodPurification(
    norm="inf",           # L-infinity norm (also supports p-norms)
    num_steps=20,         # Gradient descent iterations
    step_size=None,       # Auto: 2.5 * radius / num_steps
    random_start=False,   # Start from original input
    eps=1e-12,            # Log clamping floor
)

purified, log_px = purifier.purify(
    born=bornmachine,
    data=adversarial_data,   # Inputs to purify
    radius=0.15,              # Maximum perturbation radius
    device=device
)
# purified: (batch, data_dim) — purified inputs
# log_px: (batch,) — log p(x) of purified inputs
```

**Purification Algorithm:**
```
1. Initialize perturbation δ (zero or random within Lp ball)
2. For each step:
   a. Compute NLL = -mean(log p(x + δ)) via born.marginal_log_probability()
   b. Compute gradient of NLL w.r.t. δ
   c. Update: δ = δ - step_size * normalized_gradient  (descent, not ascent!)
   d. Project: δ = clip(δ, -radius, radius)  [for L∞]
   e. Clamp: x + δ to input_range
3. Return purified = x + δ (clamped), log p(purified)
```

**Key difference from PGD:** PGD does gradient **ascent** on classification loss to attack; purification does gradient **descent** on NLL to defend (moves toward higher likelihood).

### PurificationConfig (schemas.py)

```python
from src.utils.schemas import PurificationConfig

cfg = PurificationConfig(
    norm="inf",
    num_steps=20,
    step_size=None,
    random_start=False,
    radii=[0.1, 0.2, 0.3],  # Multiple radii for evaluation
    eps=1e-12,
)
```

## Key Patterns

### Adding a New Embedding

```python
# In embeddings.py:

# 1. Implement embedding function
def my_embedding(data: torch.Tensor, degree: int, axis: int = -1):
    ...
    return embedded

# 2. Add to map
_EMBEDDING_MAP = {
    ...
    "myembed": my_embedding,
}

# 3. Add range
_EMBEDDING_TO_RANGE = {
    ...
    "myembed": (-1., 1.),
}
```

### Adding a New Loss Function

```python
# In criterions.py:

# 1. Implement loss class
class MyLoss(nn.Module):
    def __init__(self, **kwargs):
        ...
    def forward(self, p, t):
        ...

# 2. Add to appropriate loss map
_CLASSIFICATION_LOSSES = {
    ...
    "myloss": MyLoss,
}
# Or for generative losses:
_GENERATIVE_LOSSES = {
    ...
    "myloss": MyLoss,
}
```

## What Breaks If You Change...

| Change | Impact |
|--------|--------|
| Schema field name | Config loading fails, code using field breaks |
| Schema field type | Type errors, config validation fails |
| Embedding function signature | BornClassifier.embed() breaks |
| Loss function signature | Trainer loss computation breaks |
| `_EMBEDDING_TO_RANGE` | Data scaling becomes incorrect |

## Common Gotchas

1. **Config struct mode**: Hydra configs are "struct" by default, meaning you can't add fields dynamically. Use `OmegaConf.set_struct(cfg, False)` temporarily if needed.

2. **Loss expects probabilities**: `ClassificationNLL` expects already-normalized probabilities, not raw amplitudes.

3. **Embedding range matters**: The data must be scaled to match the embedding's expected range.

4. **Optional config fields**: Use `| None` type hints and handle None cases in code:
   ```python
   if cfg.trainer.classification is not None:
       ...
   ```

## File Reference

| File | Lines | Key Functions/Classes |
|------|-------|----------------------|
| `schemas.py` | ~434 | All `*Config` dataclasses |
| `get.py` | ~75 | `optimizer`, `indim_and_ncls` (re-exports from embeddings/criterions) |
| `embeddings.py` | ~175 | `embedding`, `range_from_embedding`, `legendre_embedding`, `hermite_embedding` |
| `criterions.py` | ~163 | `criterion`, `ClassificationNLL`, `GenerativeNLL` |
| `_utils.py` | ~100 | `set_seed`, `sample_quality_control` |
| `evasion/minimal.py` | ~291 | `FastGradientMethod`, `ProjectedGradientDescent`, `RobustnessEvaluation` |
| `purification/minimal.py` | ~130 | `LikelihoodPurification`, `normalizing` |
