# Models Module Guide

This module contains the core model implementations: the Born Machine (MPS-based classifier/generator) and the Critic (discriminator for GAN-style training).

## Module Structure

```
src/models/
├── GUIDE.md              # This file
├── __init__.py           # Exports: BornMachine, BornClassifier, BornGenerator, Critic
├── born.py               # BornMachine: unified view managing classifier + generator
├── classifier.py         # BornClassifier: MPS for classification (extends tk.models.MPSLayer)
├── generator/
│   ├── __init__.py
│   ├── generator.py      # BornGenerator: MPS for sampling (extends tk.models.MPS)
│   └── differential_sampling.py  # Differentiable sampling algorithms
└── discriminator/
    ├── __init__.py
    ├── discriminator.py  # Critic: discriminator for GAN training
    ├── backbones.py      # Feature extractors (MLP, etc.)
    └── heads.py          # Classification heads (aware/agnostic)
```

## Key Classes

### BornMachine (`born.py`)

The central class that manages the entire Born Machine. It acts as a **unified view** over two synchronized models:

1. **BornClassifier** — For computing p(c|x) (classification)
2. **BornGenerator** — For sampling x ~ p(x|c) (generation)

**Critical Design Point**: Both share the same underlying tensors! Changes to one affect the other.

```python
class BornMachine:
    """
    Key Attributes:
    - classifier: BornClassifier instance
    - generator: BornGenerator instance
    - embedding: function to embed inputs
    - cfg: BornMachineConfig

    Key Methods:
    - class_probabilities(data) -> p(c|x)
    - sample(cfg, cls) -> samples from p(x|c)
    - parameters(mode) -> parameters for optimizer
    - sync_tensors(after) -> sync classifier/generator after training
    - save/load -> checkpoint management
    """
```

**Tensor Synchronization** (`born.py:126`):
```python
def sync_tensors(self, after: str, verify: bool=False):
    """
    CRITICAL: Call this after training to keep classifier and generator in sync!

    after="classification" -> copy from classifier to generator
    after="gan" -> copy from generator to classifier
    """
```

**Uncertainty Quantification Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `cache_log_Z()` | — | `float` | Compute and cache log partition function |
| `marginal_log_probability(data, eps)` | `(batch, D)` | `(batch,)` | Log marginal p(x) = log(sum_c |psi(x,c)|^2) - log(Z) |

**How `marginal_log_probability` works:**
```python
embs = self.classifier.embed(data)             # (batch, D, in_dim)
amplitudes = self.classifier.forward(data=embs) # (batch, num_classes) — single contraction
unnorm_px = torch.square(amplitudes).sum(dim=-1)  # (batch,) — sum_c |psi(x,c)|^2
log_px = log(unnorm_px) - log_Z                   # normalized log p(x)
```

**Differentiability:** Differentiable w.r.t. input data (for purification) but NOT w.r.t. model parameters (`_log_Z` is a detached constant). This is correct for purification (which optimizes over inputs, not model params).

**log Z caching:** `_log_Z` is computed once via `generator.log_partition_function()` and stored as a Python float. It is persisted in checkpoints via `save()`/`load()`. Call `cache_log_Z()` to recompute after model parameter changes.

### BornClassifier (`classifier.py`)

Extends `tensorkrowch.models.MPSLayer` for classification using the Born rule.

```
Input: x ∈ R^D (data_dim features)
  ↓
Embedding: φ(x) ∈ R^(D × in_dim)
  ↓
MPS Contraction: contract with MPS tensors
  ↓
Output: amplitudes ∈ R^num_classes
  ↓
Born Rule: |amplitude|² / Σ|amplitudes|² = p(c|x)
```

**Key Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `embed(data)` | `(batch, D)` | `(batch, D, in_dim)` | Embed raw features |
| `forward(data)` | `(batch, D, in_dim)` | `(batch, num_cls)` | **Amplitudes** (not probabilities!) |
| `parallel(embs)` | `(batch, D, in_dim)` | `(batch, num_cls)` | Probabilities (squared & normalized) |
| `probabilities(data)` | `(batch, D)` | `(batch, num_cls)` | End-to-end: data → p(c|x) |
| `prepare(...)` | — | — | Reset and prepare for training |

**CRITICAL WARNING** (`classifier.py:112`):
```python
# The forward() method returns AMPLITUDES, not probabilities!
# To get probabilities, use:
p = torch.square(self.forward(data=embs))
return p / p.sum(dim=-1, keepdim=True)
```

### BornGenerator (`generator/generator.py`)

**The BornGenerator is equally critical as BornClassifier** — it enables sampling from the learned distribution p(x|c), which is essential for:
- **GAN-style training**: Generating synthetic samples to train against the critic
- **Evaluating generative quality**: FID metric, visualizations
- **The core research hypothesis**: Generative understanding → robustness

Extends `tensorkrowch.models.MPS` for sampling using **sequential contraction** (different from the parallel contraction used in classification).

**Why Sequential Contraction?**

The classifier uses parallel contraction because all features are known. The generator must sample features one-by-one, computing marginal distributions over unobserved variables:

```
Sequential Sampling (generator):           Parallel Contraction (classifier):
─────────────────────────────────          ─────────────────────────────────
For i = 0 to D-1:                          Contract all features at once:
  p(xᵢ | x₀,...,xᵢ₋₁, c) = ?               ψ(x₀, x₁, ..., xD) = scalar
  ↓ marginalize over xᵢ₊₁,...,xD
  sample xᵢ
  condition on sampled xᵢ
```

**Sampling Process** (`_single_class`, line 143):
```
For each feature site i (in order):
  1. Build partial embedding dict: {cls_pos: cls_emb, 0: x₀_emb, ..., i-1: xᵢ₋₁_emb, i: grid_emb}
  2. Call sequential() with marginalize_output=True → density matrix
  3. Extract diagonal → p(xᵢ | previous samples, class)
  4. Differentiable sampling → sample xᵢ value
  5. Embed sampled xᵢ and add to conditioning set
```

**The `sequential()` Method** (line 79):

This is the workhorse that handles both full contraction and marginalization:

```python
def sequential(self, embs: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    embs: {site_index: embedded_tensor} — partial input

    If len(embs) < n_features:
        → marginalize over missing sites
        → return diagonal of density matrix (marginal probabilities)
    Else:
        → full contraction
        → return |amplitude|²
    """
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `sequential(embs)` | Sequential MPS contraction (supports marginalization) |
| `_single_class(...)` | Core sampling loop for one class |
| `sample_single_class(cls, cfg)` | API: sample from specific class (handles batching) |
| `sample_all_classes(cfg)` | API: sample from all classes |
| `prepare()` | Reset MPS state before sampling |

**Output Shapes:**
- `sample_single_class(cls, cfg)` → `(num_spc, data_dim)`
- `sample_all_classes(cfg)` → `(num_spc, num_classes, data_dim)`

**Memory Management:**

Sampling is memory-intensive (scales with `num_spc × num_bins × in_dim`). Use `batch_spc` in `SamplingConfig` to control memory:

```python
sampling_cfg = SamplingConfig(
    num_spc=1000,    # Total samples wanted
    batch_spc=64,    # Generate in batches of 64
    num_bins=200,    # Resolution of probability grid
    method="secant"
)
```

### Differentiable Sampling (`generator/differential_sampling.py`)

Implements sampling methods that allow gradient flow for GAN training.

**Current Implementation: Secant Method** (`os_secant`)

```
Given: p(xᵢ) as PMF over bins, grid z of bin centers
1. Compute CDF from PMF
2. Sample u ~ Uniform(0,1)
3. Find bin j where CDF(j-1) < u ≤ CDF(j)
4. Linear interpolation within bin (provides gradient)
```

The secant interpolation step (`differential_sampling.py:77`):
```python
samples = a + (b - a) * (nu - cdf_a) / (cdf_b - cdf_a)
```
This is differentiable w.r.t. the CDF values, enabling backprop through sampling.

### Critic (`discriminator/discriminator.py`)

Discriminator for GAN-style training. Distinguishes real from generated samples.

**Architecture:**
```
Input x ∈ R^D
    ↓
BackBone (MLP) → features ∈ R^F
    ↓
Head → logit(s)
    ↓
Loss (BCE or WGAN-GP)
```

**Two Head Types:**
1. **Class-Aware** (`AwareHead`): One output per class, for per-class discrimination
2. **Class-Agnostic** (`AgnosticHead`): Single output, for overall real/fake

**Loss Functions:**

| Loss | Critic | Generator |
|------|--------|-----------|
| BCE | `-log(σ(D(real))) - log(1-σ(D(fake)))` | `-log(σ(D(fake)))` (swapped) |
| WGAN-GP | `D(fake) - D(real) + λ·GP` | `-D(fake)` |

**Training Methods:**
- `ganstyle_pretrain(...)` — Pre-train critic before GAN loop
- `train_step(...)` — Single gradient step on critic
- `discriminate(...)` — Run discrimination over a split

## Tensor Network Details

### MPS Structure

An MPS with n features has n+1 tensors (including output tensor):

```
T[0] -- T[1] -- T[2] -- ... -- T[out_position] -- ... -- T[n]
 |       |       |                   |                    |
 x₀      x₁      x₂                 cls                   xₙ
```

Each tensor T[i] has shape:
- `(bond_left, phys_dim, bond_right)` for internal tensors
- `(phys_dim, bond_right)` for left boundary (OBC)
- `(bond_left, phys_dim)` for right boundary (OBC)
- Output tensor has additional `out_dim` axis

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_features` | Number of MPS sites | data_dim + 1 |
| `in_dim` | Physical dimension (embedding size) | 4, 10, 30 |
| `out_dim` | Number of classes | 2 (for toy data) |
| `bond_dim` | Bond dimension (expressivity) | 3, 4, 6, 10, 18 |
| `out_position` | Position of output tensor | middle by default |
| `boundary` | Boundary conditions | "obc" (open) |

### tensorkrowch Integration

The codebase uses `tensorkrowch` for tensor network operations:

```python
# BornClassifier inherits from:
class BornClassifier(tk.models.MPSLayer):
    ...

# BornGenerator inherits from:
class BornGenerator(tk.models.MPS):
    ...
```

**Important tensorkrowch methods used:**
- `self.tensors` — List of underlying tensor parameters
- `self.forward(data)` — Contract MPS with input
- `self.reset()` — Reset computation graph
- `self.unset_data_nodes()` — Clear cached data nodes
- `self.trace(...)` — Trace computation for optimization
- `self.initialize(tensors)` — Set tensor values

## Common Operations

### Creating a BornMachine

```python
from src.models import BornMachine
from src.utils.schemas import BornMachineConfig

# From config
bm = BornMachine(cfg=config, data_dim=2, num_classes=2, device=device)

# From checkpoint
bm = BornMachine.load("path/to/checkpoint")
```

### Classification

```python
# data: (batch_size, data_dim)
probs = bm.class_probabilities(data)  # (batch_size, num_classes)
predictions = probs.argmax(dim=1)      # (batch_size,)
```

### Sampling

```python
from src.utils.schemas import SamplingConfig

sampling_cfg = SamplingConfig(
    method="secant",
    num_spc=100,      # samples per class
    num_bins=200,     # resolution
    batch_spc=32      # batch size for memory
)

# Sample from all classes
samples = bm.sample(sampling_cfg)  # (100, num_classes, data_dim)

# Sample from specific class
samples = bm.sample(sampling_cfg, cls=0)  # (100, data_dim)
```

### Training Preparation

```python
# Before classification training:
bm.classifier.prepare(device=device, train_cfg=train_cfg)

# After classification training:
bm.sync_tensors(after="classification", verify=True)

# Before GAN training:
bm.generator.prepare()

# After GAN training:
bm.sync_tensors(after="gan", verify=True)
```

## What Breaks If You Change...

| Change | Impact |
|--------|--------|
| `BornClassifier.forward()` | Breaks training, evaluation, sampling (called everywhere) |
| `BornGenerator.sequential()` | Breaks all sampling |
| `sync_tensors()` | Classifier/generator become inconsistent |
| Embedding function | Must update `input_range` and data scaling |
| `out_position` | Changes MPS structure, breaks saved models |
| `in_dim` / `bond_dim` | Changes tensor shapes, breaks saved models |

## Debugging Tips

1. **Verify sync**: After training, check `bm.classifier.tensors[0]` matches `bm.generator.tensors[0]`

2. **Check probabilities sum to 1**:
   ```python
   probs = bm.class_probabilities(data)
   assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))
   ```

3. **Sampling produces valid range**:
   ```python
   samples = bm.sample(cfg)
   assert (samples >= 0).all() and (samples <= 1).all()  # for Fourier embedding
   ```

4. **Gradient flow in sampling**:
   ```python
   samples = bm.sample(cfg)
   loss = some_loss(samples)
   loss.backward()  # Should not error
   assert bm.generator.tensors[0].grad is not None
   ```

## Debugging: Uncertainty Quantification

### Verify marginal_log_probability
```python
# 1. Verify marginal_log_probability produces valid values
bm = BornMachine.load("path/to/checkpoint.pt")
bm.to(device)
bm.cache_log_Z()
print(f"log Z = {bm._log_Z}")  # Should be a finite positive number

# Load test data
data = datahandler.data["test"].to(device)
log_px = bm.marginal_log_probability(data)
print(f"log p(x) range: [{log_px.min():.2f}, {log_px.max():.2f}]")
assert torch.isfinite(log_px).all(), "Non-finite log probabilities detected"
assert (log_px <= 0).all(), "Log probabilities should be <= 0 (probabilities <= 1)"

# 2. Verify partition function consistency
# On a fine grid over input_range, sum of p(x) * dx should approximate 1.0
grid = torch.linspace(*bm.input_range, 200, device=device)
grid_2d = torch.stack(torch.meshgrid(grid, grid, indexing='ij'), dim=-1).reshape(-1, 2)
log_px_grid = bm.marginal_log_probability(grid_2d)
dx = (bm.input_range[1] - bm.input_range[0]) / 200
integral = torch.exp(log_px_grid).sum() * dx**2
print(f"Integral of p(x) over grid: {integral:.4f}")  # Should be close to 1.0

# 3. Verify gradient flow for purification
data_sample = data[:5].clone().requires_grad_(True)
log_px = bm.marginal_log_probability(data_sample)
(-log_px.sum()).backward()
assert data_sample.grad is not None, "No gradient on input data"
print(f"Gradient norm: {data_sample.grad.norm():.4e}")  # Should be non-zero

# 4. Verify purification moves points toward higher likelihood
from src.utils.purification.minimal import LikelihoodPurification
purifier = LikelihoodPurification(norm="inf", num_steps=20)
# Add noise to create low-likelihood inputs
noisy = data[:10] + 0.1 * torch.randn_like(data[:10])
noisy = noisy.clamp(*bm.input_range)
log_px_before = bm.marginal_log_probability(noisy).detach()
purified, log_px_after = purifier.purify(bm, noisy, radius=0.15, device=device)
print(f"Mean log p(x) before: {log_px_before.mean():.2f}")
print(f"Mean log p(x) after:  {log_px_after.mean():.2f}")
assert (log_px_after >= log_px_before - 1e-6).all(), "Purification should not decrease likelihood"
```

## File-by-File Reference

| File | Lines | Key Classes/Functions |
|------|-------|----------------------|
| `born.py` | ~200 | `BornMachine` |
| `classifier.py` | ~132 | `BornClassifier` |
| `generator/generator.py` | ~475 | `BornGenerator` |
| `generator/differential_sampling.py` | ~132 | `os_secant`, `main` |
| `discriminator/discriminator.py` | ~281 | `Critic` |
| `discriminator/backbones.py` | ~102 | `BackBone`, `MLP`, `get_backbone` |
| `discriminator/heads.py` | ~164 | `AwareHead`, `AgnosticHead`, `get_head` |
