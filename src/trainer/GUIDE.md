# Trainer Module Guide

This module contains the training loops for different training paradigms: classification (discriminative), GAN-style (generative), generative NLL, and adversarial (robust).

## Module Structure

```
src/trainer/
├── GUIDE.md              # This file
├── __init__.py           # Exports: ClassificationTrainer, GANStyleTrainer, GenerativeTrainer, AdversarialTrainer
├── classification.py     # Discriminative classifier training
├── ganstyle.py          # GAN-style generative training
├── generative.py        # Generative NLL training
└── adversarial.py       # Adversarial robustness training (stub)
```

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Classification Training (ClassificationTrainer)          │
│     ┌──────────────────────────────────────────────┐        │
│     │ For each epoch:                               │        │
│     │   - Forward: data → embed → MPS → amplitudes │        │
│     │   - Loss: NLL on squared amplitudes          │        │
│     │   - Backward & optimizer step                │        │
│     │   - Evaluate on validation set               │        │
│     │   - Early stopping check                     │        │
│     └──────────────────────────────────────────────┘        │
│                          ↓                                   │
│  2. GAN-style Training (GANStyleTrainer) [optional]         │
│     ┌──────────────────────────────────────────────┐        │
│     │ For each epoch:                               │        │
│     │   - Sample from generator                    │        │
│     │   - Train critic (inner loop)                │        │
│     │   - Train generator (single step)            │        │
│     │   - Check if retraining needed               │        │
│     │   - If yes: run ClassificationTrainer        │        │
│     └──────────────────────────────────────────────┘        │
│                          ↓                                   │
│  3. Adversarial Training (AdversarialTrainer) [planned]     │
│     ┌──────────────────────────────────────────────┐        │
│     │ Not yet implemented                          │        │
│     └──────────────────────────────────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## ClassificationTrainer (`classification.py`)

Trains the BornMachine as a discriminative classifier.

### Initialization

```python
trainer = ClassificationTrainer(
    bornmachine=bm,      # BornMachine instance
    cfg=cfg,             # Full Config object
    stage="pre",         # "pre" for initial, "re" for retraining
    datahandler=dh,      # DataHandler with loaded data
    device=device
)
```

**Stage Options:**
- `"pre"` — Initial classification training (uses `cfg.trainer.classification`)
- `"re"` — Retraining during GAN loop (uses `cfg.trainer.ganstyle.retrain`)

### Training Flow

```python
def train(self, goal: Dict[str, float] | None = None):
    """
    goal: Optional early stopping target, e.g., {"acc": 0.95}
    """
```

**Flow (`classification.py:183-210`):**
1. Prepare classifier (`bm.classifier.prepare()`)
2. Initialize criterion (NLL) and optimizer
3. For each epoch:
   - `_train_epoch()` — One epoch of gradient updates
   - `sync_tensors()` — Sync for evaluation metrics
   - `evaluator.evaluate()` — Compute metrics
   - `_update()` — Check for improvement, early stopping
4. `_summarise_training()` — Restore best weights, test eval, save

### Key Methods

| Method | Description |
|--------|-------------|
| `_train_epoch()` | One epoch: forward → loss → backward → step |
| `_update()` | Check improvement, update best tensors, patience |
| `_summarise_training()` | Restore best model, test eval, save checkpoint |
| `_best_perf_factory()` | Initialize tracking for best metric values |

### Training Loop Detail (`classification.py:65-84`)

```python
def _train_epoch(self):
    for data, labels in self.datahandler.classification["train"]:
        # Forward pass
        probs = self.bornmachine.class_probabilities(data)
        loss = self.criterion(probs, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        log_grads(...)  # Log gradients to W&B
        self.optimizer.step()
```

### Early Stopping (`classification.py:86-126`)

Monitors a metric on validation set:
- **Metric options**: `"acc"`, `"loss"`, `"rob"`, `"fid"`
- **Patience**: Number of epochs without improvement before stopping
- **Goal**: Optional target value to reach early

## GANStyleTrainer (`ganstyle.py`)

Improves generative capabilities using adversarial training with a critic.

### Initialization

```python
trainer = GANStyleTrainer(
    bornmachine=bm,
    cfg=cfg,
    datahandler=dh,
    critic=critic,       # Critic instance
    device=device,
    best={"acc": 0.95}   # Goal from pre-training
)
```

### Training Flow (`ganstyle.py:101-161`)

```
For each epoch:
  For each batch of natural data:
    1. Sample from generator (no grad)
    2. Inner loop: Train critic for max_epoch_gan steps
    3. Sample from generator (with grad)
    4. Compute generator loss via critic
    5. Backward & optimizer step

  After epoch:
    - Sync tensors
    - Evaluate on validation
    - If accuracy dropped too much: retrain classifier
```

### Retraining Mechanism

If classification accuracy drops below tolerance:
```python
def _toRetrain(self, validation_metrics):
    toRetrain = (
        (self.goal["acc"] - validation_metrics["acc"]) > self.train_cfg.tolerance
    )
```

This triggers a `ClassificationTrainer` run to recover discriminative performance.

### Important Issue

**Bug in imports (`ganstyle.py:9-10`):**
```python
# WRONG:
from data.handler import DataHandler
from models import Critic, BornMachine

# SHOULD BE:
from src.data.handler import DataHandler
from src.models import Critic, BornMachine
```

## GenerativeTrainer (`generative.py`)

Trains the BornMachine generator using NLL minimization on p(x|c).

### Initialization

```python
from src.utils.generative_losses import GenerativeNLL

# User must implement their own criterion subclass
class MyGenerativeNLL(GenerativeNLL):
    def compute_unnormalized_log_prob(self, bornmachine, data, labels):
        # Implement: log(|psi(x,c)|^2)
        pass

    def compute_log_partition(self, bornmachine, labels):
        # Implement: log(Z_c) normalization
        pass

criterion = MyGenerativeNLL()

trainer = GenerativeTrainer(
    bornmachine=bm,
    cfg=cfg,
    datahandler=dh,
    criterion=criterion,  # User-implemented criterion
    device=device
)
```

### Training Flow

```python
def train(self, goal: Dict[str, float] | None = None):
    """
    goal: Optional early stopping target, e.g., {"fid": 10.0}
    """
```

**Flow:**
1. Prepare generator and optimizer
2. For each epoch:
   - `_train_epoch()` — Forward through criterion, backward, step
   - `sync_tensors(after="generation")` — Sync to classifier view
   - `evaluator.evaluate()` — Compute metrics (fid, viz, etc.)
   - `_update()` — Check for improvement, early stopping
3. `_summarise_training()` — Restore best weights, test eval, save

### Key Difference from ClassificationTrainer

The criterion takes the full BornMachine (not just probabilities):
```python
loss = self.criterion(self.bornmachine, data, labels)
```

This allows the criterion to access the generator's internal state for computing
unnormalized probabilities and partition functions.

## AdversarialTrainer (`adversarial.py`)

**Status: Stub / Not Implemented**

```python
class Trainer:
    def __init__(self):
        pass

    def train(bornmachine, cfg):
        return NotImplementedError
```

**Planned functionality:**
- Train classifier for robustness against adversarial examples
- Use attacks from `src/utils/evasion/` during training
- Alternate between normal and adversarial examples

## Configuration

### ClassificationConfig (`schemas.py:208-220`)

```yaml
max_epoch: 300
batch_size: 64
criterion:
  name: "negative log-likelihood"
  kwargs: {eps: 1e-8}
optimizer:
  name: "adam"
  kwargs: {lr: 1e-4, weight_decay: 0.0}
patience: 40
stop_crit: "loss"          # Metric to monitor
watch_freq: 1000           # Gradient logging frequency
metrics: {loss: 1, acc: 1, viz: 30, fid: 30, rob: 30}
save: true
auto_stack: true           # tensorkrowch option
auto_unbind: false         # tensorkrowch option
```

### GANStyleConfig (`schemas.py:225-270`)

```yaml
max_epoch: 100
critic: {...}              # CriticConfig
sampling: {...}            # SamplingConfig
r_real: 1.0               # Ratio of real to synthetic samples
optimizer: {...}
watch_freq: 1
metrics: {loss: 1, acc: 1, rob: 1}
retrain_crit: "acc"        # When to retrain
tolerance: 0.05            # Accuracy drop tolerance
retrain: {...}             # ClassificationConfig for retraining
save: false
```

### GenerativeConfig (`schemas.py:289-307`)

```yaml
max_epoch: 100
batch_size: 64
criterion:
  name: "generative-nll"   # User implements GenerativeNLL subclass
  kwargs: {eps: 1e-8}
optimizer:
  name: "adam"
  kwargs: {lr: 1e-4, weight_decay: 0.0}
patience: 50
stop_crit: "loss"          # "loss" or "fid"
watch_freq: 100
metrics: {loss: 1, fid: 10, viz: 10}
save: false
auto_stack: true
auto_unbind: false
```

## Integration Points

### With DataHandler

```python
# Classification training uses:
datahandler.classification["train"]  # DataLoader for training
datahandler.classification["valid"]  # DataLoader for validation

# GAN training uses:
datahandler.discrimination["train"]  # DataLoader of (num_spc, num_cls, data_dim)
```

### With PerformanceEvaluator

```python
from src.tracking import PerformanceEvaluator

evaluator = PerformanceEvaluator(cfg, datahandler, train_cfg, device)
results = evaluator.evaluate(bornmachine, "valid", epoch)
# results = {"acc": 0.95, "loss": 0.1, "rob/0.1": 0.8, ...}
```

### With W&B

```python
import wandb

# Define metrics
wandb.define_metric("pre/train/loss", summary="none")

# Log metrics
wandb.log({"pre/train/loss": loss, "pre/valid/acc": acc})

# Summary at end
wandb.summary["pre/test/acc"] = test_acc
```

### Epoch Timing Metrics

Both trainers automatically record average epoch times to W&B summary:

**ClassificationTrainer:**
```python
wandb.summary[f"{stage}/avg_epoch_time_s"]  # e.g., "pre/avg_epoch_time_s"
```

**GANStyleTrainer:**
```python
wandb.summary["gan/avg_epoch_time_total_s"]      # Including retraining epochs
wandb.summary["gan/avg_epoch_time_no_retrain_s"] # Excluding retraining epochs
```

These metrics help track training performance across different configurations.

## What Breaks If You Change...

| Change | Impact |
|--------|--------|
| `_train_epoch` loop | Core training breaks |
| Criterion interface | Must update loss computation |
| `sync_tensors` call location | Classifier/generator inconsistency |
| Evaluation metric names | W&B logging breaks |
| `_update` logic | Early stopping may malfunction |
| Retraining trigger | GAN loop quality affected |

## Debugging Tips

1. **Check loss decreasing**:
   ```python
   # Watch W&B or print:
   print(f"Epoch {epoch}: loss={loss:.4f}")
   ```

2. **Verify gradient flow**:
   ```python
   for name, param in bm.classifier.named_parameters():
       if param.grad is None:
           print(f"No gradient for {name}")
   ```

3. **Check tensor sync**:
   ```python
   # After sync_tensors:
   for i, (t1, t2) in enumerate(zip(
       bm.classifier.tensors, bm.generator.tensors)):
       assert torch.allclose(t1, t2), f"Tensor {i} mismatch"
   ```

4. **Monitor patience counter**:
   ```python
   print(f"Patience: {self.patience_counter}/{self.train_cfg.patience}")
   ```

## Example Usage

### Classification Only

```python
from src.trainer import ClassificationTrainer

trainer = ClassificationTrainer(bm, cfg, "pre", datahandler, device)
trainer.train()
# Model saved to outputs/<exp>/models/
```

### Classification + GAN

```python
from src.trainer import ClassificationTrainer, GANStyleTrainer
from src.models import Critic

# Phase 1: Classification
pre_trainer = ClassificationTrainer(bm, cfg, "pre", datahandler, device)
pre_trainer.train()

# Phase 2: GAN-style
critic = Critic(cfg.trainer.ganstyle, datahandler, device=device)
gan_trainer = GANStyleTrainer(bm, cfg, datahandler, critic, device, pre_trainer.best)
gan_trainer.train()
```
