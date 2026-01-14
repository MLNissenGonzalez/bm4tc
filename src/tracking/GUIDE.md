# Tracking Module Guide

This module handles experiment tracking, evaluation metrics, visualization, and Weights & Biases integration.

## Module Structure

```
src/tracking/
├── GUIDE.md              # This file
├── __init__.py           # Exports: init_wandb, log_grads, record, PerformanceEvaluator
├── evaluator.py          # Metric classes and PerformanceEvaluator
├── wandb_utils.py        # W&B initialization and logging utilities
├── visualisation.py      # Sample visualization (2D scatter plots)
└── fid_like.py           # FID-like metric for generative quality
```

## PerformanceEvaluator (`evaluator.py`)

Central class that orchestrates all evaluation metrics.

### Initialization

```python
from src.tracking import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    cfg=config,           # Full Config object
    datahandler=dh,       # DataHandler with loaded data
    train_cfg=train_cfg,  # Training config (determines which metrics)
    device=device
)
```

### Usage

```python
# Evaluate on validation set at epoch 10
results = evaluator.evaluate(bornmachine, "valid", step=10)
# results = {"loss": 0.15, "acc": 0.92, "rob/0.1": 0.85, "rob/0.3": 0.72}

# Evaluate on test set
results = evaluator.evaluate(bornmachine, "test", step=100)
```

### Metric Frequency

Metrics are evaluated based on their frequency setting in config:

```yaml
metrics: {"loss": 1, "acc": 1, "viz": 30, "fid": 30, "rob": 30}
```

- `loss` and `acc`: Every epoch
- `viz`, `fid`, `rob`: Every 30 epochs

The stopping criterion metric is always evaluated every epoch.

## Available Metrics

### LossMetric

Computes negative log-likelihood loss on class probabilities.

```python
loss = -mean(log(p[i, true_class[i]]))
```

### AccuracyMetric

Computes classification accuracy.

```python
accuracy = (predicted == labels).sum() / len(labels)
```

### FIDMetric (`fid_like.py`)

FID-like metric measuring distribution similarity between real and generated samples.

**Limitation**: Only computed when `data_dim < 100` (covariance matrix computation).

```python
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real·Σ_gen)^(1/2))
```

### RobustnessMetric

Evaluates accuracy under adversarial attacks at multiple perturbation strengths.

```python
# Returns dict with keys like "rob/0.1", "rob/0.3"
results = {"rob/0.1": 0.85, "rob/0.3": 0.72}
```

Uses `RobustnessEvaluation` from `src/utils/evasion/minimal.py`.

### VisualizationMetric

Generates 2D scatter plots of sampled data.

```python
# Returns matplotlib axis object
ax = visualise_samples(samples)
```

**Limitation**: Only works for 2D data.

## Adding a New Metric

1. **Create metric class** (`evaluator.py`):
```python
class MyMetric(BaseMetric):
    def __init__(self, freq, cfg, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        # Initialize any resources

    def evaluate(self, bornmachine, split, context):
        # Use self._labels_n_probs() for classifier metrics
        # Use self._generate() for generator metrics

        # Compute and return scalar value
        return value
```

2. **Register in MetricFactory** (`evaluator.py:182-197`):
```python
class MetricFactory:
    @staticmethod
    def create(...):
        mapping = {
            ...
            "mymetric": MyMetric,  # Add here
        }
```

3. **Update config** to include the metric:
```yaml
metrics: {"loss": 1, "acc": 1, "mymetric": 10}
```

## W&B Integration (`wandb_utils.py`)

### Initialization

```python
from src.tracking import init_wandb

run = init_wandb(cfg)
# Creates W&B run with:
# - Project: cfg.tracking.project
# - Entity: cfg.tracking.entity
# - Group: "{experiment}_{dataset}_{date}"
# - Name: "job{n}/{total}_D{bond_dim}-d{in_dim}pre{epochs}..."
```

### Logging

```python
from src.tracking import record, log_grads

# Log evaluation metrics
record(
    results={"acc": 0.95, "loss": 0.1},
    stage="pre",     # "pre", "gan", "adv"
    set="valid"      # "train", "valid", "test"
)
# Logs as: "pre/valid/acc", "pre/valid/loss"

# Log gradient histograms (during training)
log_grads(
    bm_view=bornmachine.classifier,
    step=step,
    watch_freq=100,
    stage="pre"
)
```

### Visualization Logging

```python
# VisualizationMetric returns axis, record() handles conversion:
if metric_name == "viz":
    fig = result.figure if hasattr(result, "figure") else result
    upload_dict["samples/pre"] = wandb.Image(fig)
```

## Context Caching

The evaluator uses a `context` dict to cache expensive computations:

```python
def _labels_n_probs(self, bornmachine, split, context):
    """Cache class probabilities for reuse across metrics."""
    if "labels_n_probs" not in context:
        context["labels_n_probs"] = []
        for data, labels in loader:
            probs = bornmachine.class_probabilities(data)
            context["labels_n_probs"].append((labels, probs))

def _generate(self, bornmachine, context):
    """Cache generated samples for reuse across metrics."""
    if "synths" not in context:
        context["synths"] = bornmachine.sample(cfg=self.cfg.tracking.sampling)
```

This prevents redundant forward passes when evaluating multiple metrics.

## Visualization (`visualisation.py`)

### 2D Scatter Plot

```python
from src.tracking.visualisation import visualise_samples, create_2d_scatter

# For generated samples: (num_spc, num_classes, data_dim)
ax = visualise_samples(samples)

# For labeled data: (N, data_dim) with labels (N,)
ax = visualise_samples(samples, labels=labels)

# Direct scatter plot
ax = create_2d_scatter(X, t, title="My Plot")
```

### Limitations

- Only 2D data supported
- Higher dimensions raise `ValueError`

## FID-like Metric (`fid_like.py`)

### Matrix Square Root

Uses custom autograd function for differentiable matrix square root:

```python
class MatrixSquareRoot(Function):
    @staticmethod
    def forward(ctx, input):
        # Uses scipy.linalg.sqrtm
        ...

    @staticmethod
    def backward(ctx, grad_output):
        # Solves Sylvester equation
        ...

sqrtm = MatrixSquareRoot.apply
```

### FIDLike Class

```python
class FIDLike(nn.Module):
    def forward(self, real, generated):
        # Compute means and covariances
        # Return FID-like distance
        return diff_mu + diff_cov
```

### FIDEvaluation Wrapper

```python
class FIDEvaluation:
    def __init__(self, cfg, datahandler, device):
        # Only enabled for data_dim < 100
        if datahandler.data_dim < 1e2:
            self.toEval = True
            self.stat_r = datahandler.means, datahandler.covs
        else:
            self.toEval = False

    def evaluate(self, synths):
        # Compute per-class FID and average
        ...
```

## W&B Metric Hierarchy

```
{stage}/{split}/{metric}
├── pre/
│   ├── train/
│   │   └── loss
│   ├── valid/
│   │   ├── loss
│   │   ├── acc
│   │   ├── fid
│   │   └── rob/{strength}
│   └── test/
│       └── ...
├── gan/
│   ├── train/
│   │   ├── g_loss
│   │   └── d_loss
│   ├── dis_train/
│   │   └── loss
│   └── ...
└── samples/
    └── {stage}  # Images
```

## What Breaks If You Change...

| Change | Impact |
|--------|--------|
| Metric return type | W&B logging may fail |
| Metric frequency | Evaluation schedule changes |
| Context keys | Cache invalidation issues |
| W&B metric names | Dashboard organization breaks |
| FID threshold (100) | Large datasets may OOM |

## Debugging Tips

1. **Check metric values**:
   ```python
   for name, result in results.items():
       print(f"{name}: {result}")
   ```

2. **Verify context caching**:
   ```python
   print(f"labels_n_probs cached: {'labels_n_probs' in context}")
   print(f"synths cached: {'synths' in context}")
   ```

3. **W&B offline mode**:
   ```yaml
   tracking:
     mode: disabled  # No network calls
   ```

4. **Check W&B run**:
   ```python
   print(f"W&B run: {wandb.run.name}")
   print(f"W&B URL: {wandb.run.url}")
   ```

## File Reference

| File | Lines | Key Classes/Functions |
|------|-------|----------------------|
| `evaluator.py` | 261 | `BaseMetric`, `*Metric`, `PerformanceEvaluator` |
| `wandb_utils.py` | 185 | `init_wandb`, `record`, `log_grads` |
| `visualisation.py` | 69 | `visualise_samples`, `create_2d_scatter` |
| `fid_like.py` | 181 | `MatrixSquareRoot`, `FIDLike`, `FIDEvaluation` |
