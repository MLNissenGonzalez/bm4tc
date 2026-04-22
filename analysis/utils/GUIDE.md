# Analysis Utils — Evaluation Concepts and Implementation

This guide explains **what is being computed** in the analysis pipeline and **how it works**, covering the math, implementation choices, and important caveats.

---

## Module map

| File | Purpose |
|------|---------|
| `evaluate.py` | Top-level `evaluate_run()` / `evaluate_sweep()` — orchestrates all metrics |
| `uq.py` | Uncertainty quantification: detection + likelihood purification |
| `mia.py` | Membership inference attack evaluation |
| `mia_utils.py` | Config loading (`load_run_config`) and checkpoint finding |
| `wandb_fetcher.py` | W&B API + local summary loading |
| `resolve.py` | Path-to-regime/embedding detection, range-size table, param shorthands |
| `statistics.py` | Summary tables, Pareto frontiers, correlation heatmaps |

---

## Input range and epsilon convention

Every embedding has a domain — the range of input values it was designed for:

| Embedding | Domain | Range size |
|-----------|--------|-----------|
| `fourier` | (0, 1) | 1.0 |
| `legendre` | (-1, 1) | 2.0 |
| `hermite` | (-4, 4) | 8.0 |
| `chebychev1` | (-0.99, 0.99) | 1.98 |
| `chebychev2` | (-1, 1) | 2.0 |

The `DataHandler` fits a MinMax scaler on training data and rescales all splits into the embedding's domain. This is how `input_range` ends up on the model:

```python
# DataHandler.split_and_rescale:
self.input_range = bornmachine.input_range   # from get.range_from_embedding(cfg.embedding)
self.scaler = MinMaxScaler(feature_range=self.input_range, clip=True)
```

**Attack epsilons must be expressed in these units.** A perturbation of epsilon=0.1 L∞ means each feature can shift by at most 0.1 in the rescaled space. For Fourier that is 10% of the range; for Hermite the same 0.1 is only 1.25% of the range — 8× weaker in relative terms.

`seed_sweep_analysis.py` therefore uses **range-relative fractions** multiplied by `_RANGE_SIZE` (from `analysis/utils/resolve.py`):

```python
_STRENGTH_FRACTIONS = [0.05, 0.10, 0.2, 0.5, 0.8]
strengths = [f * _RANGE_SIZE for f in _STRENGTH_FRACTIONS]
# fourier:  [0.05, 0.10, 0.20, 0.50, 0.80]
# hermite:  [0.40, 0.80, 1.60, 4.00, 6.40]
```

---

## Adversarial attacks

### Fast Gradient Method (FGM)

Single-step attack. Given a loss `L(x, y)`:

```
x_adv = x + ε · sign(∇_x L(x, y))      [L∞ norm]
x_adv = x + ε · ∇_x L(x,y) / ‖∇_x L‖_2  [L2 norm]
```

Fast but weak — underestimates robustness for multi-step capable models.

### Projected Gradient Descent (PGD)

Iterative attack (Madry et al., 2018). Start from `x_0 = x + δ_0` where δ_0 is uniform in the ε-ball (random_start=True), then iterate:

```
δ_{t+1} = Π_{‖δ‖≤ε}[ δ_t + α · sign(∇_δ L(x + δ_t, y)) ]   [L∞]
```

where `α = 2.5ε / T` by default (T = num_steps). The projection Π keeps δ inside the ε-ball.

**Implementation note**: The current code (`evasion/minimal.py`) does **not clamp** `x + δ` to the embedding's input domain. This means adversarial examples can stray slightly outside the valid range at the boundaries. For Hermite the Gaussian damping `exp(-x²/2)` suppresses out-of-range embedding components naturally; for Fourier the cos/sin functions remain well-defined but are off-distribution.

**In `seed_sweep_analysis.py` the default attack is PGD with L2 norm, 20 steps.**

### Robustness metric

For each epsilon:
```
rob(ε) = E_{(x,y)~test}[ 1{ argmax_c p(c|x_adv) = y } ]
```
i.e., accuracy on the adversarially perturbed test set. Reported as `eval/test/rob/<eps>`.

---

## Uncertainty Quantification (UQ)

Born Machines learn the joint `p(x, c) ∝ |ψ(x, c)|²`, giving access to the marginal:

```
p(x) = Σ_c p(x, c) = Σ_c |ψ(x, c)|² / Z
```

This is computed by `BornMachine.marginal_log_probability(x)`. The partition function `Z` is cached once via `bm.cache_log_Z()`.

### Detection

Adversarial examples typically have **lower log p(x)** than clean examples — they lie off the learned distribution. The detection threshold `τ` is calibrated from the clean test set:

```
τ_p = p-th percentile of { log p(x) : x ∈ test_clean }
```

A test input is **flagged as adversarial** if `log p(x) < τ_p`.

**Detection rate** at (percentile p, epsilon ε):
```
det_rate(p, ε) = fraction of x_adv with log p(x_adv) < τ_p
```

By design, `1 - det_rate` percent of clean samples are also rejected (false positive rate ≈ p/100 on clean data).

Reported as `eval/uq_detection/<p>pct/<eps>`.

### Likelihood Purification

For adversarial inputs that pass detection (or as a standalone defense), purification finds a nearby high-likelihood point within a radius `r`:

```
x* = argmax_{‖x' - x_adv‖ ≤ r}  log p(x')
```

Implemented as **projected gradient ascent on log p(x)** within the Lp ball:

```
δ_{t+1} = Π_{‖δ‖≤r}[ δ_t - α · ∇_δ NLL(x_adv + δ_t) ]
x* = clamp(x_adv + δ_T, input_range)
```

Note: gradient **descent** on NLL = ascent on log p(x). The final clamp ensures x* stays in the valid embedding domain.

**After purification, classify x* instead of x_adv.**

Reported metrics:
- `eval/uq_purify_acc/<eps>/<r>` — accuracy on purified samples
- `eval/uq_purify_recovery/<eps>/<r>` — fraction of originally misclassified samples that become correct after purification

**Choosing r**: r should be ≥ ε (the attack radius) to have any chance of reversing the attack. In practice r ≈ ε works well; too large means purification can change the semantic content of the input.

---

## Membership Inference Attack (MIA)

Asks: can an attacker tell whether a sample was in the training set, using only the model's outputs?

**Data splits used**:
- members = training set
- non-members = test set
- validation set is excluded entirely

### Features

All features are derived from `p(c|x)`:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `max_prob` | `max_c p(c|x)` | Higher confidence on training data |
| `entropy` | `-Σ p(c) log p(c)` | Lower entropy = more confident |
| `correct_prob` | `p(y|x)` at true label | Higher on training data |
| `loss` | `-log p(y|x)` | Lower loss on training data |
| `margin` | `p_1 - p_2` (top two) | Larger margin = more confident |
| `modified_entropy` | `1 - H / log(C)` | Normalized confidence |

`use_true_labels=True` (default): uses ground-truth y for `correct_prob`/`loss` — worst-case risk, assumes attacker knows labels.
`use_true_labels=False`: uses predicted label — realistic attacker who only queries the model.

### Attack modes

**Logistic regression** (LR attack): train LR on 70% of (member, non-member) feature vectors, evaluate on 30%. Simulates a shadow-model attacker. Reported as `eval/mia_accuracy` and `eval/mia_auc_roc`.

**Worst-case (oracle) threshold**: for each feature, sweep all thresholds on the full dataset and pick the one maximizing accuracy. This is an upper bound — a real attacker cannot do this without ground-truth membership labels. Reported as `eval/mia_wc/<feat>` and `eval/mia_wc_best`.

**Adversarial MIA**: same features, but extracted from `p(c|x_adv)` where `x_adv` is a PGD adversarial example. Adversarial transferability differs between members and non-members, providing a stronger membership signal. Only uses worst-case threshold evaluation. Reported as `eval/adv_mia_wc/<feat>` and `eval/adv_mia_wc_best`.

### Interpreting results

| AUC-ROC | Privacy |
|---------|---------|
| < 0.55 | Excellent |
| 0.55 – 0.60 | Good |
| 0.60 – 0.70 | Moderate leakage |
| ≥ 0.70 | Significant leakage |

Random chance = 0.50. Accuracy 0.50 = attacker cannot distinguish members from non-members.

---

## `evaluate_run()` pipeline

`analysis/utils/evaluate.py` orchestrates per-run evaluation:

```
1. load_run_config(run_dir)           → Hydra OmegaConf config
2. apply evasion/sampling overrides   → consistent eval settings across all runs
3. BornMachine.load(checkpoint)       → model with correct embedding + input_range
4. OmegaConf.update(cfg, "dataset.overwrite", True)  → force correct data split
5. DataHandler.split_and_rescale(bm)  → rescale to bm.input_range
6. For each split: acc, clsloss, genloss, fid
7. For non-test splits: rob (via MetricFactory/RobustnessEvaluation)
8. MIA (train vs test, uses classification["train"] and classification["test"])
9. UQ (test only: detection + purification)
10. Test rob: reuse UQ's adv examples where eps overlaps; generate separately for missing eps
```

Step 10 avoids generating adversarial examples twice when UQ and rob use the same epsilons — UQ already generates them, so `evaluate_run` reuses the `adv_accuracies` dict from `UQResults`.

`dataset.overwrite = True` forces the dataset to be regenerated with the correct seed from the run's config, ensuring the data split matches exactly what was used during training.

---

## `resolve.py` — auto-detection from path

`seed_sweep_analysis.py` calls these at startup to configure itself:

```python
_EMBEDDING = resolve_embedding_from_path(SWEEP_DIR)
# → "fourier" | "legendre" | "hermite" | "chebychev1" | "chebychev2" | None

_RANGE_SIZE = embedding_range_size(_EMBEDDING)
# → 1.0 | 2.0 | 8.0 | 1.98 | 2.0

REGIME = resolve_regime_from_path(SWEEP_DIR)
# → "pre" | "gen" | "adv" | "gan"
```

Both functions tokenize the path on `/` and `_` and match against known strings. This works for both new-style paths (`outputs/seed_sweep/gen/hermite/d4D3/moons_4k_2102`) and old-style flat names (`seed_sweep_adv_d30D18fourier_moons_4k_1202`).

---

## `statistics.py` — summary and Pareto

**`create_summary_table(df, acc_col, rob_cols, ...)`**: computes mean, std, stderr across runs, plus the value from the single best run selected on validation.

**`get_pareto_runs(df, acc_col, rob_col, ...)`**: returns the subset of runs that are Pareto-optimal in the (clean accuracy, robust accuracy) plane — no other run dominates both simultaneously.

**`compute_metric_correlations(df, cols)`**: Pearson correlations between all metric pairs. Useful for checking whether clean accuracy and robust accuracy trade off.

---

## Caveats and known limitations

1. **PGD does not clamp to input domain**: adversarial examples can stray outside the embedding's valid range at the boundaries. Purification corrects this (final clamp to `input_range`), but the attack itself does not.

2. **L2 norm by default in seed_sweep_analysis.py**: The training-time evasion config (in `configs/tracking/online.yaml`) also uses L2. They should match for meaningful sanity checks.

3. **FID disabled for data_dim > 100**: FID is meaningless in high-dimensional spaces where covariance estimation is unreliable.

4. **MIA uses train vs test only** — the validation split plays no role. This is by design (no ambiguity about membership).

5. **UQ recovery_rate can be 1.0 when all adversarial examples were already correct before purification** (adv_acc ≈ clean_acc). This is a sign the attack at that epsilon is too weak, not that purification works perfectly.
