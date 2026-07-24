# JEM baseline guide

This package is an isolated, conservative MLP/JEM baseline for the existing
`mnist_full_r12` MPS experiments. Run every command from the repository root
while on the `jem` branch.

## Scope

- Same 70k MNIST pool, split, 12×12 bilinear resize and global scaling to
  `[-1, 1]` as the MPS pipeline.
- Parameter-matched MLP: `144 → 347 → 347 → 10`, 174,551 real parameters
  versus 174,520 complex elements in `d3r20c64`.
- Convex objective
  `L_alpha = (1-alpha) CE + alpha joint_contrastive_loss`.
- Alpha ladder: `0, 0.01, 0.1, 0.2, 0.5, 1`.
- Separate discriminative MLP-PGD-AT baseline.
- No JEM+AT interpolation in this phase.
- Basic classification PGD and a likelihood-aware adaptive PGD.
- Gradient-score and projected-SGLD purification. No Gibbs for JEM.

The JEM marginal score is unnormalised because its partition function is
intractable. It is valid for ranking, detection, SGLD and purification, but its
absolute value is not comparable to the exact MPS log-likelihood.

## 0. Smoke test

```bash
pytest tests/baselines/jem -q

python -m baselines.jem.train \
  +experiment=test
```

Use `dataset=mnist_1k_r12` for cheap manual checks.

## 1. HPO for the discriminative MLP

```bash
python -m baselines.jem.train --multirun \
  +experiment=hpo/a0
```

Inspect the Optuna results and copy the selected learning rate and weight decay
into `configs/experiment/seed_sweep/a0.yaml` or pass them as CLI overrides.
Selection uses validation accuracy; the returned Hydra objective is `-accuracy`.

## 2. Train the alpha=0 seed sweep

```bash
python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a0
```

Runs seeds 1–5. Note the path of the selected alpha=0 checkpoint, for example:

```text
outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt
```

## 3. HPO for each alpha>0

Warm-start from the selected alpha=0 model. Run the same HPO for every alpha:

```bash
python -m baselines.jem.train --multirun \
  +experiment=hpo/pretrained \
  run_name=pretrained_a001 \
  trainer.alpha=0.01 \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt
```

Repeat with:

| Experiment | `trainer.alpha` |
|---|---:|
| `pretrained_a001` | 0.01 |
| `pretrained_a01` | 0.1 |
| `pretrained_a02` | 0.2 |
| `pretrained_a05` | 0.5 |
| `pretrained_a1` | 1.0 |

The small search varies learning rate, energy regularisation and SGLD step
size. Copy the selected values into the corresponding seed-sweep YAML before
the final run, or provide them through CLI overrides.

## 4. Train the alpha ladder

For each config, provide the selected alpha=0 checkpoint:

```bash
python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a001 \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a01 \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a02 \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a05 \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a1 \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt
```

## 5. MLP adversarial-training baseline

First tune the AT learning rate and clean weight:

```bash
python -m baselines.jem.train --multirun \
  +experiment=hpo/at \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt
```

Then run five seeds with the chosen settings:

```bash
python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/at \
  model_path=outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a0_DDMM/2/models/model.pt
```

This trainer is deliberately discriminative. It does not expose `alpha`.

## 6. Analyze one run or a seed sweep

One run:

```bash
python -m baselines.jem.analysis \
  outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a001_DDMM/0
```

Full sweep:

```bash
python -m baselines.jem.sweep \
  outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a001_DDMM
```

The sweep produces:

```text
analysis/outputs/baselines/jem/.../evaluation_data.csv
analysis/outputs/baselines/jem/.../evaluation_summary.csv
analysis/outputs/baselines/jem/.../evaluation_summary.txt
```

Metric names follow the current MPS analysis where applicable:

- `acc`, `dis_loss`, `rob/{eps}`
- `uq_detection/{q}pct/{eps}`
- `uq_det_err_detected/{q}pct/{eps}`
- `uq_det_err_passed/{q}pct/{eps}`
- `uq_purify_acc/{eps}/{radius}`

JEM-specific additions are:

- `adaptive_rob/{eps}`, `adaptive_detection/{q}pct/{eps}`
- `sgld_purify_acc/{eps}/{radius}` and purification recovery rates
- adaptive-attack purification metrics with the `adaptive_` prefix
- `ood/{dataset}/{auroc,aupr_in,aupr_out,fpr95}` and `ood_msp/...`

Attack epsilons are absolute in `[-1,1]`, matching MPS analysis. By default,
detector percentiles and purification use the complete clean test set, matching
the current MPS analysis. Use `--threshold-split valid` for a stricter
validation-calibrated detector, or `--defense-subsample 1000` for a cheap fixed
purification estimate. Pass `--no-ood` when OOD datasets should not be
downloaded. `--adaptive-score-weight` controls the classification/score balance
of the likelihood-aware attack and defaults to `1.0`.

## 7. Generate samples

```bash
python -m baselines.jem.generate \
  outputs/baselines/jem/mnist_full_r12/natural/mlp_h347/seed_sweep/a001_DDMM/0 \
  --steps 1000 --per-class 8
```

This performs class-conditional SGLD, the natural sampler for JEM.

## 8. Combine with the existing MPS results

The comparison command consumes the CSVs produced by each pipeline; it does not
reload or reinterpret MPS checkpoints:

```bash
python -m baselines.jem.compare \
  --jem 0=analysis/outputs/baselines/jem/.../a0/evaluation_data.csv \
  --jem 0.01=analysis/outputs/baselines/jem/.../a001/evaluation_data.csv \
  --jem 0.1=analysis/outputs/baselines/jem/.../a01/evaluation_data.csv \
  --mps 0=analysis/outputs/outputs/mnist_full_r12/.../a0/evaluation_data.csv \
  --mps 0.01=analysis/outputs/outputs/mnist_full_r12/.../a001/evaluation_data.csv \
  --at MLP-AT=analysis/outputs/baselines/jem/.../at/evaluation_data.csv
```

Add the remaining alpha values in the same way. It writes the per-run combined
table, mean/std/count summary and common alpha plots. Only identically named
metrics are compared.

## Recommended execution order

1. Tests and `+experiment=test`.
2. Alpha=0 HPO.
3. Alpha=0 seed sweep.
4. Per-alpha pretrained HPO.
5. Alpha ladder seed sweeps.
6. MLP-AT HPO and seed sweep.
7. Analyze every sweep.
8. Generate samples for the selected JEM runs.

Validation data are used for model selection. Detector thresholds default to
the clean test percentiles solely to reproduce the current MPS protocol; the
guide exposes `--threshold-split valid` for future leakage-free evaluation.
