# JEM baseline guide

This package is an isolated, conservative MLP/JEM baseline for the existing
`mnist_full_r12` MPS experiments. Run every command from the repository root
while on the `jem` branch.

## Scope

- Same 70k MNIST pool, split, 12×12 bilinear resize and global scaling to
  `[-1, 1]` as the MPS pipeline.
- Degree-of-freedom-matched MLP: `144 → 550 → 480 → 10`, exactly 349,040
  real parameters versus 174,520 complex elements in `d3r20c64`
  (two real degrees of freedom per complex element).
- Convex objective
  `L_alpha = (1-alpha) CE + alpha joint_contrastive_loss`.
- Alpha ladder: `0, 0.01, 0.1, 0.2, 0.5, 1`.
- Separate discriminative MLP-PGD-AT baseline.
- No JEM+AT interpolation in this phase.
- Basic classification PGD and a likelihood-aware adaptive PGD.
- Gradient-score and locally projected-SGLD purification. No Gibbs for JEM.

The JEM marginal score is unnormalised because its partition function is
intractable. It is valid for ranking, detection, SGLD and purification, but its
absolute value is not comparable to the exact MPS log-likelihood.

All entry points use `device=auto`, matching the MPS training entry point:
CUDA is selected whenever `torch.cuda.is_available()` and CPU is the fallback.
Training minibatches, adversarial examples, purification and SGLD transitions
therefore run on the GPU. Use `device=cuda:0` or `device=cpu` to override
training, and `--device cuda:0` or `--device cpu` for analysis/generation.

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

Selection uses validation accuracy; the returned Hydra objective is `-accuracy`.
When Optuna finishes, its callback automatically writes the selected learning
rate and weight decay into `configs/experiment/seed_sweep/a0.yaml`.
This intentionally leaves a small version-controlled YAML diff so the selected
hyperparameters remain reproducible and reviewable.

## 2. Train the alpha=0 seed sweep

```bash
python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a0
```

Runs seeds 1–5. Note the path of the selected alpha=0 checkpoint, for example:

```text
outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt
```

## 3. HPO for each alpha>0

Warm-start from the selected alpha=0 model. Run the same HPO for every alpha:

```bash
python -m baselines.jem.train --multirun \
  +experiment=hpo/pretrained \
  run_name=pretrained_a001 \
  trainer.alpha=0.01 \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt
```

Repeat with:

| Experiment | `trainer.alpha` |
|---|---:|
| `pretrained_a001` | 0.01 |
| `pretrained_a01` | 0.1 |
| `pretrained_a02` | 0.2 |
| `pretrained_a05` | 0.5 |
| `pretrained_a1` | 1.0 |

The search varies learning rate, energy regularisation, and the training-SGLD
step size, noise and number of steps. For alpha>0, checkpoints and Optuna trials
are selected by validation mixed CD loss. Validation uses an independent fixed
100-step sampler from `configs/validation_sampler/default.yaml`, so a weak
candidate sampler cannot win merely by producing easy negatives. Accuracy is
still logged and CE retains unit weight in the mixed objective. On successful
completion, the Optuna callback writes all five selected values directly into
the seed-sweep YAML corresponding to `trainer.alpha`.

The values currently present in the alpha>0 seed-sweep YAMLs came from the
previous accuracy-only protocol. Re-running this HPO replaces all five selected
parameters (`lr`, `energy_l2`, SGLD step size, noise and number of steps)
automatically before the final comparison.

`log.json` and W&B also record training- and validation-chain score gain,
L2 displacement and boundary saturation. These diagnostics distinguish a
useful negative sampler from chains that remain near their initialization or
collapse onto the `[-1,1]` boundary.

## 4. Train the alpha ladder

For each config, provide the selected alpha=0 checkpoint:

```bash
python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a001 \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a01 \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a02 \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a05 \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt

python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/a1 \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt
```

## 5. MLP adversarial-training baseline

First tune the AT learning rate and clean weight:

```bash
python -m baselines.jem.train --multirun \
  +experiment=hpo/at \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt
```

The callback updates `seed_sweep/at.yaml`; then run five seeds with the chosen
settings:

```bash
python -m baselines.jem.train --multirun \
  +experiment=seed_sweep/at \
  model_path=outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_DDMM/2/models/model.pt
```

This trainer is deliberately discriminative. It does not expose `alpha`.

## 6. Analyze one run or a seed sweep

One run:

```bash
python -m baselines.jem.analysis \
  outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a001_DDMM/0
```

Full sweep:

```bash
python -m baselines.jem.sweep \
  outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a001_DDMM
```

The sweep produces:

```text
analysis/outputs/baselines/jem/.../evaluation_data.csv
analysis/outputs/baselines/jem/.../evaluation_summary.csv
analysis/outputs/baselines/jem/.../evaluation_summary.txt
```

Metric names follow the current MPS analysis where applicable:

- `acc`, `dis_loss`, `rob/{eps}`
- `uq_adv_acc/{eps}` and `uq_joint_adv_acc/{eps}`
- `uq_detection/{q}pct/{eps}`
- `uq_det_err_detected/{q}pct/{eps}`
- `uq_det_err_passed/{q}pct/{eps}`
- `uq_purify_acc/{eps}/{radius}`

The likelihood-aware adaptive attack also uses the MPS-compatible
`uq_joint_*` metric family. JEM-specific additions are:

- `adaptive_rob/{eps}`, `adaptive_detection/{q}pct/{eps}`
- `px_cd_loss`, `joint_cd_loss` and separated positive/negative scores
- `sgld_purify_acc/{eps}/{k}` and purification recovery rates
- `sgld_joint_purify_acc/{eps}/{k}`

Attack epsilons and likelihood-purification radii are absolute in `[-1,1]`,
matching MPS analysis. Sampling purification uses `delta=0.2` absolute, which
matches the MPS Gibbs setting `0.1 * input_range_size`. One SGLD sweep runs 20
transitions inside the local L-infinity ball around its starting state; the
next sweep is recentered on the previous output. The analysis records the same
snapshots as MPS, `k=1,3,5`, under `Purif. (samp., k=...)`. Purification fixes
`step_size=0.01` and `noise_std=0.005` across every model instead of inheriting
the model-specific training-SGLD hyperparameters.

By default, sampling purification uses a fixed 1000-example subset, matching
the MPS Gibbs protocol, while likelihood purification uses the complete clean
test set. Use `--sampling-subsample` or `--defense-subsample` to override these
independently. Use `--threshold-split valid` for a stricter
validation-calibrated detector. As in MPS, “OOD detection” here means
likelihood-based detection of adversarial MNIST examples; no external image
dataset is loaded. `--adaptive-score-weight` controls the
classification/score balance of the likelihood-aware attack and defaults to
`1.0`.

## 7. Generate samples

```bash
python -m baselines.jem.generate \
  outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a001_DDMM/0 \
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
  --mps 0=analysis/outputs/mnist_full_r12/.../a0/evaluation_data.csv \
  --mps 0.01=analysis/outputs/mnist_full_r12/.../a001/evaluation_data.csv \
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

The same workflow is available interactively in
`notebooks/jem_mnist.ipynb`. It mirrors the MPS `notebooks/mnist.ipynb`
structure and produces alpha curves, purification-radius comparisons,
per-epsilon defense bar charts, detection-threshold curves, the per-class mean
sample grid and the same LaTeX/plain-text tables as the MPS notebook under
`figures/jem_mnist/`. Figure and strategy names are preserved. Locally
projected SGLD replaces Gibbs and is reported as `Purif. (samp., k=...)` at
`k=1,3,5`; the other purification remains named `Purif. (lk.)`.

Validation data are used for model selection. Detector thresholds default to
the clean test percentiles solely to reproduce the current MPS protocol; the
guide exposes `--threshold-split valid` for future leakage-free evaluation.
