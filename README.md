# bm4tc — Born Machines for Trustworthy Classification

This repository studies whether **MPS-based Born Machines** trained as generative classifiers offer trustworthy properties — robustness against adversarial examples, resistance to membership inference attacks, and calibrated uncertainty — compared to their purely discriminative counterparts.

The core hypothesis is that learning the joint distribution p(x, c) rather than only p(c|x) may confer inherent robustness, because the model has an explicit notion of what in-distribution data looks like.

## Background

A **Born Machine** models a probability distribution via the Born rule from quantum mechanics: probability = |amplitude|². The amplitude function is represented as a **Matrix Product State (MPS)**, a structured tensor network that factorises a high-dimensional function into a chain of low-rank tensors. Inputs are mapped into a Hilbert space by a feature embedding before contraction with the MPS.

For classification, a special output tensor yields a class-conditioned amplitude vector; squaring and normalising gives p(c|x). For generation, the same tensors enable exact ancestral sampling of p(x|c) via sequential marginalisation.

## Training Regimes

| Script | Regime | Description |
|--------|--------|-------------|
| `experiments/classification.py` | `cls` | Discriminative classifier (NLL on p(c\|x)) |
| `experiments/generative.py` | `gen` | Classification pretraining + generative NLL on p(x,c) |
| `experiments/adversarial.py` | `adv` | Classification pretraining + PGD-AT or TRADES adversarial training |
| `experiments/ganstyle.py` | `gan` | Classification pretraining + GAN-style training with an MLP critic |

## Trustworthiness Evaluation

**Adversarial robustness** — Post-hoc PGD attack (L2, 20 steps) at multiple ε fractions of the embedding range. Robust accuracy reported per seed; sweep statistics include mean ± std and Pareto frontiers (clean accuracy vs. robust accuracy).

**Uncertainty quantification** — `BornMachine.marginal_log_probability(x)` gives log p(x) = log Σ_c |ψ(x,c)|² − log Z. Used for (i) likelihood-based detection of adversarial examples (threshold calibrated on clean test percentiles) and (ii) likelihood purification (projected gradient ascent on log p(x) within an Lp ball).

**Membership inference** — Logistic-regression attack and worst-case oracle threshold attack on confidence features derived from p(c|x). Also evaluated on adversarial inputs (adversarial MIA).

## Feature Embeddings

| Embedding | Input range | Notes |
|-----------|-------------|-------|
| `fourier` | (0, 1) | tensorkrowch built-in |
| `legendre` | (−1, 1) | Normalized Legendre polynomials, orthonormal on L²[−1,1] |
| `hermite` | (−4, 4) | Physicist's Hermite functions with Gaussian damping |
| `chebychev1` | (−0.99, 0.99) | Range capped at ±0.99 to avoid boundary weight divergence |
| `chebychev2` | (−1, 1) | Boundary-safe (weight → 0 at ±1) |

## Datasets

**2D toy** (moons, circles, spirals — 2k/4k samples): for visualising decision boundaries, generative distributions, and sanity-checking training dynamics.

**MNIST**: full and 1k-sample subsets, complex-valued MPS with Legendre embedding recommended.

**UCR univariate time series**: ECG200, ItalyPowerDemand, ChlorineConcentration, SyntheticControl, CricketX/Y/Z. The last five match the benchmark of [Ding et al. (2022)](https://arxiv.org/abs/2207.04307) for direct comparison with neural-network time-series classifiers.

## Setup

```bash
conda env create -f environment.yml   # creates env 'bm4tc'
conda activate bm4tc
```

Requires PyTorch ≥ 2.1.0 (Adam optimizer fix for complex-typed parameters). See `environment.yml` for the full dependency list.

## Running Experiments

All experiments are run as Python modules from the project root. Configurations are managed with [Hydra](https://hydra.cc/); the canonical way to design an experiment is to write a config under `configs/experiments/` and reference it with `+experiments=<path>`.

```bash
# Single run
python -m experiments.classification +experiments=classification/fourier/d4D3/hpo/moons

# Multirun / seed sweep
python -m experiments.generative --multirun +experiments=generative/legendre/d10D6/seed_sweep/moons_4k

# Batch-run all unrun configs in a filter set
python -m experiments.queue_experiments --filter-type gen --filter-embedding legendre --dry-run

# Disable W&B for local debugging
python -m experiments.classification +experiments=tests/classification tracking.mode=disabled
```

## Post-Hoc Analysis

```bash
# Analyse all completed but unanalysed sweeps
python analysis/queue_seed_sweep.py

# Analyse a specific sweep
python analysis/seed_sweep_analysis.py outputs/seed_sweep/gen/legendre/d10D6/moons_4k_1802

# Regenerate distribution plots for analysed sweeps
python analysis/queue_visualize.py
```

Results land in `analysis/outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{date}/` as `evaluation_data.csv` (one row per seed), a human-readable summary, and PNG figures.

## Repository Structure

```
bm4tc/
├── experiments/        # Entry-point scripts (classification, generative, adversarial, ganstyle)
├── configs/            # Hydra configs — born/, dataset/, trainer/, tracking/, experiments/
├── src/
│   ├── models/         # BornMachine, BornClassifier, BornGenerator, Critic
│   ├── trainer/        # Training loops for each regime
│   ├── tracking/       # PerformanceEvaluator, W&B utilities, FID, visualisation
│   ├── data/           # DataHandler, dataset generation and loading
│   └── utils/          # Schemas, embeddings, losses, PGD/FGM attacks, purification
├── analysis/
│   ├── seed_sweep_analysis.py   # Post-hoc metrics for one sweep
│   ├── queue_seed_sweep.py   # Batch runner over all sweeps
│   ├── queue_visualize.py  # Batch distribution plot regenerator
│   ├── utils/              # evaluate.py, uq.py, mia.py, resolve.py, statistics.py
│   └── outputs/            # Generated analysis artifacts (git-ignored)
└── environment.yml
```

See `GUIDE.md` for a detailed walkthrough of the codebase, and the per-module `GUIDE.md` files under each subdirectory.

## Key Dependencies

| Library | Role |
|---------|------|
| [tensorkrowch](https://joserapa98.github.io/tensorkrowch/) | MPS construction, contraction, and training |
| [Hydra](https://hydra.cc/) | Configuration management and HPO (Optuna sweeper) |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking |
| PyTorch ≥ 2.1.0 | Autograd, optimizers |
