# bm4tc — Born Machines for Trustworthy Classification

This repository studies whether **MPS-based Born Machines** trained as generative classifiers offer trustworthy properties — robustness against adversarial examples, resistance to membership inference attacks, and calibrated uncertainty — compared to their purely discriminative counterparts.

The core hypothesis is that learning the joint distribution p(x, c) rather than only p(c|x) may confer inherent robustness, because the model has an explicit notion of what in-distribution data looks like.

## Background

A **Born Machine** models a probability distribution via the Born rule from quantum mechanics: probability = |amplitude|². The amplitude function is represented as a **Matrix Product State (MPS)**, a structured tensor network that factorises a high-dimensional function into a chain of low-rank tensors. Inputs are mapped into a Hilbert space by a feature embedding before contraction with the MPS.

For classification, a special output tensor yields a class-conditioned amplitude vector; squaring and normalising gives p(c|x). The marginal p(x) = Σ_c |ψ(x,c)|² / Z is available analytically via `cbm.marginal_log_probability(x)` (partition function Z cached via `cbm.cache_log_Z()`), enabling likelihood-based detection and purification without sampling.

## Training Regimes

| Script | Trainer token | Description |
|--------|--------------|-------------|
| `experiments/train.py` | `nat` | NLL training; α=0 discriminative (`hpo_a0`, `seed_sweep_a0`) or α>0 generative (`seed_sweep_a1`, …) |
| `experiments/train.py` | `at` | Adversarial training; loads a pretrained `nat` checkpoint via `model_path` |

**Norm control** (shared by both regimes, configured via `trainer.*.norm_control`) keeps the log-partition function log Z near a target during α>0 training: `hard_every` steps the tensors are rescaled (`cbm.renormalize_`), and `soft_strength>0` adds a `(log Z − log_target)²` penalty to the loss. `log_target` may be a float, `null` (the pretrained model's log Z), or an expression in `n_features`/`in_dim`/`out_dim`/`bond_dim`. It is **on by default for `nat`** (`hard_every=1`) and **off by default for `at`** (`hard_every=0, soft_strength=0`).

## Trustworthiness Evaluation

**Adversarial robustness** — Post-hoc PGD attack (L2, 20 steps) at multiple ε fractions of the embedding range. Robust accuracy reported per seed; sweep statistics include mean ± std and Pareto frontiers (clean accuracy vs. robust accuracy).

**Uncertainty quantification** — `ConditionalBornMachine.marginal_log_probability(x)` gives log p(x) = log Σ_c |ψ(x,c)|² − log Z. Used for (i) likelihood-based detection of adversarial examples (threshold calibrated on clean test percentiles) and (ii) likelihood purification (projected gradient ascent on log p(x) within an Lp ball).

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

## Installation

### Option A — conda (local development)

```bash
conda env create -f environment.yml   # creates env 'bm4tc'
conda activate bm4tc
```

### Option B — pip virtualenv (clusters without conda)

```bash
# Create and activate the virtualenv
virtualenv -p python3.10 bm4tc_env
source bm4tc_env/bin/activate

# Core dependencies
pip install numpy scipy matplotlib jupyter ipykernel tqdm \
    scikit-learn pandas h5py opt_einsum \
    hydra-core hydra-optuna-sweeper wandb pytest

# PyTorch with CUDA (adjust --index-url for your driver version)
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Tensor network library
pip install tensorkrowch
```

If your cluster requires a proxy, prepend `--proxy http://proxy.example.fr:3128` to each
`pip install` call.

Requires PyTorch ≥ 2.1.0 (Adam optimizer fix for complex-typed parameters).

### Data path (`BM4TC_DATA_ROOT`)

By default all outputs and cached datasets are written relative to the repository root.
On clusters where code and data live on separate filesystems, set:

```bash
export BM4TC_DATA_ROOT=/path/to/data/root
# e.g. /ceph/chercheurs/nisseng261/bm4tc
```

Add this line to your virtualenv's `activate` script (or `.bashrc` / SLURM job header)
so it is always set when running experiments.  When `BM4TC_DATA_ROOT` is unset the
repository root is used — the local development layout is unchanged.

Paths under `BM4TC_DATA_ROOT`:

| Subdirectory | Contents |
|---|---|
| `outputs/` | Training run checkpoints and Hydra configs |
| `analysis/outputs/` | Post-hoc evaluation CSVs and figures |
| `.datasets/` | Cached datasets (MNIST, UCR, 2D toy) |

### Weights & Biases

Run `wandb login` once per machine (stores the API key in `~/.netrc`).

On clusters where `/home/` is not mounted on compute nodes, `~/.netrc` is invisible at
runtime and W&B raises a `CommError`.  Use the environment variable instead:

```bash
export WANDB_API_KEY=<your_key>   # add to job script or virtualenv activate
```

The default entity and project are set in `configs/tracking/online.yaml`.  To override
at runtime:

```bash
python -m experiments.train ... tracking.entity=my-entity tracking.project=my-project
```

To run without W&B logging: `tracking.mode=disabled`.

## Running Experiments

All experiments are run as Python modules from the project root. Configurations are managed with [Hydra](https://hydra.cc/); the canonical way to design an experiment is to write a config under `configs/experiments/` and reference it with `+experiments=<path>`.

```bash
# Single run (NLL discriminative, alpha=0)
python -m experiments.train +experiments=moons/nat/fourier/d4r3/hpo_a0

# Multirun / seed sweep (NLL generative, alpha=1)
python -m experiments.train --multirun +experiments=circles/nat/legendre/d10r6/seed_sweep_a1

# Adversarial training seed sweep
python -m experiments.train --multirun +experiments=circles/at/legendre/d10r6/seed_sweep

# Batch-run all unrun configs in a filter set
python -m experiments.batch --trainer nat --embedding legendre --dry-run

# Disable W&B for local debugging
python -m experiments.train +experiments=tests/nll tracking.mode=disabled
```

## Post-Hoc Analysis

```bash
# Analyse a specific sweep
python -m analysis.sweep outputs/circles/nat/legendre/d10r6/seed_sweep_a1_1802

# Analyse all completed but unanalysed sweeps in batch
python -m analysis.batch
```

Results land in `analysis/outputs/{dataset}/{nat|at}/{embedding}/{arch}/{kind}_{date}/` as `evaluation_data.csv` (one row per seed), a human-readable summary, and PNG figures.

## Reproduction Notebooks

Self-contained notebooks under `notebooks/` walk from running sweeps → analysis → paper
figures. Activate the `bm4tc` env, launch Jupyter, and run top-to-bottom (a bootstrap cell
resolves the repo root, so the working directory does not matter):

| Notebook | Benchmark | Figures |
|----------|-----------|---------|
| `notebooks/2dtoy.ipynb` | spirals (Legendre, d10r6) | distribution panel · alpha curve · regime barplot |
| `notebooks/mnist.ipynb` | MNIST (Legendre, d3r20c64) | sampling (mean digit) · alpha curve · robustness curves (purification + detection) |

Generated figures are written under `figures/` (git-ignored; regenerate by running the notebook).

## Repository Structure

```
bm4tc/
├── experiments/        # Entry-point scripts (train.py, run_local.py, batch.py)
├── configs/            # Hydra configs — born/, dataset/, trainer/, tracking/, experiments/
├── src/
│   ├── model.py        # ConditionalBornMachine
│   ├── datahandler.py  # DataHandler, dataset generation and loading
│   ├── train/          # NLLTrainer, AdversarialTrainer
│   ├── analysis/       # viz.py, purification.py, mia.py, uq.py (no W&B dependency)
│   └── utils/          # Embeddings, PGD/FGM attacks, optimizer config, train utilities
├── analysis/
│   ├── sweep.py        # Post-hoc metrics for one seed sweep / alpha curve
│   ├── batch.py        # Batch-run all unanalysed sweeps
│   ├── hpo.py          # HPO result exploration
│   ├── run.py          # Single-model analysis (MIA, UQ)
│   ├── utils/          # statistics.py, resolve.py, wandb_fetcher.py, mia_utils.py
│   └── outputs/        # Generated analysis artifacts (git-ignored)
├── tools/              # Pipeline tools (fill_hpo.py, delete_runs.py, …)
├── notebooks/          # Reproduction notebooks (2dtoy.ipynb, mnist.ipynb; archive/ git-ignored)
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
