# Born Machines for Trustworthy Classification — Codebase Guide

## What this is

This repo studies whether MPS-based Born Machines trained as **generative classifiers** (learning the joint distribution p(x, c)) offer trustworthy properties — adversarial robustness, membership-inference resistance, calibrated uncertainty — compared to purely discriminative counterparts.

See `README.md` for background and setup.

---

## Architecture in a nutshell

**Born rule**: probability = |amplitude|². Inputs are embedded into a Hilbert space; the amplitude is computed by contracting the embedded input with an MPS tensor chain.

**ConditionalBornMachine** (`src/model.py`) is a single MPS that represents the joint amplitude ψ(x, c). Two contraction modes over the same tensors:
- **Classification** — parallel contraction over x yields amplitude vector ψ(x, c); squared and normalised → p(c|x). No explicit syncing required.
- **Generation (marginal)** — Σ_c |ψ(x,c)|² / Z gives p(x). Partition function Z is cached once via `cbm.cache_log_Z()`; log p(x) is then available via `cbm.marginal_log_probability(x)`.

**Architecture naming**: `d{d}r{r}` where `d` = physical/embedding dimension (in_dim), `r` = bond dimension.  
Examples: `d4r3`, `d10r6`, `d30r18`.

---

## Training regimes

| Script | Trainer token | Description |
|--------|--------------|-------------|
| `experiments/train.py` | `nat` | NLL training, any α (α=0 discriminative, α>0 generative) |
| `experiments/train.py` | `at` | Adversarial training; loads pretrained `nat` checkpoint via `model_path` |

Run as a Python module from the project root:
```bash
python -m experiments.train +experiments=moons/nat/fourier/d4r3/hpo_a0
python -m experiments.train --multirun +experiments=circles/nat/legendre/d10r6/seed_sweep_a1
python -m experiments.train --multirun +experiments=circles/at/legendre/d10r6/seed_sweep
```

---

## Configuration system

Configurations are managed with [Hydra](https://hydra.cc/). The canonical workflow:

1. Write an experiment config in `configs/experiments/` that overrides group defaults
2. Run with `+experiments=<path>` (without `.yaml`)

**Config tree** (`configs/experiments/{dataset}/{nat|at}/{embedding}/{arch}/{kind}.yaml`):
```yaml
# @package _global_
defaults:
  - override /born: legendre/d10r6
  - override /dataset: 2Dtoy/circles
  - override /trainer/nll: default
  - override /tracking: online
trainer:
  alpha: 0.0
  stop_crit: dis_loss   # dis_loss | gen_loss | mixed_loss | acc | rob
```

**Config group layout**:
```
configs/
├── config.yaml              # root defaults
├── born/{embedding}/        # d{d}r{r}.yaml files — in_dim, bond_dim, boundary
├── dataset/2Dtoy/           # circles.yaml, moons.yaml, spirals.yaml, *_small.yaml
├── dataset/mnist/           # mnist.yaml, mnist_1k.yaml, mnist_full_r12.yaml
├── dataset/ucr_ts/          # ECG200.yaml, ItalyPowerDemand.yaml, ...
├── trainer/nll/             # NLLConfig defaults + variants (debug: false required)
├── trainer/adversarial/     # AdversarialConfig (pgd_at, trades, ...)
└── tracking/                # online.yaml, offline.yaml, disabled.yaml
```

**Config dataclass location**: each module owns its config dataclass (e.g., `NLLConfig` in `src/train/nll.py`). The top-level `Config`, `TrainerConfig`, `TrackingConfig` and `register()` live in `experiments/config.py`.

**Hydra gotcha**: any field added to a config dataclass must also appear with a value in the corresponding base YAML under `configs/trainer/`. Dataclass defaults alone are not enough — Hydra raises "key not in config" otherwise.

---

## Logging

Training always writes `log.json` to the output directory — no W&B required.  
W&B is opt-in: set `tracking.mode: online` in your experiment config, or `tracking.mode: disabled` to suppress it.

```bash
# Disable W&B explicitly
python -m experiments.train +experiments=tests/nll tracking.mode=disabled
```

The epoch logger is constructed via `experiments.tracking.make_logger(output_dir, wandb_run)` and passed as `on_epoch_end` callback to the trainer.

---

## Output directory structure

```
outputs/{dataset}/{nat|at}/{embedding}/{arch}/{kind}_{date}/
```
- `dataset`: `circles`, `moons`, `spirals`, `mnist`, `mnist_full_r12`, UCR names, …
- `nat|at`: trainer token
- `arch`: `d4r3` | `d6r4` | `d10r6` | `d30r18` | `d3r20c64` (complex, 3-class)
- `kind`: `hpo_a0` | `seed_sweep_a1` | `seed_sweep` (at) | `alpha_curve` | `test`
- `date`: `DDMM` (multirun) or `DDMM_HHMM` (single run)
- Multiruns: numbered subdirs `0/`, `1/`, … each with `.hydra/` inside

Analysis mirrors the structure under `analysis/outputs/`.

---

## Analysis

Post-training analysis lives in `analysis/`. See [`analysis/GUIDE.md`](analysis/GUIDE.md) for full documentation.

Quick start:
```bash
# Analyse a completed seed sweep
python -m analysis.sweep outputs/circles/nat/legendre/d10r6/seed_sweep_a1_1802

# Analyse all unanalysed sweeps in batch
python -m analysis.batch
```

Analysis outputs land in `analysis/outputs/<sweep_path>/` as `evaluation_data.csv`, `evaluation_summary.txt`, and optionally distribution plots.

---

## Reproduction notebooks

`notebooks/` holds end-to-end notebooks (run sweeps → analyse → figures), one per benchmark:

| Notebook | Benchmark | Figures |
|----------|-----------|---------|
| `notebooks/2dtoy.ipynb` | spirals · Legendre · d10r6 | distribution panel · alpha curve · regime barplot |
| `notebooks/mnist.ipynb` | MNIST (`mnist_full_r12`) · Legendre · d3r20c64 | sampling (mean digit, α=0.01) · alpha curve · robustness curves with purification + detection overlays |

Each notebook's first cell walks up to the repo root and adds it to `sys.path`, so it runs
correctly from `notebooks/` regardless of the Jupyter working directory; all `outputs/`,
`analysis/`, and `figures/` paths are anchored on that `PROJECT_ROOT`. Figures are written
under `figures/`, which is git-ignored (regenerated by running the notebooks), as is
`notebooks/archive/` (personal experimentation).

---

## Navigation guide

| Task | Location |
|------|----------|
| Modify MPS architecture / init | `src/model.py` |
| Change embedding | `src/utils/train.py` (`_EMBEDDING_MAP`) |
| Modify NLL training loop | `src/train/nll.py` |
| Modify adversarial training loop | `src/train/adversarial.py` |
| Validation metrics (eval functions) | `src/utils/train.py` |
| Add adversarial attack | `src/utils/evasion.py` |
| Purification (likelihood-based) | `src/analysis/purification.py` |
| Data loading and rescaling | `src/datahandler.py` |
| Visualisation (matplotlib, no W&B) | `src/analysis/viz.py` |
| Experiment entry point (Hydra) | `experiments/train.py` |
| Self-contained local runner | `experiments/run_local.py` |
| W&B init + dataset viz logging | `experiments/tracking.py` |
| Config dataclasses + Hydra register | `experiments/config.py` |
| Post-hoc sweep evaluation | `analysis/sweep.py` |
| HPO result exploration | `analysis/hpo.py` |
| Single-model analysis (MIA / UQ) | `analysis/run.py` |
| Fill seed_sweep configs from HPO | `tools/fill_hpo.py` |
| Reproduce paper figures (notebooks) | `notebooks/2dtoy.ipynb`, `notebooks/mnist.ipynb` |
| Run unit tests (fast) | `pytest -m "not slow" -q` |
| Run full test suite | `pytest -q` |

---

## Known issues & gotchas

**`randn_eye` amplitude collapse with non-Fourier embeddings on high-dim data** — `randn_eye` sets the identity at physical index 0; initial amplitude ≈ φ₀^n_sites. Fourier: φ₀=1 (safe). Legendre: φ₀=√0.5 → on MNIST (n_sites=785), amplitude ≈ 10⁻¹¹⁸ → float32 underflow → all Born probs zero → silent training failure. **Fixed in `ConditionalBornMachine.__init__`** (`src/model.py`): rescales tensors by 1/φ₀ when `randn_eye` is used and φ₀≠1. Exact for Legendre; use `canonical` init for Hermite/Chebyshev.

**Evasion attacks don't clamp to `cbm.input_range`** — PGD/FGM in `src/utils/evasion.py` project delta to the ε-ball but do not clamp `naturals + delta` to the valid embedding domain. Purification correctly clamps.

**Complex MPS requires PyTorch ≥ 2.1.0** — Adam has a `foreach` bug with complex-typed parameters in older versions, causing NaN updates.

**Purification broken when amplitudes overflow** — `purification.py:258` uses `abs_square` to compute Gibbs sampling weights; `draw_from_grid` maps `posinf → 0.0`, so overflow candidates are silently zeroed and sampling is wrong. Not yet fixed.
