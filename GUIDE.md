# Born Machines for Trustworthy Classification — Codebase Guide

## What this is

This repo studies whether MPS-based Born Machines trained as **generative classifiers** (learning the joint distribution p(x, c)) offer trustworthy properties — adversarial robustness, membership-inference resistance, calibrated uncertainty — compared to purely discriminative counterparts.

See `README.md` for background and setup.

---

## Architecture in a nutshell

**Born rule**: probability = |amplitude|². Inputs are embedded into a Hilbert space; the amplitude is computed by contracting the embedded input with an MPS tensor chain.

**BornMachine** (`src/models/born.py`) owns two views over the same shared tensors:
- **BornClassifier** — parallel contraction, yields class-conditional amplitude vector ψ(x, c); squared and normalised → p(c|x)
- **BornGenerator** — sequential contraction for exact ancestral sampling of p(x|c)

Both share tensors directly. After a training step on one view, call `bm.sync_tensors(after=...)` before evaluating with the other — otherwise they drift.

**Architecture naming**: `d{d}r{r}` where `d` = physical/embedding dimension (in_dim), `r` = bond dimension.  
Examples: `d4r3`, `d10r6`, `d30r18`.

---

## Training regimes

| Script | Regime | Description |
|--------|--------|-------------|
| `experiments/discriminative.py` | `dis` | Discriminative NLL on p(c\|x) |
| `experiments/generative.py` | `gen` | Classification pretraining + generative NLL on p(x,c) |
| `experiments/adversarial.py` | `adv` | Classification pretraining + PGD-AT or TRADES adversarial training |

Each experiment script runs as a Python module from the project root:
```bash
python -m experiments.discriminative +experiments=discriminative/fourier/d4r3/hpo/moons
python -m experiments.generative     +experiments=generative/legendre/d10r6/seed_sweep/circles
python -m experiments.adversarial    +experiments=adversarial/fourier/d4r3/hpo/moons
```

---

## Configuration system

Configurations are managed with [Hydra](https://hydra.cc/). The canonical workflow:

1. Write an experiment config in `configs/experiments/` that overrides group defaults
2. Run with `+experiments=<path>` (without `.yaml`)

**Example experiment config** (`configs/experiments/discriminative/fourier/d4r3/hpo/moons.yaml`):
```yaml
# @package _global_
defaults:
  - override /born: fourier/d4r3
  - override /dataset: 2Dtoy/moons
  - override /trainer/discriminative: adam500_loss
  - override /tracking: online
  - override /trainer/generative: null
  - override /trainer/adversarial: null
```

**Config group layout**:
```
configs/
├── config.yaml              # root defaults
├── born/{embedding}/        # d{d}r{r}.yaml files — in_dim, bond_dim, boundary
├── dataset/2Dtoy/           # circles.yaml, moons.yaml, spirals.yaml, *_small.yaml
├── dataset/mnist/           # mnist.yaml, mnist_1k.yaml
├── dataset/ucr_ts/          # ECG200.yaml, ItalyPowerDemand.yaml, ...
├── trainer/discriminative/  # DiscriminativeConfig defaults + variants
├── trainer/generative/      # GenerativeConfig defaults
├── trainer/adversarial/     # AdversarialConfig (pgd_at, trades, ...)
└── tracking/                # online.yaml, offline.yaml, disabled.yaml
```

**Config dataclass location**: each module owns its config dataclass (e.g., `DiscriminativeConfig` in `src/trainer/discriminative.py`). The top-level `Config`, `TrainerConfig`, `TrackingConfig` and `register()` live in `experiments/config.py`.

---

## Logging

Training always writes `log.json` to the output directory — no W&B required.  
W&B is opt-in: set `tracking.mode: online` in your experiment config, or `tracking.mode: disabled` to suppress it.

```bash
# Disable W&B explicitly
python -m experiments.discriminative +experiments=tests/discriminative tracking.mode=disabled
```

The epoch logger is constructed via `experiments.tracking.make_logger(output_dir, wandb_run)` and passed as `on_epoch_end` callback to the trainer.

---

## Output directory structure

```
outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{date}/
```
- `kind`: `hpo` | `seed_sweep` | `alpha_curve` | `test`
- `regime`: `dis` | `gen` | `adv`
- `arch`: `d4r3` | `d6r4` | `d10r6` | `d30r18`
- `date`: `DDMM` (multirun) or `DDMM_HHMM` (single run)
- Single runs: `.hydra/` directly in sweep root
- Multiruns: numbered subdirs `0/`, `1/`, … each with `.hydra/` inside

---

## Analysis

Post-training analysis lives in `analysis/`. See [`analysis/GUIDE.md`](analysis/GUIDE.md) for full documentation.

Quick start:
```bash
# Analyse a completed seed sweep
python analysis/seed_sweep_analysis.py outputs/seed_sweep/gen/fourier/d4r3/moons_1802

# Analyse all unanalysed sweeps in batch (no distribution plots)
python analysis/queue_seed_sweep.py
```

Analysis outputs land in `analysis/outputs/<sweep_path>/` as `evaluation_data.csv`, `evaluation_summary.txt`, and optionally distribution plots.

---

## Navigation guide

| Task | Location |
|------|----------|
| Modify MPS architecture / init | `src/models/born.py` |
| Change embedding | `src/utils/get.py` (`_EMBEDDING_MAP`) |
| Add / change loss function | `src/utils/criterions.py` |
| Modify training loop | `src/trainer/discriminative.py`, `generative.py`, `adversarial.py` |
| Validation metrics (eval functions) | `src/trainer/eval.py` |
| Add adversarial attack | `src/utils/evasion.py` |
| Purification (likelihood-based) | `src/analysis/purification.py` |
| Data loading and rescaling | `src/data/handler.py`, `src/data/gen_n_load.py` |
| Visualisation (matplotlib, no W&B) | `src/analysis/viz.py` |
| Experiment entry points | `experiments/discriminative.py`, `generative.py`, `adversarial.py` |
| W&B init + dataset viz logging | `experiments/tracking.py` |
| Config dataclasses + Hydra register | `experiments/config.py` |
| Post-hoc sweep evaluation | `analysis/seed_sweep_analysis.py` |
| HPO result exploration | `analysis/hpo_analysis.py` |
| MIA deep-dive (single run) | `analysis/mia_analysis.py` |
| UQ deep-dive (single run) | `analysis/uq_analysis.py` |
| Fill seed_sweep configs from HPO | `configs/tools/fill_hpo.py` |
| Run unit tests (fast) | `pytest -m "not slow" -q` |
| Run full test suite | `pytest -q` |

---

## Known issues & gotchas

**`randn_eye` amplitude collapse with non-Fourier embeddings on high-dim data** — `randn_eye` sets the identity at physical index 0; initial amplitude ≈ φ₀^n_sites. Fourier: φ₀=1 (safe). Legendre: φ₀=√0.5 → on MNIST (n_sites=785), amplitude ≈ 10⁻¹¹⁸ → float32 underflow → all Born probs zero → silent training failure. **Fixed in `BornMachine.__init__`**: rescales tensors by 1/φ₀ when `randn_eye` is used and φ₀≠1. Exact for Legendre; use `canonical` init for Hermite/Chebyshev.

**Evasion attacks don't clamp to `bm.input_range`** — PGD/FGM in `src/utils/evasion/minimal.py` project delta to the ε-ball but do not clamp `naturals + delta` to the valid embedding domain. Purification correctly clamps.

**Complex BornMachines require PyTorch ≥ 2.1.0** — Adam has a `foreach` bug with complex-typed parameters in older versions, causing NaN updates.

**`sync_tensors` is required after each training phase** — classifier and generator share tensor data but track their own parameter views. Without sync, one view can silently diverge from the other.
