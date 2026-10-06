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
| `experiments/train.py` | `nat` | `trainer.evasion: null`; any α (α=0 discriminative, α>0 generative) |
| `experiments/train.py` | `at` | `trainer.evasion` set; always warm from the study's `warm_from` NAT runs |

Run as a Python module from the project root:
```bash
# Train every (cell, seed) of a study; finished runs are skipped, so relaunching resumes
python -m experiments.train spirals_nat

# AT studies are warm: their alpha=0 NAT study (warm_from, default {dataset}_nat) first
python -m experiments.train spirals_at

# One cell / one seed; --replace archives a finished run whose config changed
python -m experiments.train spirals_nat --cell legendre/d10r6/a0.01 --seed 3

# The seam study, as a quick local check (no W&B)
python -m experiments.train tests/seam_nat
```

A study (`configs/studies/<name>.yaml`) is merged onto `configs/defaults.yaml`: the
dataset, the regime (`nat` | `at`), `init` (cold | warm), a grid (embedding x arch x
alpha [x eps for AT]), fixed run-config values (`config:`) and the HPO space. Studies
with an HPO space need their selected hparams in `configs/hparams/<study>.yaml`
(written by `select`, Phase 5) and fail loudly without them. Runs land in
`outputs/{study}/{embedding}/{arch}/a{alpha}[/eps{eps}]/s{seed}/` with a `run.json`
(identity, warm start, git sha, resolved config and its hash).

**The objective.** `L = (1-α)·[(1-cw)·L_dis(x_adv) + cw·L_dis(x)] + α·L_gen(x)`, with
`cw = trainer.clean_weight`; NAT has no `x_adv` and is `(1-α)·L_dis(x) + α·L_gen(x)`. The
generative term always sees clean data. Validation mirrors the same objective every
`trainer.eval_every` epochs (attacking a fixed `(1-cw)` subset of the valid split); the
epoch with the lowest `objective/valid` is kept, and `patience` counts validations.

**Possible ablation (D18):** the generative term on adversarial inputs,
`(1-cw)·mixed_nll(x_adv, α) + cw·mixed_nll(x, α)`. It was the default before the
`ousterhout` refactor and was deleted: it trains p(x) to put high density on adversarial
points, which works against detection and purification. That is what would make it an
informative ablation, and why it must not be the default. Recover it from the tag
`pre-ousterhout` (`gen_on_clean: false`).


---

## Configuration system

Configurations are managed with [Hydra](https://hydra.cc/) and checked against the
structured schema in `experiments/config.py` (D25): an unknown key in a YAML file or a
plain override fails at composition. Selection is a fixed rule, not a parameter: the
best epoch is the one with the lowest `objective/valid` (D8).

**Config layout**:
```
configs/
├── config.yaml              # the run config: picks group options; constants are schema defaults
├── defaults.yaml            # study-level defaults: seeds, alpha ladder, radius, HPO space, budgets
├── studies/                 # one file per study (D36); studies/tests/ holds the seam study
├── hparams/                 # <study>.yaml, written by `select` only (D34)
├── dataset/                 # spirals, circles, moons, mnist12, mnist, mnist_1k, UCR datasets
├── trainer/                 # nat.yaml, at.yaml (the PGD-AT preset)
└── tracking/                # online.yaml, disabled.yaml
```

**Config dataclass location**: each module owns its config dataclass (e.g., `TrainConfig` in `src/train/trainer.py`). The top-level `Config`, `TrackingConfig` and `register()` live in `experiments/config.py`.

**Logged metric keys** follow `quantity/split[/budget]` and are built only in `experiments/metrics.py` (D48): `objective/{train,valid}`, `penalty/train`, `loss_{dis,gen,adv}/valid`, `acc/valid`, `rob/valid/{eps_rel}`, `n_rob/valid`, `eps_rel/train`, `norm/*`.

---

## Logging

Training always writes `log.json` to the output directory — no W&B required.  
W&B is on (`tracking: online`); a study sets `tracking.mode: disabled` in its `config:` to suppress it. W&B groups a grid cell's seeds (group `{study}/{cell}`, run name `s{seed}`, D47).

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
python -m analysis.sweep outputs/spirals/nat/legendre/d10r6c64/seed_sweep/cold_a1_0208

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
| Modify the training loop (NAT and AT) | `src/train/trainer.py` |
| Validation metrics (eval functions) | `src/utils/train.py` |
| Add adversarial attack | `src/utils/evasion.py` |
| Purification (likelihood-based) | `src/analysis/purification.py` |
| Data loading and rescaling | `src/datahandler.py` |
| Visualisation (matplotlib, no W&B) | `src/analysis/viz.py` |
| Experiment entry point (Hydra) | `experiments/train.py` |
| W&B init + dataset viz logging | `experiments/tracking.py` |
| Config dataclasses + Hydra register | `experiments/config.py` |
| Post-hoc sweep evaluation | `analysis/sweep.py` |
| Single-model analysis (rob / UQ) | `analysis/run.py` |
| Reproduce paper figures (notebooks) | `notebooks/2dtoy.ipynb`, `notebooks/mnist.ipynb` |
| Run unit tests (fast) | `pytest -m "not slow" -q` |
| Run full test suite | `pytest -q` |

---

## Known issues & gotchas

**`randn_eye` amplitude collapse with non-Fourier embeddings on high-dim data** — `randn_eye` sets the identity at physical index 0; initial amplitude ≈ φ₀^n_sites. Fourier: φ₀=1 (safe). Legendre: φ₀=√0.5 → on MNIST (n_sites=785), amplitude ≈ 10⁻¹¹⁸ → float32 underflow → all Born probs zero → silent training failure. **Fixed in `ConditionalBornMachine.__init__`** (`src/model.py`): rescales tensors by 1/φ₀ when `randn_eye` is used and φ₀≠1. Exact for Legendre; use `canonical` init for Hermite/Chebyshev.

**Evasion attacks don't clamp to `cbm.input_range`** — PGD in `src/utils/evasion.py` project delta to the ε-ball but do not clamp `naturals + delta` to the valid embedding domain. Purification correctly clamps.

**Complex MPS requires PyTorch ≥ 2.1.0** — Adam has a `foreach` bug with complex-typed parameters in older versions, causing NaN updates.

**Purification broken when amplitudes overflow** — `purification.py:258` uses `abs_square` to compute Gibbs sampling weights; `draw_from_grid` maps `posinf → 0.0`, so overflow candidates are silently zeroed and sampling is wrong. Not yet fixed.
