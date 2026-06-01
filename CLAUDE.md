# bm4tc — Claude Code Guide

## Environment

```bash
conda activate bm4tc   # or use full path: /home/martin-lanigo/projects/miniconda3/envs/bm4tc/bin/python
```

Always invoke Python as a module from the project root:
```bash
/home/martin-lanigo/projects/miniconda3/envs/bm4tc/bin/python -m experiments.train ...
```

Never use `conda run`. Never use relative `python` without the full env path when running experiments.

## Running experiments

```bash
# Single run (NLL discriminative, alpha=0)
python -m experiments.train +experiments=moons/nat/fourier/d4r3/hpo_a0 tracking.mode=disabled

# Multirun seed sweep (NLL generative, alpha=1)
python -m experiments.train --multirun +experiments=circles/nat/legendre/d10r6/seed_sweep_a1

# Adversarial training seed sweep
python -m experiments.train --multirun +experiments=circles/at/legendre/d10r6/seed_sweep

# Test run (tiny config, no W&B)
python -m experiments.train +experiments=tests/nll         tracking.mode=disabled
python -m experiments.train +experiments=tests/adversarial tracking.mode=disabled

# Debug: print resolved config without running
python -m experiments.train --cfg job +experiments=tests/nll
```

## Running tests

```bash
pytest tests/ -x -q              # all tests, stop on first failure
pytest -m "not slow" -q          # unit tests only (~90, fast)
pytest tests/integration/ -q     # integration tests (~49, slower)
```

## Project structure

```
src/
  model.py        ConditionalBornMachine
  datahandler.py  DataHandler (load, split_and_rescale, DatasetConfig, DataGenDowConfig)
  train/          NLLTrainer (nll.py), AdversarialTrainer (adversarial.py)
  analysis/       viz.py, purification.py, mia.py, uq.py  (pure Python/matplotlib, no W&B)
  utils/          embeddings.py, evasion.py, train.py (OptimizerConfig, eval functions)

experiments/
  train.py        unified entry point: NLL (alpha=0 dis, alpha>0 gen) + adversarial
  batch.py        batch-queue runner for experiment configs (--type, --embedding, …)
  resolvers.py    OmegaConf custom resolvers (training_regime, geom_lr, dtype_suffix, …)
  config.py       Config, TrainerConfig, TrackingConfig dataclasses + register()
  tracking.py     make_logger, init_wandb, log_dataset_viz

configs/
  born/{embedding}/d{d}r{r}.yaml   MPS arch configs (d=physical dim, r=bond dim)
  dataset/                          2Dtoy/, mnist/, ucr_ts/
  trainer/nll/                      NLLConfig variants
  trainer/adversarial/              AdversarialConfig variants (pgd_at, trades)
  experiments/                      full experiment override configs

analysis/
  run.py               single-model analysis entry point
  sweep.py             seed_sweep / alpha_curve post-hoc analysis
  hpo.py               HPO result exploration
  batch.py             batch-queue runner for sweep.py
  utils/               statistics.py, resolve.py, wandb_fetcher.py, mia_utils.py
  visualize/           distribution plots + batch.py

tools/
  fill_hpo.py          patch seed_sweep configs from HPO best run (W&B or local)
  delete_runs.py       delete local outputs, W&B runs/artifacts, analysis dirs
  migrate_configs.py   one-off config layout migration (historical)
  alpha_lr_interp.py   geom-interp LR patcher for alpha_curve configs (historical)
  fetcher.ipynb        ad-hoc W&B data fetching notebook
```

## Naming conventions

- **Arch**: `d{d}r{r}` — `d4r3`, `d10r6`, `d30r18` (d=in_dim, r=bond_dim)
- **Trainer tokens**: `nat` (natural NLL training, any alpha) | `at` (adversarial training)
- **Alpha in kind suffix**: `_a0` (α=0 discriminative), `_a1` (α=1 generative), `_a05` (α=0.5), `_a01`, …
- **Metrics**: `dis_loss`, `gen_loss`, `acc`, `rob`
- **Datasets (2D toy)**: `circles`, `moons`, `spirals` (+ `_small` variants); the `_4k` suffix was dropped in Phase 7

## Output structure

```
outputs/{dataset}/{nat|at}/{embedding}/{arch}/{kind}_{DDMM}/
  0/.hydra/config.yaml    resolved config
  0/models/model          checkpoint (tensorkrowch format)
  1/...
analysis/outputs/{dataset}/{nat|at}/{embedding}/{arch}/{kind}_{DDMM}/
  evaluation_data.csv     one row per run, all metrics
```

Examples:
- `outputs/circles/nat/legendre/d10r6/seed_sweep_a0_2804/`
- `outputs/moons/at/legendre/d10r6/seed_sweep_2804/`

## Config dataclasses

Each module owns its config:
- `NLLConfig`, `NormControlConfig` in `src/train/nll.py`
- `AdversarialConfig` in `src/train/adversarial.py`
- `CBMConfig`, `MPSInitConfig` in `src/model.py`
- `DatasetConfig`, `DataGenDowConfig` in `src/datahandler.py`
- `OptimizerConfig` in `src/utils/train.py`
- `EvasionConfig` in `src/utils/evasion.py`
- `PurificationConfig` in `src/analysis/purification.py`
- `Config`, `TrainerConfig`, `TrackingConfig` in `experiments/config.py`

## Key invariants

- `DataHandler.split_and_rescale(cbm)` uses `cbm.input_range` (derived from embedding at load time) — always correct.
- Eval functions (`eval_metrics`, `eval_rob` in `src/utils/train.py`) call `cbm.eval()` themselves.
- Attack epsilons in `analysis/sweep.py` are fractions of the embedding range size, not absolute values.
