# bm4tc — Claude Code Guide

## Environment

```bash
conda activate bm4tc   # or use full path: /home/martin-lanigo/projects/miniconda3/envs/bm4tc/bin/python
```

Always invoke Python as a module from the project root:
```bash
/home/martin-lanigo/projects/miniconda3/envs/bm4tc/bin/python -m experiments.nll ...
```

Never use `conda run`. Never use relative `python` without the full env path when running experiments.

## Running experiments

```bash
# Single run (NLL discriminative)
python -m experiments.nll +experiments=nll/dis/fourier/d4r3/hpo/moons tracking.mode=disabled

# Multirun seed sweep (NLL generative)
python -m experiments.nll --multirun +experiments=nll/gen/legendre/d10r6/seed_sweep/circles

# Test run (tiny config, no W&B)
python -m experiments.nll         +experiments=tests/nll         tracking.mode=disabled
python -m experiments.adversarial +experiments=tests/adversarial tracking.mode=disabled

# Debug: print resolved config without running
python -m experiments.nll --cfg job +experiments=tests/nll
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
  models/         ConditionalBornMachine (cbm.py)
  trainer/        NLLTrainer (nll.py), AdversarialTrainer (adversarial.py), SoftmaxTrainer (softmax.py), utils.py
  analysis/       viz.py, purification.py, mia.py, uq.py  (pure Python/matplotlib, no W&B)
  data/           DataHandler (handler.py), gen_n_load.py
  utils/          embeddings.py, evasion.py, get.py, resolvers.py

experiments/
  nll.py          entry point: NLL on p(c|x) with alpha=0, p(x,c) with alpha>0
  adversarial.py  entry point: classification pretraining + PGD-AT/TRADES
  softmax_sanity.py   sanity check: raw amplitudes as softmax logits
  config.py           Config, TrainerConfig, TrackingConfig dataclasses + register()
  tracking.py         make_logger, init_wandb, log_dataset_viz

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
  queue.py             batch-queue runner for sweep.py
  configs/             pipeline tool scripts (fill_hpo.py, delete_runs.py, alpha_lr_interp.py)
  utils/               statistics.py, resolve.py, wandb_fetcher.py, mia_utils.py
  visualize/           distribution plots + queue.py
```

## Naming conventions

- **Arch**: `d{d}r{r}` — `d4r3`, `d10r6`, `d30r18` (d=in_dim, r=bond_dim)
- **Regime codes**: `dis` (discriminative), `gen` (generative), `adv` (adversarial)
- **Metrics**: `dis_loss`, `gen_loss`, `acc`, `rob`
- **Datasets (2D toy)**: `circles`, `moons`, `spirals` (+ `_small` variants); the `_4k` suffix was dropped in Phase 7

## Output structure

```
outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{DDMM}/
  0/.hydra/config.yaml    resolved config
  0/models/model.pt       checkpoint
  1/...
analysis/outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{DDMM}/
  evaluation_data.csv     one row per run, all metrics
```

## Config dataclasses

Each module owns its config:
- `NLLConfig`, `NormControlConfig` in `src/trainer/nll.py`
- `AdversarialConfig` in `src/trainer/adversarial.py`
- `CBMConfig`, `MPSInitConfig` in `src/models/cbm.py`
- `DatasetConfig`, `DataGenDowConfig` in `src/data/gen_n_load.py`
- `OptimizerConfig`, `CriterionConfig` in `src/utils/get.py`
- `EvasionConfig` in `src/utils/evasion.py`
- `PurificationConfig` in `src/analysis/purification.py`
- `Config`, `TrainerConfig`, `TrackingConfig` in `experiments/config.py`

## Key invariants

- `DataHandler.split_and_rescale(cbm)` uses `cbm.input_range` (derived from embedding at load time) — always correct.
- Eval functions (`eval_metrics`, `eval_rob` in `src/trainer/utils.py`) call `cbm.eval()` themselves.
- Attack epsilons in `seed_sweep_analysis.py` are fractions of the embedding range size, not absolute values.
