# Experiments Guide

The pipeline weaving: configs, the training entry point, logging and metric keys. Run
everything as a Python module from the project root.

| File | Purpose |
|------|---------|
| `train.py` | Hydra entry point for NAT and AT training; returns the best `objective/valid` |
| `config.py` | Top-level schema (`Config`, `TrainerConfig`, `TrackingConfig`) and `register()` |
| `metrics.py` | The only place that builds logged metric keys (`quantity/split[/budget]`, D48) |
| `tracking.py` | `log.json` writer and W&B init |
| `resolvers.py` | OmegaConf resolvers used in the output-path templates |

---

## `train.py`

The regime follows from the trainer group that is set: `trainer/at` → AT
(`AdversarialTrainer`), `trainer/nat` → NAT (`NLLTrainer`). The best epoch is the one
with the lowest `objective/valid` (D8); `train.py` returns that value, so an Optuna
sweep always uses `direction: minimize`.

```bash
# NAT, cold start
python -m experiments.train dataset=2Dtoy/spirals born=legendre/d10r6c64 \
    trainer/nat=default trainer.nat.alpha=0.0

# AT, warm-started from a NAT checkpoint
python -m experiments.train dataset=2Dtoy/spirals born=legendre/d10r6c64 \
    '~trainer/nat' trainer/at=pgd_at trainer.at.alpha=0.01 model_path=<run>/models/model

# Quick checks, no W&B
python -m experiments.train +experiments=tests/nat tracking.mode=disabled
python -m experiments.train +experiments=tests/at  tracking.mode=disabled

# Print the composed config without running
python -m experiments.train --cfg job +experiments=tests/nat
```

The config is checked against the schema (D25): a typo in a key fails at composition.
The per-experiment YAML tree was deleted (D49); Phase 4 study files replace it.

---

## Output directory

```
outputs/{dataset}/{nat|at}/{embedding}/{arch}/[{stage}/]{experiment}_{DDMM_HHMM}/
    .hydra/config.yaml     # resolved config
    models/model           # checkpoint (if save=true)
    log.json               # epoch metrics, keys from metrics.py
```

A multirun adds one numbered subdirectory per job. Training needs an `outputs/`
ancestor in the run directory, because the W&B group is derived from the path (D47
replaces this in Phase 4).
