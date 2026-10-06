# Experiments Guide

The pipeline weaving: configs, the training entry point, logging and metric keys. Run
everything as a Python module from the project root.

| File | Purpose |
|------|---------|
| `train.py` | Trains a study's jobs (NAT and AT); `train(job)` returns the best `objective/valid` |
| `runs.py` | Studies, grid cells, jobs, run names and dirs, `run.json`, warm-start lookup (D14, D17, D19) |
| `config.py` | The run-config schema (`Config`, `TrackingConfig`) and `register()` |
| `metrics.py` | The only place that builds logged metric keys (`quantity/split[/budget]`, D48) |
| `tracking.py` | `log.json` writer and W&B init |

---

## `train.py`

One `Trainer` for both regimes: AT is a run with an attack (`trainer.evasion` set,
as in the `at` preset), NAT a run without (`evasion: null`). The best epoch is the one
with the lowest `objective/valid` (D8); `train.py` returns that value, so an Optuna
sweep always uses `direction: minimize`.

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

`runs.Job.compose()` builds a job's config with the Hydra compose API: `config.yaml`
with the study's dataset and regime, then the cell, the embedding settings, the
study's `config:`, the selected hparams and the seed, key by key on the schema (D25).

---

## Output directory

```
outputs/{study}/{embedding}/{arch}/a{alpha}[/eps{eps}]/s{seed}/
    run.json               # written last: identity, init source, git sha, launch time,
                           # result, resolved config and its hash (D17)
    models/model           # checkpoint
    log.json               # epoch metrics, keys from metrics.py
    train.log
outputs/{study}/.replaced/{date}/...   # runs moved aside by --replace, deleted only by prune
```

A run is finished when it holds `run.json`. Relaunching skips finished runs with the
same config hash, restarts unfinished ones, and refuses to touch a finished run whose
config changed unless `--replace` (see `runs.Job.claim`). Nothing parses these paths:
runs are found by their `run.json` (`runs.find_runs`).
