# Guide

How bm4tc works: the model and the objective, the vocabulary, the pipeline from a
study file to a paper figure, and where in the code each piece lives. The reasons
behind the design are numbered decisions (D1, D2, ...) in
[docs/decisions.md](docs/decisions.md); they are cited here where they matter.

## 1. The model

**MPS Born machine** (`bm4tc/core/model.py`, `ConditionalBornMachine`, a
`tensorkrowch` MPS). Each input feature x_i is embedded into a d-dimensional vector
φ(x_i) (`bm4tc/core/embeddings.py`: legendre, fourier, hermite, chebychev1/2). An MPS
of bond dimension r, with one extra site for the class, contracts the embedded input
into amplitudes ψ(x, c) (complex64 by default, D4). The Born rule gives

    p(x, c) = |ψ(x, c)|² / Z,   Z = Σ_c ∫ |ψ(x, c)|² dx   (exact, by contracting the MPS with itself)

so p(c | x), log p(x) and their gradients are all exact. Long chains (MNIST: 145
sites) overflow float32, so contractions run in the log domain with per-site norm
accumulation (`accumulate: true`), and `log Z` is computed by a zip-up contraction.
These caches belong to the model, keyed on the parameters' versions (D31, D55). This
numerical core is not touched outside a dedicated track (D30, docs/plan.md Phase 9).

| Embedding | Input range | Note |
|---|---|---|
| `legendre` (default) | [−1, 1] | orthonormal Legendre polynomials |
| `fourier` | [0, 1] | tensorkrowch's |
| `hermite` | [−4, 4] | Hermite functions; `init_method: canonical` |
| `chebychev1` | [−0.99, 0.99] | the weight diverges at ±1 |
| `chebychev2` | [−0.99, 0.99] | φ(±1) = 0 would be unreachable |

Data are rescaled to the embedding's input range (JEM: [−1, 1], no embedding).

**Architecture names:** `d3r40` = embedding dimension d = 3, bond dimension r = 40.

**The model interface** (`bm4tc/core/interface.py`, D69). Everything after training
sees a model only through `log_joint(x)` = log p(x, c) + C (shape B × K,
differentiable in x), `log_normalizer()` = C, `input_range`, `out_dim` and `reset()`.
`class_probabilities`, `log_px`, the PGD loss and the joint-attack loss are defined
once on top. The MPS gives C = log Z exactly; JEM gives C = 0, so its log p(x) is
unnormalised (fine for thresholds and purification, not comparable across models).

**JEM baseline** (`bm4tc/core/jem/`, D35, D70–D74). A two-hidden-layer MLP whose
logits are log p(x, c) up to a constant. Its parameter count is matched to an MPS:
a JEM study's `arch: d3r40` means "the MLP whose real parameter count is nearest to
the real degrees of freedom of the complex d3r40 MPS". It trains with the same
objective; L_gen is contrastive, with SGLD negatives from a replay buffer. Its own
defence is SGLD purification, the counterpart of the MPS's Gibbs purification.

## 2. Training

One objective for both regimes and both models (`bm4tc/core/objective.py`,
`bm4tc/core/train.py`, D15, D18, D71):

    L = (1 − α) · [(1 − cw) · L_dis(x_adv) + cw · L_dis(x)] + α · L_gen(x)

- **NAT** (natural training): no attack, so L = (1 − α) · L_dis(x) + α · L_gen(x).
  α = 0 is a plain discriminative classifier, α = 1 a pure density model.
- **AT** (adversarial training): x_adv from PGD at radius ε (`trainer.evasion`);
  cw = `trainer.clean_weight`. The generative term always sees clean data. Training
  it on adversarial points would teach p(x) to like them, which defeats detection
  and purification (D18; kept in mind as an ablation). AT runs start from the
  selected α = 0 NAT run of the same cell and seed (warm start, D19). The radius ramps
  up over the first 70% of epochs (curriculum, D40, D61).
- **Norm control** (MPS): a soft penalty strength · (log Z − target)² keeps log Z
  near a target, so the amplitudes stay in float range (`trainer.norm_control`;
  optional hard rescaling every k steps).
- **Selection** (D8): every `eval_every` epochs the same objective is evaluated on the
  validation split (AT: on a fixed attacked subset). The epoch with the lowest
  `objective/valid` is kept. `patience` counts validations. HPO and the choice
  between trials use the same quantity. There is no other selection rule.

## 3. Vocabulary

| Term | Meaning |
|---|---|
| `nat`, `at` | the regimes above (D10) |
| α | weight of the generative term |
| ε, δ, budget | attack radius ε, purification radius δ: **relative**, a fraction of the input range (legendre: [−1, 1], so ε = 0.1 is 0.2 in input units). One grid, `budgets` in `configs/defaults.yaml`, serves AT training, attacks, detection and purification (D3) |
| study | `configs/studies/<name>.yaml`: a dataset, a regime, a model, a grid, seeds, an HPO space |
| cell | one grid point: `legendre/d3r40/a0.01[/eps0.1]` |
| run | one (cell, seed): `outputs/<study>/<cell>/s<seed>/` |
| paper | `configs/papers/<name>.yaml`: figures and tables drawn from studies |

Names hold identity axes only, as prefix + value (`d3r40`, `a0.01`, `eps0.1`, `s3`).
Learning rates and other hyperparameters never appear in names (D14).

## 4. Code map

```
bm4tc/core/        the maths; no files, no configs
  model.py           ConditionalBornMachine (MPS): amplitudes, log Z, sampling, Gibbs conditionals
  embeddings.py      feature maps; budget conversion (rel_to_abs, fmt_budget)
  interface.py       the model interface for analysis (D69)
  objective.py       losses, mixed objective, norm control, evaluate()
  train.py           Trainer (MPS, NAT and AT)
  attacks.py         PGD and the joint (class + density) PGD
  jem/               JEM: model, SGLD sampler, JEMTrainer, SGLD purifier
bm4tc/analysis/    evaluating a trained model -> numbers; imports only core
  uq.py              attacks at every budget, detection, purification (UQEvaluation)
  purification.py    likelihood (gradient) and Gibbs purification
  ceiling.py         data-only upper bound on robust accuracy (two-class 2-D data)
  privacy.py         membership inference, standalone, not in the pipeline (D1)
bm4tc/pipeline/    configs, files, runs
  config.py          the run-config schema (D25)
  data.py            datasets (generated, MNIST, UCR), splits, rescaling to a model's input range
  runs.py            studies, cells, jobs, run dirs, run.json, warm-start lookup
  stages.py          hpo, select, train, analyse, collect, prune, status
  analyse.py         the analysis parts of one run
  executor.py        the job pool (GPU slots, dependencies, logs)
  metrics.py         metric keys: the only place they are built (D48, D64)
  tracking.py        log.json and W&B
  paths.py           BM4TC_DATA_ROOT
  figures/           paper figures and tables (D75)
bm4tc/__main__.py  the CLI
```

**Import rule** (D32, enforced by `tests/unit/test_import_rule.py`): `core` imports
neither `analysis` nor `pipeline`; `analysis` imports only `core`.

## 5. Configuration

```
configs/
  config.yaml        the run config: picks dataset, trainer preset, tracking
  defaults.yaml      study-level defaults: seeds (5), α ladder, AT radius, HPO space,
                     budgets, the analysis settings
  studies/           one file per study; studies/tests/ holds the seam studies
  hparams/           <study>.yaml, written by `select` only (D34)
  papers/            paper manifests; papers/tests/ for the figures test
  dataset/           spirals, mnist12 (12x12), mnist (28x28), UCR time series, ...
  trainer/           nat.yaml, at.yaml (the PGD-AT preset)
  tracking/          online.yaml, disabled.yaml
```

Run-level constants (model, trainer) are dataclass defaults in the schema
(`bm4tc/pipeline/config.py`, `TrainConfig` in `core/train.py`, `CBMConfig` in
`core/model.py`, `JEMConfig` in `core/jem/train.py`). An unknown key anywhere fails
at composition (D25).

**A study file** (merged onto `defaults.yaml`; unknown keys fail):

```yaml
dataset: mnist12            # a configs/dataset/ option
regime: at                  # nat | at
model: mps                  # mps | jem (JEM: grid.embedding [raw])
init: warm                  # cold | warm; AT is always warm
warm_from: mnist12_nat      # default [jem_]{dataset}_nat
grid:                       # each axis overrides the default
  arch: [d3r20, d3r40]
  alpha: [0, 1e-2]
  eps: [0.1]                # AT only; must be in budgets
config:                     # fixed run-config values, any schema key
  trainer.max_epoch: 100
hpo:                        # null: no HPO, the study fixes every hparam
  n_trials: 20
  space:
    trainer.optimizer.kwargs.lr: {log: [1e-5, 1e-1]}
    trainer.clean_weight: [0.0, 1.0]
hparams_from: {study: jem_mnist12_nat, params: [jem.sampler.step_size]}   # optional (D74)
analysis:
  sweep_purify: {enabled: true}
```

Each run's config is composed with the Hydra compose API in this order: the cell,
the embedding's settings, `config:`, inherited hparams, the selected hparams, the
seed. The result and its hash go into `run.json`.

## 6. The pipeline

```bash
python -m bm4tc hpo     <study>   # Optuna per cell, trials on the first seed, no checkpoints
python -m bm4tc select  <study>   # best trial per cell (argmin objective/valid) -> configs/hparams/<study>.yaml
python -m bm4tc train   <study>   # every (cell, seed) with the selected hparams
python -m bm4tc analyse <study>   # every run on test -> analysis.json; the study's results.csv
python -m bm4tc run     <study>   # all four in order, skipping what is done
python -m bm4tc status  <study>   # per cell: trials, selected, trained, analysed; failed units with logs
python -m bm4tc prune   <study> --keep-one | --all | --old [--yes]
python -m bm4tc figures <paper> [--item NAME]
```

Options: `--cell legendre/d3r40/a0.01` and `--seed 3` (repeatable) narrow a stage;
`--gpus 0,1 --per-gpu 2` runs units in parallel (each pinned to a GPU via
`CUDA_VISIBLE_DEVICES`); `--replace` archives finished results whose config changed
and redoes them.

**Execution** (D38, D66). Each unit (an HPO worker, one run's training, one run's
analysis) is a subprocess with its own log in `outputs/<study>/.logs/`. Units start
once their dependencies succeed. A failed unit skips what depends on it and the rest
goes on. Only one launch per study runs at a time (a lock file). Every stage resumes
after a crash or a closed terminal, because finished work is recognised on disk:

- **HPO**: one Optuna study per cell in `{cell}/hpo/journal.log`. A relaunch continues
  until `n_trials` are finished (D62).
- **train**: a run is finished when it has `run.json`. A relaunch skips it if its
  config hash is unchanged, restarts it if it never finished, and refuses to touch it
  if the config changed, unless `--replace` moves the old run to
  `outputs/<study>/.replaced/<date>/` (D58).
- **analyse**: each analysis part is cached in `analysis.json` under a hash of its
  settings, so changing one setting recomputes only the affected part (D63).

**Warm studies** (AT, JEM-AT) need their `warm_from` study trained first; launch them
in that order. The warm-start source of every run is recorded in its `run.json`.

**Run directory:**

```
outputs/<study>/<embedding>/<arch>/a<alpha>[/eps<eps>]/
  hpo/journal.log, hpo/t<n>/       trials (curves only)
  s<seed>/
    run.json                       identity, warm start, git sha, times, result, config + hash
    models/model                   the selected checkpoint (prune deletes these)
    log.json                       per-epoch metrics
    analysis.json                  analysis parts, each with its settings hash
outputs/<study>/results.csv        one row per analysed run
outputs/<study>/analysis.yaml      the analysis settings behind it
outputs/<study>/.logs/             one log per unit, status.json
```

**Analysis parts** (`bm4tc/pipeline/analyse.py`, on the test split; settings in
`defaults.yaml` → `analysis`):

| Part | Computes |
|---|---|
| `clean` | accuracy, losses, the robust-accuracy ceiling (2-D data) |
| `uq` | PGD-40 at every budget; detection at clean-percentile thresholds q (calibrated on valid, D2); likelihood purification at radius δ |
| `uq_joint` | the same under the joint attack on class and log p(x), the adaptive attack (if `joint_attack`) |
| `gibbs` / `sgld` | Gibbs (MPS) or SGLD (JEM) purification of the PGD examples, per number of sweeps (if `sweep_purify.enabled`) |

**Metric keys** (`quantity/split[/budget][/setting]`) are documented in
`bm4tc/pipeline/metrics.py`: e.g. `acc/test`, `rob/test/0.1`, `detect/test/0.1/q5`,
`purify/test/0.1/d0.1`, `purify_gibbs/test/0.1/k6`, `rob_joint/test/0.1`. Budget 0
means clean data (`detect/test/0/q5` is the false-positive rate).

**Prune** (D67) deletes checkpoints only: `--all`, or `--keep-one` (all but the first
seed's, which the figures and warm starts use), or `--old` (the `.replaced/` archive).
Results, curves and logs stay. It asks first unless `--yes`.

## 7. Figures

`python -m bm4tc figures paper` draws each item of `configs/papers/paper.yaml` into
`figures/journal/` (committed). An item names a kind and its models:

```yaml
mnist_headline:
  kind: table
  eps: 0.1                                  # fills {eps}; a list gives one output per budget
  models:                                   # each must select exactly one grid cell
    - {study: mnist_nat, where: {alpha: 0.01}, label: "MPS $\\alpha=0.01$"}
    - {study: jem_mnist_nat, where: {alpha: 0.01}, label: "JEM $\\alpha=0.01$"}
  metrics:
    - {key: acc/test, label: Clean, best: max}
    - {key: "purify/test/{eps}/d0.1", minus: "rob/test/{eps}", label: Gain}
```

| Kind | Draws | From |
|---|---|---|
| `curve` | a metric against `x`: `alpha`, `bond_dim`, or `{x}` inside the key (ε, sweeps) | results.csv |
| `bars` | models × metrics at one budget | results.csv |
| `coverage` | accuracy on passed examples vs fraction passed, over q | results.csv |
| `table` | models × metrics, mean ± std, booktabs `.tex` | results.csv |
| `density` | p(c \| x) and p(x) over the input square (2-D data) | checkpoint |
| `samples` | class-conditional samples (images: per-class means) | checkpoint |
| `transfer` | PGD examples crafted on one model, purified by each | checkpoints |

Values are the mean and std over the cell's seeds. Checkpoint kinds use the study's
first seed. An item that fails (e.g. its study has not run) is reported and the
others are still drawn. The fields of each kind are in the docstrings of
`bm4tc/pipeline/figures/`.

## 8. Tests

```bash
pytest -q                     # everything, ~3 min (CPU and GPU)
pytest -q -m "not slow"       # without training
pytest tests/e2e -q           # the seams
python -m tests.bench.bench_train_step   # ms per training step, NAT and AT (not a test)
```

The **seam tests** (`tests/e2e/`) run tiny spirals studies (`configs/studies/tests/
seam_*`) through the CLI and pin their numbers: end-point metrics, and per-epoch
training curves to 1e-6. A refactor must keep them, or change a number on purpose in
its own commit, saying why (D26, D51). The seam studies run once per session and are
shared by the MPS seam, the JEM seam and the figures test.

## 9. Gotchas

- `tqdm` around a DataLoader (even disabled) and every extra pass over a loader
  consume a torch RNG draw; adding one moves every RNG-dependent number after it.
- A config field named like a dict method (`keys`, `items`) is unreachable as an
  attribute in OmegaConf.
- Generated datasets are cached in `$BM4TC_DATA_ROOT/.datasets/` under a name with
  every generation setting (D76); delete the file to regenerate it.
- `randn_eye` initialisation with a non-Fourier embedding rescales the tensors by
  1/φ₀(0), otherwise the initial amplitude underflows on long chains; use
  `init_method: canonical` for Hermite (set in `defaults.yaml`).
- Complex parameters need torch ≥ 2.1 (an Adam `foreach` bug produced NaNs before).
