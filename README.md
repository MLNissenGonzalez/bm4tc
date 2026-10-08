# bm4tc: Born machines for trustworthy classification

Code for *Robust Classification and Purification with a Single Tensor Network Born
Machine* (TPM 2026) and its journal extension.

A matrix product state (MPS) Born machine models the joint density p(x, c) =
|ψ(x, c)|² / Z of inputs and classes. Its normalizer Z is exact and cheap, so the same
trained model gives a classifier p(c | x), an exact log-likelihood log p(x), and
gradients of both. The code trains such models on a continuum between discriminative
and generative training,

    L = (1 − β) · L_dis + β · L_gen / N,     L_dis = −log p(c | x),  L_gen = −log p(x, c),

with N = n + 1 the number of modelled variables (n features and the class), so every
loss is in nats per variable, optionally with adversarial training, and evaluates what the density buys against
adversarial examples: detection (flag inputs with low log p(x)), likelihood
purification (gradient ascent on log p(x) within a small ball) and Gibbs purification
(resampling features from the model's conditionals). A joint energy-based model (JEM,
Grathwohl et al. 2020) with a parameter-matched MLP is the baseline: same objective,
same attacks, same defences, but an intractable normalizer handled by SGLD.

## Install

```bash
conda env create -f environment.yml    # env "bm4tc": Python 3.10, torch 2.1, tensorkrowch 1.1.6
conda activate bm4tc
```

Outputs and cached datasets go under `$BM4TC_DATA_ROOT` (default: the repository
root). Training curves go to Weights & Biases (`wandb login` once, or
`WANDB_API_KEY`; a study turns it off with `tracking.mode: disabled`); every run also
writes them to `log.json`.

## Quick start

A study is a grid of runs (dataset × embedding × architecture × β [× attack radius])
times seeds, in `configs/studies/<study>.yaml`. One command trains and analyses it:

```bash
python -m bm4tc run tests/seam_nat       # a tiny spirals study, a minute on a CPU
python -m bm4tc status tests/seam_nat    # what is done, running, failed
```

`run` chains the stages `hpo → select → train → analyse`, each also a verb of its own;
everything resumes after an interruption and skips finished work. The result is
`outputs/<study>/results.csv`: one row per seed run, with its identity, selected
hyperparameters and test metrics (clean and robust accuracy, detection, purification
at every attack radius).

```bash
python -m bm4tc run mnist12_nat --gpus 0,1 --per-gpu 2   # four runs in parallel
python -m bm4tc run mnist12_at                           # adversarial training, warm from mnist12_nat
python -m bm4tc figures paper                            # every figure and table -> figures/journal/
```

## Reproducing the paper

`configs/papers/paper.yaml` maps each figure and table of the paper to the studies it
reads; `python -m bm4tc figures paper` draws them all from the studies' results (and,
for sample and density plots, from one kept checkpoint per model). The studies, the
order to run them in and the pilots that come first are listed in
[docs/plan.md](docs/plan.md).

## Repository

```
bm4tc/
  core/        the models and their training: MPS Born machine, embeddings, objective,
               Trainer, PGD attacks, JEM (model, SGLD sampler, trainer, purifier)
  analysis/    evaluating a trained model: detection, purification, robustness ceiling,
               membership inference
  pipeline/    studies, runs, stages, job pool, metric keys, figures; the CLI is
               bm4tc/__main__.py
configs/       defaults.yaml, studies/, papers/, datasets, trainer presets, hparams/
tests/         unit, integration and end-to-end (seam) tests; a training benchmark
docs/          design decisions, the remaining plan, design notes
figures/       generated paper figures and tables
```

[GUIDE.md](GUIDE.md) explains the concepts, the pipeline and the code in depth.

```bash
pytest -q                    # the full suite, ~3 min
pytest -q -m "not slow"      # without model training
```
