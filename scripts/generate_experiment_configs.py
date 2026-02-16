"""Generate experiment configs for architecture x embedding grid.

Creates HPO and seed_sweep configs for:
- 4 architectures: d2D2, d3D3, d4D3, d10D6
- 2 embeddings: fourier, legendre
- 3 trainers: classification, generative, adversarial (PGD-AT)
- 3 datasets: circles_4k, moons_4k, spirals_4k

Skips files that already exist.
"""

import os
from pathlib import Path

BASE = Path("configs/experiments")
DATASETS = ["circles", "moons", "spirals"]
ARCHITECTURES = [
    {"d": 2, "D": 2},
    {"d": 3, "D": 3},
    {"d": 4, "D": 3},
    {"d": 10, "D": 6},
]
EMBEDDINGS = ["fourier", "legendre"]


def cls_hpo(emb: str, d: int, D: int, dataset: str) -> str:
    return f"""\
# @package _global_
experiment: size_hpo

defaults:
  - override /born: {emb}/d{d}D{D}
  - override /dataset: 2Dtoy/{dataset}_4k
  - override /trainer/ganstyle: null
  - override /trainer/adversarial: null
  - override /tracking: online
  - override /hydra/sweeper: optuna

tracking:
  seed: 42
dataset:
  split_seed: 11
  gen_dow_kwargs:
    seed: 25

trainer:
  classification:
    max_epoch: 300
    batch_size: 256
    patience: 300
    stop_crit: "clsloss"
    watch_freq: 10000
    save: false
    optimizer:
      name: "adam"
      kwargs:
        lr: 1e-4
        weight_decay: 0.0
    criterion:
      name: "negative log-likelihood"
      kwargs:
        eps: 1e-8
    metrics: {{"clsloss": 1}}

hydra:
  sweeper:
    study_name: "${{experiment}}_${{dataset.name}}"
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10
    direction: minimize
    n_trials: 30
    n_jobs: 1
    params:
      trainer.classification.optimizer.kwargs.lr: tag(log, interval(1e-4, 5e-2))
      trainer.classification.optimizer.kwargs.weight_decay: tag(log, interval(1e-9, 1e-1))
"""


def gen_hpo(emb: str, d: int, D: int, dataset: str) -> str:
    return f"""\
# @package _global_
experiment: size_hpo

defaults:
  - override /born: {emb}/d{d}D{D}
  - override /dataset: 2Dtoy/{dataset}_4k
  - override /trainer/classification: null
  - override /trainer/generative: test
  - override /trainer/ganstyle: null
  - override /trainer/adversarial: null
  - override /tracking: online
  - override /hydra/sweeper: optuna

tracking:
  seed: 42
dataset:
  split_seed: 11
  gen_dow_kwargs:
    seed: 25

trainer:
  generative:
    max_epoch: 300
    batch_size: 256
    patience: 300
    stop_crit: "genloss"
    watch_freq: 10000
    save: false
    optimizer:
      name: "adam"
      kwargs:
        lr: 1e-4
        weight_decay: 0.0
    criterion:
      name: "nll"
      kwargs:
        eps: 1e-8
    metrics: {{"genloss": 1}}

hydra:
  sweeper:
    study_name: "${{experiment}}_${{dataset.name}}"
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10
    direction: minimize
    n_trials: 30
    n_jobs: 1
    params:
      trainer.generative.optimizer.kwargs.lr: tag(log, interval(1e-4, 5e-2))
      trainer.generative.optimizer.kwargs.weight_decay: tag(log, interval(1e-9, 1e-1))
"""


def adv_hpo(emb: str, d: int, D: int, dataset: str) -> str:
    return f"""\
# @package _global_
experiment: hpo

defaults:
  - override /born: {emb}/d{d}D{D}
  - override /dataset: 2Dtoy/{dataset}_4k
  - override /trainer/classification: null
  - override /trainer/adversarial: pgd_at
  - override /trainer/ganstyle: null
  - override /trainer/generative: null
  - override /tracking: online
  - override /hydra/sweeper: optuna

model_path: ???

dataset:
  split_seed: 11
  gen_dow_kwargs:
    seed: 25

tracking:
  seed: 42
  evasion:
    method: "PGD"
    norm: "inf"
    criterion:
      name: "negative log-likelihood"
      kwargs:
        eps: 1e-8
    strengths: [0.15]
    num_steps: 10
    step_size: null
    random_start: true

trainer:
  adversarial:
    max_epoch: 300
    batch_size: 256
    method: "pgd_at"
    stop_crit: "rob"
    patience: 300
    watch_freq: 10000
    metrics: {{"rob": 1, "clsloss": null, "acc": null}}

    evasion:
      method: "PGD"
      norm: "inf"
      criterion:
        name: "negative log-likelihood"
        kwargs:
          eps: 1e-8
      strengths: [0.15]
      num_steps: 10
      step_size: null
      random_start: true

    curriculum: true
    curriculum_start: 0.01
    curriculum_end_epoch: 200

    save: false
    auto_stack: true
    auto_unbind: false

hydra:
  sweeper:
    storage: null
    study_name: "${{experiment}}_${{dataset.name}}"
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10
    direction: minimize
    n_trials: 30
    n_jobs: 1
    params:
      trainer.adversarial.optimizer.kwargs.lr: tag(log, interval(1e-5, 1e-2))
      trainer.adversarial.optimizer.kwargs.weight_decay: tag(log, interval(1e-9, 1e-1))
      trainer.adversarial.clean_weight: interval(0.2, 1.0)
"""


def cls_seed(emb: str, d: int, D: int, dataset: str) -> str:
    return f"""\
# @package _global_
experiment: seed_sweep

defaults:
  - override /born: {emb}/d{d}D{D}
  - override /dataset: 2Dtoy/{dataset}_4k
  - override /trainer/ganstyle: null
  - override /trainer/adversarial: null
  - override /tracking: online

tracking:
  seed: 42
dataset:
  split_seed: 11
  gen_dow_kwargs:
    seed: 25

trainer:
  classification:
    max_epoch: 300
    batch_size: 256
    patience: 300
    stop_crit: "clsloss"
    watch_freq: 1000
    save: true
    optimizer:
      name: "adam"
      kwargs:
        lr: ???    # FILL FROM HPO
        weight_decay: ???  # FILL FROM HPO
    criterion:
      name: "negative log-likelihood"
      kwargs:
        eps: 1e-8
    metrics: {{"clsloss": 1}}

hydra:
  sweeper:
    params:
      tracking.seed: range(1,21)
"""


def gen_seed(emb: str, d: int, D: int, dataset: str) -> str:
    return f"""\
# @package _global_
experiment: seed_sweep

defaults:
  - override /born: {emb}/d{d}D{D}
  - override /dataset: 2Dtoy/{dataset}_4k
  - override /trainer/classification: null
  - override /trainer/generative: test
  - override /trainer/ganstyle: null
  - override /trainer/adversarial: null
  - override /tracking: online

tracking:
  seed: 42
dataset:
  split_seed: 11
  gen_dow_kwargs:
    seed: 25

trainer:
  generative:
    max_epoch: 300
    batch_size: 256
    patience: 300
    stop_crit: "genloss"
    watch_freq: 1000
    save: true
    optimizer:
      name: "adam"
      kwargs:
        lr: ???    # FILL FROM HPO
        weight_decay: ???  # FILL FROM HPO
    criterion:
      name: "nll"
      kwargs:
        eps: 1e-8
    metrics: {{"genloss": 1}}

hydra:
  sweeper:
    params:
      tracking.seed: range(1,21)
"""


def adv_seed(emb: str, d: int, D: int, dataset: str) -> str:
    return f"""\
# @package _global_
experiment: seed_sweep

defaults:
  - override /born: {emb}/d{d}D{D}
  - override /dataset: 2Dtoy/{dataset}_4k
  - override /trainer/classification: null
  - override /trainer/adversarial: pgd_at
  - override /trainer/ganstyle: null
  - override /trainer/generative: null
  - override /tracking: online

model_path: ???

dataset:
  split_seed: 11
  gen_dow_kwargs:
    seed: 25

tracking:
  seed: 42
  evasion:
    method: "PGD"
    norm: "inf"
    criterion:
      name: "negative log-likelihood"
      kwargs:
        eps: 1e-8
    strengths: [0.15]
    num_steps: 10
    step_size: null
    random_start: true

trainer:
  adversarial:
    max_epoch: 300
    batch_size: 256
    method: "pgd_at"
    stop_crit: "rob"
    patience: 300
    watch_freq: 1000
    metrics: {{"rob": 1, "clsloss": null, "acc": null}}

    evasion:
      method: "PGD"
      norm: "inf"
      criterion:
        name: "negative log-likelihood"
        kwargs:
          eps: 1e-8
      strengths: [0.15]
      num_steps: 10
      step_size: null
      random_start: true

    curriculum: true
    curriculum_start: 0.01
    curriculum_end_epoch: 200

    save: true
    auto_stack: true
    auto_unbind: false

    optimizer:
      kwargs:
        lr: ???    # FILL FROM HPO
        weight_decay: ???  # FILL FROM HPO
    clean_weight: ???  # FILL FROM HPO

hydra:
  sweeper:
    params:
      tracking.seed: range(1,21)
"""


GENERATORS = {
    "classification": {
        "hpo": cls_hpo,
        "seed_sweeps": cls_seed,
    },
    "generative": {
        "hpo": gen_hpo,
        "seed_sweeps": gen_seed,
    },
    "adversarial": {
        "hpo": adv_hpo,
        "seed_sweeps": adv_seed,
    },
}


def main():
    created = 0
    skipped = 0

    for arch in ARCHITECTURES:
        d, D = arch["d"], arch["D"]
        for emb in EMBEDDINGS:
            arch_dir = f"{emb}_d{d}D{D}"
            for trainer, config_types in GENERATORS.items():
                for config_type, gen_fn in config_types.items():
                    for dataset in DATASETS:
                        path = BASE / trainer / arch_dir / config_type / f"{dataset}.yaml"
                        if path.exists():
                            print(f"SKIP (exists): {path}")
                            skipped += 1
                            continue
                        path.parent.mkdir(parents=True, exist_ok=True)
                        content = gen_fn(emb, d, D, dataset)
                        path.write_text(content)
                        print(f"CREATED: {path}")
                        created += 1

    print(f"\nDone: {created} created, {skipped} skipped")


if __name__ == "__main__":
    main()
