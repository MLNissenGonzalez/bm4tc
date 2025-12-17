# Copilot instructions — Adversarial Training with MPS

Purpose: provide concise, actionable notes so an AI coding agent can be productive quickly.

- Quick start
  - Create the conda env: `conda env create -f environment.yml` (root of repo).
  - Configs are Hydra-based under `configs/`. Example run pattern from the repository README:
    - Single run: `python -m experiments.<script> +experiments=<experiment>`
    - Multirun: `python -m experiments.<script> --multirun +experiments=<experiment>`
  - Hydra writes outputs to the run dir (see `configs/config.yaml` -> `hydra.run.dir`); tracking/logs and saved models are written there.

- Big-picture architecture (what to inspect first)
  - `src/models/` — MPS (`BornMachine`) and discriminator/critic implementations; the BornMachine is the probabilistic generator backbone.
  - `src/trainer/` — training logic: `classification.py`, `gantrain.py` (GAN-style), and `adversarial.py` (attack-aware training).
  - `src/data/` — data generation and `DataHandler` that provides loaders used by trainers.
  - `src/tracking/` — `wandb_utils.py` wraps W&B initialization, logging, and model saving using Hydra runtime paths.
  - `configs/` — Hydra config groups (models, trainer, dataset, tracking). The runtime config schema is registered in `src/utils/schemas.py`.

- Project-specific conventions and patterns
  - Hydra-driven configuration: code expects a structured `Config` object (see `src/utils/schemas.py`). Use Hydra overrides (e.g. `+trainer.ganstyle.max_epoch=10`) rather than editing files.
  - Model saving & logging: use `hydra.core.hydra_config.HydraConfig.get().runtime.output_dir` as the canonical run directory; `src/tracking/wandb_utils.py` and trainers use this.
  - Optimizers, embeddings, and criteria are obtained via `src/utils/getters.py` (e.g., `get.optimizer(...)`, `get.embedding(...)`, `get.criterion(...)`) — prefer these helpers rather than instantiating directly to keep behavior consistent.
  - MPS-specific losses: uses `MPSNLLL` (in `src/utils/getters.py`) rather than standard PyTorch NLL in some places — inspect callers when changing loss logic.
  - Sampling and batch adaptation: trainers mutate sampling-related config values at runtime (see `Trainer` in `src/trainer/gantrain.py`) — be careful when refactoring to preserve this behavior.

- Integration points & external dependencies
  - Hydra (config management) — `configs/` and `src/utils/schemas.py` are the source of truth for config structure.
  - Weights & Biases (`wandb`) — `src/tracking/wandb_utils.py` centralizes init and logging. Models are logged with `wandb.log_model` after being saved to the Hydra run dir.
  - Tensor network library `tensorkrowch` (imported in `src/utils/getters.py`) — MPS models depend on this API.

- Useful files to open when making changes
  - [README.md](README.md)
  - [configs/config.yaml](configs/config.yaml)
  - [src/utils/schemas.py](src/utils/schemas.py)
  - [src/trainer/gantrain.py](src/trainer/gantrain.py)
  - [src/trainer/classification.py](src/trainer/classification.py)
  - [src/tracking/wandb_utils.py](src/tracking/wandb_utils.py)
  - [src/utils/getters.py](src/utils/getters.py)
  - [src/data/handler.py](src/data/handler.py)

- Quick editing guidelines (do not assume defaults)
  - Preserve Hydra group names and dataclass fields in `src/utils/schemas.py` when adding config keys — tests and Hydra binding rely on these names.
  - When changing training loops, keep the same WandB/hydra save semantics: write outputs to the Hydra run dir and call `wandb` helpers in `src/tracking`.
  - Prefer helper functions (`get.optimizer`, `get.embedding`) to keep behavior consistent across trainers.

- When you're unsure
  - Inspect `configs/` for example YAMLs that show expected fields and naming conventions (models, trainer, sampling groups).
  - Check `src/tracking/wandb_utils.py` to see how the runtime Hydra `output_dir` and job numbering are used — this is the canonical integration point for persistent outputs.

Please review and tell me which parts need more detail (example configs, exact run entry point, or additional file links) and I will iterate.
