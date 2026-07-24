"""Hydra entry point for the isolated MLP/JEM baseline."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.datahandler import DataHandler
from src.utils.train import OptimizerConfig, set_seed

from .device import resolve_device
from .model import JEMMLP, JEMMLPConfig, mps_parameter_count
from .sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from .trainer import (
    AdversarialTrainer,
    AdversarialTrainerConfig,
    NaturalTrainer,
    NaturalTrainerConfig,
    make_json_logger,
)


def _outputs_root() -> str:
    root = os.environ.get("BM4TC_DATA_ROOT")
    return f"{root}/outputs" if root else "outputs"


if not OmegaConf.has_resolver("jem_outputs_root"):
    OmegaConf.register_new_resolver("jem_outputs_root", _outputs_root)


def _optimizer(raw) -> OptimizerConfig:
    return OptimizerConfig(
        name=str(raw.name), kwargs=OmegaConf.to_container(raw.kwargs, resolve=True)
    )


def _natural_cfg(raw) -> NaturalTrainerConfig:
    return NaturalTrainerConfig(
        alpha=float(raw.alpha),
        max_epoch=int(raw.max_epoch),
        batch_size=int(raw.batch_size),
        patience=int(raw.patience),
        stop_crit=str(raw.stop_crit),
        input_noise_std=float(raw.input_noise_std),
        energy_l2=float(raw.energy_l2),
        grad_clip=None if raw.grad_clip is None else float(raw.grad_clip),
        save=bool(raw.save),
        optimizer=_optimizer(raw.optimizer),
    )


def _adversarial_cfg(raw) -> AdversarialTrainerConfig:
    return AdversarialTrainerConfig(
        max_epoch=int(raw.max_epoch),
        batch_size=int(raw.batch_size),
        patience=int(raw.patience),
        clean_weight=float(raw.clean_weight),
        epsilon=float(raw.epsilon),
        attack_steps=int(raw.attack_steps),
        attack_step_size=(
            None if raw.attack_step_size is None else float(raw.attack_step_size)
        ),
        eval_rob_freq=int(raw.eval_rob_freq),
        grad_clip=None if raw.grad_clip is None else float(raw.grad_clip),
        save=bool(raw.save),
        optimizer=_optimizer(raw.optimizer),
    )


def _make_model(cfg, data_dim: int, num_classes: int, device) -> JEMMLP:
    model_path = cfg.get("model_path")
    if model_path:
        model, _ = JEMMLP.load(model_path, device=device)
        if model.data_dim != data_dim or model.out_dim != num_classes:
            raise ValueError("Checkpoint dimensions do not match the configured dataset.")
        return model
    hidden = tuple(int(v) for v in cfg.model.hidden_dims)
    model_cfg = JEMMLPConfig(
        input_dim=data_dim,
        hidden_dims=hidden,
        num_classes=num_classes,
        activation=str(cfg.model.activation),
        input_range=tuple(float(v) for v in cfg.model.input_range),
    )
    return JEMMLP(model_cfg).to(device)


def _init_wandb(cfg, output_dir: Path):
    mode = str(cfg.tracking.mode)
    if mode == "disabled":
        return None
    return wandb.init(
        project=str(cfg.tracking.project),
        entity=str(cfg.tracking.entity),
        mode=mode,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        group=str(cfg.run_name),
        name=str(cfg.tracking.seed),
        reinit="finish_previous",
    )


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    device = resolve_device(str(cfg.device))
    set_seed(int(cfg.tracking.seed))

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    model = _make_model(cfg, datahandler.data_dim, datahandler.num_cls, device)
    datahandler.split_and_rescale(model)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    run = _init_wandb(cfg, output_dir)
    callback = make_json_logger(output_dir, run)

    expected_mps = mps_parameter_count(
        data_dim=datahandler.data_dim,
        physical_dim=int(cfg.model.match_mps.physical_dim),
        bond_dim=int(cfg.model.match_mps.bond_dim),
        num_classes=datahandler.num_cls,
    )
    metadata = {
        "model/parameters": model.count_parameters(),
        "model/matched_mps_complex_parameters": expected_mps,
    }
    callback(0, metadata)

    if str(cfg.regime) == "natural":
        train_cfg = _natural_cfg(cfg.trainer)
        datahandler.get_classification_loaders(train_cfg.batch_size)
        sgld_cfg = SGLDConfig(
            num_steps=int(cfg.sampler.num_steps),
            step_size=float(cfg.sampler.step_size),
            noise_std=float(cfg.sampler.noise_std),
            reinit_probability=float(cfg.sampler.reinit_probability),
            buffer_size=int(cfg.sampler.buffer_size),
        )
        buffer = ReplayBuffer(
            sgld_cfg.buffer_size,
            datahandler.data_dim,
            model.input_range,
            seed=int(cfg.tracking.seed),
        )
        sampler = SGLDSampler(sgld_cfg, buffer)
        trainer = NaturalTrainer(model, train_cfg, datahandler, sampler, device)
    elif str(cfg.regime) == "adversarial":
        train_cfg = _adversarial_cfg(cfg.trainer)
        datahandler.get_classification_loaders(train_cfg.batch_size)
        trainer = AdversarialTrainer(model, train_cfg, datahandler, device)
    else:
        raise ValueError(f"Unknown regime {cfg.regime!r}")

    trainer.train(callback=callback, output_dir=output_dir / "models")
    if run is not None:
        run.finish()

    if str(cfg.regime) == "adversarial":
        return -float(trainer.best["rob"])
    return -float(trainer.best["acc"])


if __name__ == "__main__":
    main()
